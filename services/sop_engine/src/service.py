"""SOP Engine service implementation."""

import time
from typing import Dict, List, Optional
from uuid import UUID

import structlog

from shared.interfaces.sop_engine import SOPEngineInterface
from shared.models.sop_models import SOPDocument, ProcessedSOP, Procedure
from shared.utils.exceptions import DocumentParsingException

from .config import Settings
from .parser.markdown_parser import MarkdownParser
from .parser.text_parser import TextParser
from .parser.docx_parser import DocxParser
from .parser.pdf_parser import PdfParser
from .parser.document_parser import DocumentParser
from .vector_store.vector_store import VectorStoreManager
from .search.search_engine import SemanticSearchEngine
from .schema.schema_analyzer import SchemaAnalyzer


logger = structlog.get_logger()


class SOPEngineService(SOPEngineInterface):
    """Real SOP Engine implementation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.vector_store = VectorStoreManager(settings)
        self.search_engine = SemanticSearchEngine(self.vector_store)
        self.schema_analyzer = SchemaAnalyzer()
        self.procedure_cache: Dict[str, Procedure] = {}
        self.parsers: Dict[str, DocumentParser] = {
            ".md": MarkdownParser(),
            ".markdown": MarkdownParser(),
            ".txt": TextParser(),
            ".docx": DocxParser(),
            ".pdf": PdfParser()
        }

    def _get_parser(self, document: SOPDocument) -> DocumentParser:
        extension = document.metadata.get("file_extension", ".txt")
        parser = self.parsers.get(extension)
        if not parser:
            return TextParser()
        return parser

    async def process_sop(self, document: SOPDocument, journey_id: UUID) -> ProcessedSOP:
        start_time = time.time()
        try:
            parser = self._get_parser(document)
            procedures = await parser.parse(document)

            for procedure in procedures:
                self.procedure_cache[procedure.id] = procedure

            embeddings = await self.vector_store.add_procedures(procedures)
            schema_source = document.content
            if document.metadata.get("file_extension") in {".docx", ".pdf"}:
                schema_source = "\n".join(
                    [proc.title for proc in procedures] +
                    [step for proc in procedures for step in proc.steps]
                )
            self.schema_analyzer.update_from_text(schema_source)

            elapsed_ms = (time.time() - start_time) * 1000
            return ProcessedSOP(
                document_id=document.id,
                status="processed",
                embedding_ids=embeddings,
                chunks_count=len(embeddings),
                entities_extracted=[entity["name"] for entity in self.schema_analyzer.get_context()["entities"]],
                relationships=self.schema_analyzer.get_context()["relationships"],
                processing_time_ms=elapsed_ms
            )
        except Exception as exc:
            logger.error("SOP processing failed", error=str(exc))
            raise DocumentParsingException("SOP processing failed", details={"error": str(exc)})

    async def search_procedures(
        self,
        query: str,
        journey_id: UUID,
        top_k: int = 5,
        filters: Optional[dict] = None
    ) -> List[Procedure]:
        return await self.search_engine.search(query, top_k=top_k, filters=filters)

    async def get_schema_context(self, platform: str, journey_id: UUID) -> dict:
        return self.schema_analyzer.get_context()

    async def health_check(self) -> dict:
        stats = self.vector_store.get_stats()
        return {
            "status": "healthy",
            "service": self.settings.service_name,
            "vector_store": stats,
            "embedding_model": self.settings.embedding_model
        }
