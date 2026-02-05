"""ChromaDB vector store integration."""

import json
import time
from typing import Dict, List, Optional
from uuid import uuid4

import chromadb
from chromadb.config import Settings as ChromaSettings
import structlog

from shared.models.sop_models import Procedure
from shared.utils.exceptions import VectorStoreException

from ..config import Settings
from .embedder import OllamaEmbedder


logger = structlog.get_logger()


class VectorStoreManager:
    """Vector store manager for SOP procedures."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = chromadb.PersistentClient(
            path=settings.chroma_db_path,
            settings=ChromaSettings(anonymized_telemetry=False, is_persistent=True)
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection
        )
        self.embedder = OllamaEmbedder(settings.ollama_host, settings.embedding_model)

    async def add_procedures(self, procedures: List[Procedure]) -> List[str]:
        """Add procedures to the vector store."""
        try:
            ids: List[str] = []
            documents: List[str] = []
            metadatas: List[Dict[str, str]] = []
            for procedure in procedures:
                doc_text = "\n".join([procedure.title] + procedure.steps)
                emb_id = str(uuid4())
                ids.append(emb_id)
                documents.append(doc_text)
                metadatas.append(
                    {
                        "procedure_id": procedure.id,
                        "document_id": procedure.document_id,
                        "title": procedure.title,
                        "procedure_json": procedure.model_dump_json()
                    }
                )

            embeddings = await self.embedder.embed_batch(documents)
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            return ids
        except Exception as exc:
            logger.error("Failed to add procedures", error=str(exc))
            raise VectorStoreException("Failed to add procedures", details={"error": str(exc)})

    async def search(self, query: str, top_k: int) -> List[Procedure]:
        """Search procedures by semantic similarity."""
        start = time.time()
        try:
            query_embedding = await self.embedder.embed_text(query)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["metadatas", "distances"]
            )
            procedures: List[Procedure] = []
            for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
                proc_json = metadata.get("procedure_json")
                procedure = Procedure.model_validate_json(proc_json)
                procedure.similarity_score = max(0.0, 1.0 - float(distance))
                procedures.append(procedure)

            logger.info("Vector search completed", elapsed_ms=(time.time() - start) * 1000)
            return procedures
        except Exception as exc:
            logger.error("Vector search failed", error=str(exc))
            raise VectorStoreException("Vector search failed", details={"error": str(exc)})

    def get_stats(self) -> Dict[str, int]:
        """Return vector store stats."""
        return {"collection_count": self.collection.count()}
