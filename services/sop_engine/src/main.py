"""SOP Engine FastAPI service."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
import structlog

from shared.models.sop_models import SOPDocument, ProcessedSOP, Procedure
from shared.utils.logging_config import setup_logging
from shared.utils.exceptions import LexiQueryException

from .config import Settings
from .service import SOPEngineService

import os
import tempfile


project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
log_file = os.path.join(project_root, "logs", "sop-engine.log")
setup_logging(log_file=log_file)
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = Settings()
    app.state.service = SOPEngineService(app.state.settings)
    logger.info("SOP Engine service started", version=app.version)
    yield
    logger.info("SOP Engine service stopped")


app = FastAPI(
    title="LexiQuery SOP Engine",
    description="SOP document ingestion and semantic search",
    version="1.0.0-phase4",
    lifespan=lifespan
)


def get_service() -> SOPEngineService:
    return app.state.service


def get_or_create_journey_id(x_journey_id: str = Header(None)) -> UUID:
    if x_journey_id:
        try:
            return UUID(x_journey_id)
        except ValueError:
            logger.warning("Invalid journey ID", value=x_journey_id)
    return uuid4()


@app.get("/health")
async def health_check(service: SOPEngineService = Depends(get_service)) -> Dict[str, Any]:
    return await service.health_check()


@app.post("/api/v1/procedures/ingest", response_model=ProcessedSOP)
async def ingest_sop_document(
    document: SOPDocument,
    journey_id: UUID = Depends(get_or_create_journey_id),
    service: SOPEngineService = Depends(get_service)
):
    try:
        document.metadata.setdefault("file_extension", ".txt")
        return await service.process_sop(document, journey_id)
    except LexiQueryException as exc:
        raise HTTPException(status_code=500, detail=exc.to_dict())


@app.post("/api/v1/procedures/upload", response_model=ProcessedSOP)
async def upload_sop_document(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    version: Optional[str] = Form("1.0"),
    author: Optional[str] = Form(None),
    department: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    journey_id: UUID = Depends(get_or_create_journey_id),
    service: SOPEngineService = Depends(get_service)
):
    try:
        file_bytes = await file.read()
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > service.settings.max_upload_size_mb:
            raise HTTPException(status_code=413, detail="File too large")

        extension = Path(file.filename or "").suffix.lower() or ".txt"
        document_title = title or Path(file.filename or "document").stem
        doc_id = document_id or str(uuid4())

        metadata = {
            "file_extension": extension,
            "original_filename": file.filename
        }

        tags_list = [tag.strip() for tag in (tags or "").split(",") if tag.strip()]

        temp_path = None
        content = ""
        if extension in {".docx", ".pdf"}:
            with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
                temp_file.write(file_bytes)
                temp_path = temp_file.name
            content = temp_path
        else:
            content = file_bytes.decode("utf-8", errors="ignore")

        document = SOPDocument(
            id=doc_id,
            title=document_title,
            content=content,
            version=version or "1.0",
            author=author,
            department=department,
            category=category,
            tags=tags_list,
            metadata=metadata
        )

        result = await service.process_sop(document, journey_id)
        return result
    except LexiQueryException as exc:
        raise HTTPException(status_code=500, detail=exc.to_dict())
    finally:
        if "temp_path" in locals() and temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning("Failed to remove temp file", path=temp_path)


@app.get("/api/v1/procedures/search", response_model=list[Procedure])
async def search_procedures(
    q: str,
    top_k: int = 5,
    journey_id: UUID = Depends(get_or_create_journey_id),
    service: SOPEngineService = Depends(get_service)
):
    return await service.search_procedures(q, journey_id, top_k=top_k)


@app.get("/api/v1/procedures/{procedure_id}", response_model=Procedure)
async def get_procedure(
    procedure_id: str,
    service: SOPEngineService = Depends(get_service)
):
    procedure = service.procedure_cache.get(procedure_id)
    if not procedure:
        raise HTTPException(status_code=404, detail="Procedure not found")
    return procedure


@app.get("/api/v1/schema")
async def get_schema(
    platform: str = "unknown",
    journey_id: UUID = Depends(get_or_create_journey_id),
    service: SOPEngineService = Depends(get_service)
) -> Dict[str, Any]:
    return await service.get_schema_context(platform, journey_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=Settings().service_port,
        log_level=Settings().log_level.lower()
    )
