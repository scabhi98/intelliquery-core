"""Planner Agent FastAPI entrypoint."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
import structlog

from shared.models.a2a_models import A2ATaskRequest, A2ATaskResponse
from shared.utils.logging_config import setup_logging

from .config import settings
from .service import PlannerService

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
log_file = os.path.join(project_root, "logs", "planner-agent.log")
setup_logging(log_file=log_file)
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = PlannerService()
    logger.info("Planner service started", agent_id=settings.agent_id)
    yield


app = FastAPI(
    title="Planner Agent",
    description="A2A agent for creating workflow plans",
    version=settings.agent_version,
    lifespan=lifespan
)


@app.get("/")
@app.get("/a2a/descriptor")
async def descriptor() -> dict:
    return {
        "agent_id": settings.agent_id,
        "agent_name": settings.agent_name,
        "agent_type": "planner",
        "version": settings.agent_version,
        "capabilities": ["create_plan", "optimize_plan"],
        "status": "operational",
        "endpoint": f"http://localhost:{settings.service_port}",
        "metadata": {"mock": False}
    }


@app.get("/health")
async def health_check() -> dict:
    return {
        "status": "healthy",
        "agent_id": settings.agent_id,
        "version": settings.agent_version
    }


@app.post("/a2a/task", response_model=A2ATaskResponse)
async def process_task(request: A2ATaskRequest) -> A2ATaskResponse:
    logger.info(
        "Received A2A task request",
        task_id=str(request.task_id),
        action=request.action,
        journey_id=str(request.journey_id)
    )

    service: PlannerService = app.state.service

    if request.action == "create_plan":
        return await service.create_plan(request)

    if request.action == "optimize_plan":
        return await service.optimize_plan(request)

    raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.service_port,
        reload=True,
        log_level=settings.log_level.lower()
    )
