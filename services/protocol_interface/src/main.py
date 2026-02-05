"""Protocol Interface FastAPI service with MCP endpoints."""

from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import Depends, FastAPI, Header, HTTPException
import structlog

from shared.utils.logging_config import setup_logging

from .config import Settings
from .auth.provider_factory import get_auth_provider
from .a2a.handler import A2AHandler
from .mcp.models import MCPTool, MCPToolCallRequest
from .mcp.tool_registry import ToolRegistry
from .mcp.tool_handlers import MCPToolHandlers
from shared.models.a2a_models import A2AAgentDescriptor, A2ATaskRequest, A2ATaskResponse

import os


project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
log_file = os.path.join(project_root, "logs", "protocol-interface.log")
setup_logging(log_file=log_file)
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage service lifecycle."""
    app.state.settings = Settings()
    app.state.tool_registry = ToolRegistry()
    app.state.tool_handlers = MCPToolHandlers(app.state.settings)
    app.state.auth_provider = get_auth_provider(app.state.settings.auth_provider)
    app.state.a2a_handler = A2AHandler(app.state.settings)

    _register_tools(app.state.tool_registry, app.state.tool_handlers)

    logger.info("Protocol Interface service started", version=app.version)
    yield
    logger.info("Protocol Interface service stopped")


app = FastAPI(
    title="LexiQuery Protocol Interface",
    description="MCP + A2A Protocol Service",
    version="1.0.0-phase4",
    lifespan=lifespan
)


def _register_tools(registry: ToolRegistry, handlers: MCPToolHandlers) -> None:
    registry.register(
        MCPTool(
            name="process_query",
            description="Process a natural language query through LexiQuery core engine",
            input_schema={
                "type": "object",
                "properties": {
                    "natural_language": {"type": "string"},
                    "platform": {"type": "string", "enum": ["kql", "spl", "sql"]},
                    "context": {"type": "object"}
                },
                "required": ["natural_language", "platform"]
            },
            tags=["core", "query"],
            handler=handlers.process_query
        )
    )

    registry.register(
        MCPTool(
            name="get_journey_status",
            description="Get journey status and cost summary",
            input_schema={
                "type": "object",
                "properties": {
                    "journey_id": {"type": "string", "format": "uuid"}
                },
                "required": ["journey_id"]
            },
            tags=["core", "journey"],
            handler=handlers.get_journey_status
        )
    )


def get_registry() -> ToolRegistry:
    return app.state.tool_registry


def get_auth_provider_dep():
    return app.state.auth_provider


async def require_auth(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(lambda: app.state.settings),
    auth_provider=Depends(get_auth_provider_dep)
) -> None:
    if not settings.auth_enabled:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = authorization.replace("Bearer", "").strip()
    if not await auth_provider.validate_token(token):
        raise HTTPException(status_code=403, detail="Invalid token")


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "protocol-interface",
        "version": app.version
    }


@app.post("/mcp/tools/list")
async def mcp_list_tools(
    registry: ToolRegistry = Depends(get_registry),
    _: None = Depends(require_auth)
) -> Dict[str, Any]:
    tools = registry.list_tools()
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
                "tags": tool.tags
            }
            for tool in tools
        ]
    }


@app.post("/mcp/tools/call")
async def mcp_call_tool(
    request: MCPToolCallRequest,
    registry: ToolRegistry = Depends(get_registry),
    _: None = Depends(require_auth)
) -> Dict[str, Any]:
    tool = registry.get(request.name)
    if not tool or not tool.handler:
        raise HTTPException(status_code=404, detail="Tool not found")

    try:
        response = await tool.handler(request.arguments)
        return response
    except Exception as exc:
        logger.error("MCP tool execution failed", tool=request.name, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/a2a/agents/discover", response_model=list[A2AAgentDescriptor])
async def discover_agents(_: None = Depends(require_auth)) -> list[A2AAgentDescriptor]:
    handler: A2AHandler = app.state.a2a_handler
    return await handler.discover_agents()


@app.post("/a2a/task/invoke", response_model=A2ATaskResponse)
async def invoke_a2a_task(
    agent: A2AAgentDescriptor,
    task: A2ATaskRequest,
    _: None = Depends(require_auth)
) -> A2ATaskResponse:
    handler: A2AHandler = app.state.a2a_handler
    return await handler.invoke_task(agent, task)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=Settings().service_port,
        log_level=Settings().log_level.lower()
    )
