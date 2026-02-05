"""Core Engine FastAPI Service with MCP Server."""

from contextlib import asynccontextmanager
from typing import Dict
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
import structlog

from shared.models.query_models import QueryRequest, QueryResponse
from shared.models.cost_models import JourneyCostSummary
from shared.models.journey_models import JourneyContext
from shared.utils.logging_config import setup_logging
from shared.utils.exceptions import LexiQueryException

from .orchestrator import A2AOrchestrator
from .config import Settings

import os

# Setup structured logging with file output
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
log_file = os.path.join(project_root, "logs", "core-engine.log")
setup_logging(log_file=log_file)
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting Core Engine service", version=app.version)
    
    # Initialize orchestrator
    config = {
        "max_retries": app.state.settings.max_retries,
        "agent_timeout_seconds": app.state.settings.agent_timeout_seconds,
        "planner_agent_url": app.state.settings.planner_agent_url,
        "sop_knowledge_url": app.state.settings.sop_knowledge_url,
        "kql_data_url": app.state.settings.kql_data_url,
        "spl_data_url": app.state.settings.spl_data_url,
        "sql_data_url": app.state.settings.sql_data_url,
    }
    
    app.state.orchestrator = A2AOrchestrator(config)
    
    logger.info("Core Engine service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Core Engine service")


# Initialize FastAPI app
app = FastAPI(
    title="LexiQuery Core Engine",
    description="A2A Orchestrator for query intelligence workflow",
    version="1.0.0-phase2",
    lifespan=lifespan
)

# Load settings
app.state.settings = Settings()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_orchestrator() -> A2AOrchestrator:
    """Dependency to get orchestrator instance."""
    return app.state.orchestrator


def get_or_create_journey_id(
    x_journey_id: str = Header(None)
) -> UUID:
    """Get journey ID from header or create new one."""
    if x_journey_id:
        try:
            return UUID(x_journey_id)
        except ValueError:
            logger.warning("Invalid journey ID in header", value=x_journey_id)
    
    # Create new journey ID
    journey_id = uuid4()
    logger.info("Created new journey ID", journey_id=str(journey_id))
    return journey_id


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "LexiQuery Core Engine",
        "version": "1.0.0-phase2",
        "status": "operational",
        "architecture": "A2A + MCP"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "core-engine",
        "version": "1.0.0-phase2"
    }


@app.post("/api/v1/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    journey_id: UUID = Depends(get_or_create_journey_id),
    orchestrator: A2AOrchestrator = Depends(get_orchestrator)
):
    """
    Process query request through A2A orchestration.
    
    This is the main entry point for query processing:
    1. Orchestrator discovers A2A agents
    2. Planner agent creates workflow plan
    3. Knowledge agents retrieve context
    4. Data agents generate and execute queries
    5. Results are aggregated and returned
    
    Args:
        request: Query request with natural language query and platform
        journey_id: Journey identifier (from header or auto-generated)
        orchestrator: A2A orchestrator instance
        
    Returns:
        QueryResponse with comprehensive results from all agents
        
    Raises:
        HTTPException: If query processing fails
    """
    logger.info(
        "Processing query request",
        journey_id=str(journey_id),
        platform=request.platform,
        query_preview=request.natural_language[:100]
    )
    
    try:
        # Set request metadata
        request.request_id = str(uuid4())
        
        # Orchestrate workflow via A2A agents
        response = await orchestrator.orchestrate_query_workflow(
            request=request,
            journey_id=journey_id
        )
        
        logger.info(
            "Query processed successfully",
            journey_id=str(journey_id),
            request_id=request.request_id,
            queries_count=len(response.queries),
            total_cost=response.cost_summary.total_cost_usd
        )
        
        return response
        
    except LexiQueryException as e:
        logger.error(
            "Query processing failed",
            journey_id=str(journey_id),
            error=str(e),
            details=e.details
        )
        raise HTTPException(
            status_code=500,
            detail={"message": str(e), "details": e.details}
        )
    
    except Exception as e:
        logger.error(
            "Unexpected error during query processing",
            journey_id=str(journey_id),
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail={"message": "Internal server error", "error": str(e)}
        )


@app.get("/api/v1/journey/{journey_id}", response_model=JourneyContext)
async def get_journey_context(
    journey_id: UUID,
    orchestrator: A2AOrchestrator = Depends(get_orchestrator)
):
    """
    Get journey context and state.
    
    Args:
        journey_id: Journey identifier
        orchestrator: Orchestrator instance
        
    Returns:
        JourneyContext with complete state history and costs
        
    Raises:
        HTTPException: If journey not found
    """
    logger.info("Fetching journey context", journey_id=str(journey_id))
    
    if journey_id not in orchestrator.active_journeys:
        raise HTTPException(
            status_code=404,
            detail=f"Journey {journey_id} not found"
        )
    
    context = orchestrator.active_journeys[journey_id]
    return context


@app.get("/api/v1/journey/{journey_id}/cost", response_model=JourneyCostSummary)
async def get_journey_cost(
    journey_id: UUID,
    orchestrator: A2AOrchestrator = Depends(get_orchestrator)
):
    """
    Get journey cost summary.
    
    Args:
        journey_id: Journey identifier
        orchestrator: Orchestrator instance
        
    Returns:
        JourneyCostSummary with complete cost breakdown
        
    Raises:
        HTTPException: If journey not found
    """
    logger.info("Fetching journey cost", journey_id=str(journey_id))
    
    if journey_id not in orchestrator.active_journeys:
        raise HTTPException(
            status_code=404,
            detail=f"Journey {journey_id} not found"
        )
    
    context = orchestrator.active_journeys[journey_id]
    return context.cost_summary


# MCP Server Interface (Phase 2 placeholder)
@app.post("/mcp/tools/list")
async def mcp_list_tools():
    """
    MCP endpoint: List available tools.
    
    Phase 2: Returns basic tool definitions.
    Phase 4: Implements full MCP protocol.
    """
    return {
        "tools": [
            {
                "name": "process_query",
                "description": "Process a natural language query through LexiQuery engine",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "natural_language": {"type": "string"},
                        "platform": {"type": "string", "enum": ["kql", "spl", "sql"]},
                        "context": {"type": "object"}
                    },
                    "required": ["natural_language", "platform"]
                }
            },
            {
                "name": "get_journey_status",
                "description": "Get status and cost summary for a query journey",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "journey_id": {"type": "string", "format": "uuid"}
                    },
                    "required": ["journey_id"]
                }
            }
        ]
    }


@app.post("/mcp/tools/call")
async def mcp_call_tool(
    request: Dict,
    orchestrator: A2AOrchestrator = Depends(get_orchestrator)
):
    """
    MCP endpoint: Execute tool call.
    
    Phase 2: Basic implementation.
    Phase 4: Full MCP protocol with streaming.
    """
    tool_name = request.get("name")
    arguments = request.get("arguments", {})
    
    logger.info("MCP tool call", tool=tool_name, args=arguments)
    
    if tool_name == "process_query":
        # Convert to QueryRequest
        query_request = QueryRequest(
            natural_language=arguments["natural_language"],
            platform=arguments["platform"],
            context=arguments.get("context")
        )
        
        journey_id = uuid4()
        
        response = await orchestrator.orchestrate_query_workflow(
            request=query_request,
            journey_id=journey_id
        )
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Query processed successfully. Journey ID: {journey_id}"
                },
                {
                    "type": "resource",
                    "resource": {
                        "uri": f"lexi://journey/{journey_id}",
                        "mimeType": "application/json",
                        "text": response.json()
                    }
                }
            ]
        }
    
    elif tool_name == "get_journey_status":
        journey_id = UUID(arguments["journey_id"])
        
        if journey_id not in orchestrator.active_journeys:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Journey {journey_id} not found or completed"
                    }
                ],
                "isError": True
            }
        
        context = orchestrator.active_journeys[journey_id]
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Journey {journey_id} status: {context.current_state}"
                },
                {
                    "type": "resource",
                    "resource": {
                        "uri": f"lexi://journey/{journey_id}/cost",
                        "mimeType": "application/json",
                        "text": context.cost_summary.json()
                    }
                }
            ]
        }
    
    else:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Unknown tool: {tool_name}"
                }
            ],
            "isError": True
        }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        
        port=8000,
        reload=True,
        log_level="info"
    )
