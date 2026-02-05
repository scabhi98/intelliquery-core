"""Mock A2A Planner Agent for Phase 2 Testing."""

from typing import Dict, List
from uuid import UUID, uuid4
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import structlog

from shared.models.workflow_models import WorkflowPlan, WorkflowStep
from shared.models.a2a_models import A2ATaskRequest, A2ATaskResponse
from shared.utils.logging_config import setup_logging

# Setup structured logging with file output
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
log_file = os.path.join(project_root, "logs", "planner-agent.log")
setup_logging(log_file=log_file)
logger = structlog.get_logger()


class PlannerConfig(BaseModel):
    """Configuration for mock planner."""
    agent_id: str = "mock_workflow_planner"
    agent_name: str = "Mock Workflow Planner Agent"
    version: str = "1.0.0-phase2-mock"


app = FastAPI(
    title="Mock Workflow Planner Agent",
    description="A2A agent for creating workflow plans (Phase 2 Mock)",
    version="1.0.0-phase2"
)

config = PlannerConfig()


@app.get("/")
async def root():
    """Agent information endpoint."""
    return {
        "agent_id": config.agent_id,
        "agent_name": config.agent_name,
        "agent_type": "planner",
        "version": config.version,
        "capabilities": ["create_plan", "optimize_plan"],
        "status": "operational",
        "mock": True
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agent_id": config.agent_id,
        "version": config.version
    }


@app.post("/a2a/task", response_model=A2ATaskResponse)
async def process_task(request: A2ATaskRequest):
    """
    Process A2A task request.
    
    This is the main A2A endpoint that receives task requests
    from the orchestrator and returns workflow plans.
    
    Args:
        request: A2A task request
        
    Returns:
        A2A task response with workflow plan
    """
    logger.info(
        "Received A2A task request",
        task_id=request.task_id,
        action=request.action,
        source_agent=request.source_agent_id
    )
    
    if request.action == "create_plan":
        return await create_workflow_plan(request)
    elif request.action == "optimize_plan":
        return await optimize_workflow_plan(request)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown action: {request.action}"
        )


async def create_workflow_plan(request: A2ATaskRequest) -> A2ATaskResponse:
    """
    Create a workflow plan based on the request.
    
    Phase 2: Returns simple predefined plans.
    Phase 4: Will use LLM to analyze request and create intelligent plans.
    """
    parameters = request.parameters or {}
    natural_language = parameters.get("natural_language", "")
    platform = parameters.get("platform", "kql")
    
    logger.info(
        "Creating workflow plan",
        query_preview=natural_language[:100],
        platform=platform
    )
    
    # Determine which knowledge agents to invoke based on query
    knowledge_steps = []
    
    # Always include SOP knowledge
    knowledge_steps.append(
        WorkflowStep(
            step_id="step_knowledge_sop",
            agent_type="knowledge",
            agent_id="mock_sop_knowledge",
            action="retrieve_sop_context",
            parameters={
                "query": natural_language,
                "top_k": 5
            },
            estimated_duration_seconds=2.0,
            estimated_cost_usd=0.003
        )
    )
    
    # Add error knowledge if query mentions errors or issues
    error_keywords = ["error", "fail", "issue", "problem", "bug", "exception"]
    if any(keyword in natural_language.lower() for keyword in error_keywords):
        knowledge_steps.append(
            WorkflowStep(
                step_id="step_knowledge_errors",
                agent_type="knowledge",
                agent_id="mock_error_knowledge",
                action="retrieve_error_context",
                parameters={
                    "query": natural_language,
                    "top_k": 3
                },
                estimated_duration_seconds=1.5,
                estimated_cost_usd=0.002
            )
        )
    
    # Data agent step - depends on knowledge steps
    data_step = WorkflowStep(
        step_id="step_generate_query",
        agent_type="data",
        agent_id=f"mock_{platform}_data",
        action="generate_and_execute",
        depends_on=[step.step_id for step in knowledge_steps],
        parameters={
            "platform": platform,
            "execute": False,  # Phase 2: Don't execute
            "use_knowledge": True,
            "optimize": True
        },
        estimated_duration_seconds=3.0,
        estimated_cost_usd=0.005
    )
    
    # Assemble complete plan
    all_steps = knowledge_steps + [data_step]
    
    plan = WorkflowPlan(
        plan_id=f"plan_{uuid4()}",
        created_by=config.agent_id,
        steps=all_steps,
        estimated_duration_seconds=sum(s.estimated_duration_seconds for s in all_steps),
        estimated_cost_usd=sum(s.estimated_cost_usd for s in all_steps),
        metadata={
            "query_keywords": natural_language.split()[:5],
            "platform": platform,
            "knowledge_sources": [s.agent_id for s in knowledge_steps]
        }
    )
    
    logger.info(
        "Workflow plan created",
        plan_id=plan.plan_id,
        steps_count=len(plan.steps),
        estimated_cost=plan.estimated_cost_usd
    )
    
    # Return A2A response
    return A2ATaskResponse(
        task_id=request.task_id,
        agent_id=config.agent_id,
        status="completed",
        result={
            "plan": plan.model_dump(),
            "confidence": 0.85,  # Mock confidence
            "reasoning": f"Created plan with {len(knowledge_steps)} knowledge sources and {platform} query generation"
        },
        cost_info={
            "llm_calls": 1,
            "llm_tokens": 250,  # Mock token count
            "mcp_calls": 0,
            "execution_time_ms": 150.0,
            "cost_usd": 0.002
        }
    )


async def optimize_workflow_plan(request: A2ATaskRequest) -> A2ATaskResponse:
    """
    Optimize an existing workflow plan.
    
    Phase 2: Returns same plan with minor tweaks.
    Phase 4: Will use LLM to analyze and optimize.
    """
    parameters = request.parameters or {}
    existing_plan = parameters.get("plan")
    
    logger.info(
        "Optimizing workflow plan",
        plan_id=existing_plan.get("plan_id") if existing_plan else None
    )
    
    # Phase 2: Just return the same plan
    return A2ATaskResponse(
        task_id=request.task_id,
        agent_id=config.agent_id,
        status="completed",
        result={
            "plan": existing_plan,
            "optimizations_applied": ["none (phase 2 mock)"],
            "improvement": 0.0
        },
        cost_info={
            "llm_calls": 0,
            "llm_tokens": 0,
            "mcp_calls": 0,
            "execution_time_ms": 50.0,
            "cost_usd": 0.0
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        
        port=9000,
        reload=True,
        log_level="info"
    )
