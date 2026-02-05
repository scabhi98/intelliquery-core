"""Planner service implementation."""

import time
from typing import Any, Dict

import structlog

from shared.models.a2a_models import A2ATaskRequest, A2ATaskResponse
from shared.models.workflow_models import WorkflowPlan

from .agent_selector import AgentSelector
from .llm_client import PlannerLLMClient
from .models import SelectedAgents
from .plan_generator import PlanGenerator

logger = structlog.get_logger()


class PlannerService:
    """Service that creates and optimizes workflow plans."""

    def __init__(self) -> None:
        self.llm_client = PlannerLLMClient()
        self.agent_selector = AgentSelector()
        self.plan_generator = PlanGenerator()

    async def create_plan(self, request: A2ATaskRequest) -> A2ATaskResponse:
        start_time = time.time()
        parameters = request.parameters or {}
        natural_language = parameters.get("natural_language", "")
        execution_context = parameters.get("execution_context", {})

        analysis, usage = await self.llm_client.analyze_query(
            natural_language=natural_language,
            execution_context=execution_context
        )

        available_agents = execution_context.get("available_agents", {})
        selected = self.agent_selector.select_agents(analysis, available_agents)

        plan = self.plan_generator.generate_plan(
            analysis=analysis,
            selected=selected,
            natural_language=natural_language
        )

        elapsed_ms = (time.time() - start_time) * 1000
        return A2ATaskResponse(
            task_id=request.task_id,
            journey_id=request.journey_id,
            agent_id=request.agent_id,
            status="success",
            result={
                "plan": plan.model_dump(mode="json"),
                "confidence": analysis.confidence,
                "reasoning": analysis.reasoning,
                "agent_selection": selected.model_dump(mode="json")
            },
            cost_info={
                "llm_calls": 1,
                "llm_tokens": usage.total_tokens,
                "execution_time_ms": elapsed_ms,
                "model": usage.model
            },
            execution_time_ms=elapsed_ms
        )

    async def optimize_plan(self, request: A2ATaskRequest) -> A2ATaskResponse:
        start_time = time.time()
        parameters = request.parameters or {}
        plan_payload = parameters.get("plan")

        if not plan_payload:
            elapsed_ms = (time.time() - start_time) * 1000
            return A2ATaskResponse(
                task_id=request.task_id,
                journey_id=request.journey_id,
                agent_id=request.agent_id,
                status="error",
                error_message="Missing plan payload",
                execution_time_ms=elapsed_ms
            )

        plan = WorkflowPlan(**plan_payload)
        elapsed_ms = (time.time() - start_time) * 1000
        return A2ATaskResponse(
            task_id=request.task_id,
            journey_id=request.journey_id,
            agent_id=request.agent_id,
            status="success",
            result={
                "plan": plan.model_dump(mode="json"),
                "optimizations_applied": ["none"],
                "improvement": 0.0
            },
            execution_time_ms=elapsed_ms
        )
