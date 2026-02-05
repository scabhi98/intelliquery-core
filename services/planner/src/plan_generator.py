"""Workflow plan generation logic."""

from typing import List
from uuid import uuid4

from shared.models.workflow_models import WorkflowPlan, WorkflowStep

from .config import settings
from .models import QueryAnalysis, SelectedAgents


class PlanGenerator:
    """Generates workflow plans from analysis and selected agents."""

    def generate_plan(
        self,
        analysis: QueryAnalysis,
        selected: SelectedAgents,
        natural_language: str
    ) -> WorkflowPlan:
        steps: List[WorkflowStep] = []

        for idx, agent_id in enumerate(selected.knowledge_agents):
            steps.append(
                WorkflowStep(
                    step_id=f"knowledge_{idx}",
                    agent_type="knowledge",
                    agent_id=agent_id,
                    action="retrieve_context",
                    depends_on=[],
                    parameters={
                        "query": natural_language,
                        "top_k": 5
                    },
                    timeout_seconds=settings.default_step_timeout_seconds
                )
            )

        knowledge_step_ids = [step.step_id for step in steps]
        for platform, agent_id in selected.data_agents.items():
            steps.append(
                WorkflowStep(
                    step_id=f"data_{platform}",
                    agent_type="data",
                    agent_id=agent_id,
                    action="generate_and_execute",
                    depends_on=knowledge_step_ids,
                    parameters={
                        "platform": platform,
                        "execute": False,
                        "use_knowledge": True,
                        "optimize": True,
                        "analysis": analysis.model_dump(mode="json")
                    },
                    timeout_seconds=settings.default_step_timeout_seconds
                )
            )

        plan = WorkflowPlan(
            plan_id=f"plan_{uuid4()}",
            created_by=settings.agent_id,
            steps=steps,
            estimated_duration_seconds=None,
            estimated_cost_usd=None,
            parallel_execution_groups=self._build_parallel_groups(steps),
            metadata={
                "analysis": analysis.model_dump(mode="json"),
                "agent_selection": selected.model_dump(mode="json")
            }
        )
        return plan

    @staticmethod
    def _build_parallel_groups(steps: List[WorkflowStep]) -> List[List[str]]:
        groups: List[List[str]] = []
        completed = set()
        while len(completed) < len(steps):
            ready = [
                step for step in steps
                if step.step_id not in completed
                and all(dep in completed for dep in step.depends_on)
            ]
            if not ready:
                break
            group = [step.step_id for step in ready]
            groups.append(group)
            completed.update(group)
        return groups
