"""Agent selection logic for planner."""

from typing import Any, Dict, List

from .models import QueryAnalysis, SelectedAgents


class AgentSelector:
    """Selects appropriate agents based on query analysis."""

    def select_agents(
        self,
        analysis: QueryAnalysis,
        available_agents: Dict[str, List[Dict[str, Any]]]
    ) -> SelectedAgents:
        knowledge_agents = available_agents.get("knowledge", [])
        data_agents = available_agents.get("data", [])

        selected_knowledge: List[str] = []
        selected_data: Dict[str, str] = {}
        reasoning_parts: List[str] = []

        requirements = analysis.knowledge_requirements or {}
        if requirements.get("requires_sop"):
            sop_agent = self._first_by_domain(knowledge_agents, "sop")
            if sop_agent:
                selected_knowledge.append(sop_agent["agent_id"])
                reasoning_parts.append("Included SOP knowledge")

        if requirements.get("requires_error_knowledge"):
            error_agent = self._first_by_domain(knowledge_agents, "errors")
            if error_agent:
                selected_knowledge.append(error_agent["agent_id"])
                reasoning_parts.append("Included error knowledge")

        for platform in analysis.required_platforms:
            agent = self._first_by_platform(data_agents, platform)
            if agent:
                selected_data[platform] = agent["agent_id"]
                reasoning_parts.append(f"Selected {platform} data agent")

        if not selected_data and data_agents:
            fallback_agent = data_agents[0]
            platform = fallback_agent.get("platform") or "kql"
            selected_data[platform] = fallback_agent["agent_id"]
            reasoning_parts.append("Fallback to first available data agent")

        return SelectedAgents(
            knowledge_agents=selected_knowledge,
            data_agents=selected_data,
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "Selected agents based on analysis"
        )

    @staticmethod
    def _first_by_domain(agents: List[Dict[str, Any]], domain: str) -> Dict[str, Any] | None:
        for agent in agents:
            if agent.get("knowledge_domain") == domain:
                return agent
        return None

    @staticmethod
    def _first_by_platform(agents: List[Dict[str, Any]], platform: str) -> Dict[str, Any] | None:
        for agent in agents:
            if agent.get("platform") == platform or agent.get("platform") is None:
                return agent
        return None
