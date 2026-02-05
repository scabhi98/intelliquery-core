"""LLM client for planner agent."""

import json
import time
from typing import Any, Dict, List

import ollama
import structlog

from .config import settings
from .models import QueryAnalysis, LLMUsage
from .prompts import QUERY_ANALYSIS_PROMPT

logger = structlog.get_logger()


class PlannerLLMClient:
    """Ollama-based LLM client for planner."""

    def __init__(self) -> None:
        self.ollama = ollama.Client(settings.ollama_host)
        self.model = settings.ollama_model

    async def analyze_query(
        self,
        natural_language: str,
        execution_context: Dict[str, Any]
    ) -> tuple[QueryAnalysis, LLMUsage]:
        """Analyze query with LLM and return structured analysis."""
        prompt = QUERY_ANALYSIS_PROMPT.format(
            natural_language=natural_language,
            platforms_summary=self._format_platforms(
                execution_context.get("available_agents", {}).get("data", [])
            ),
            schema_hints=json.dumps(execution_context.get("schema_hints") or {}, indent=2),
            user_context=json.dumps(execution_context.get("user_context") or {}, indent=2)
        )

        start_time = time.time()
        response = self.ollama.generate(
            model=self.model,
            prompt=prompt,
            format="json"
        )
        latency_ms = (time.time() - start_time) * 1000

        usage = LLMUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency_ms=latency_ms,
            model=self.model
        )

        response_text = response.get("response", "{}")
        try:
            analysis_data = json.loads(response_text)
            analysis = QueryAnalysis(**analysis_data)
            return analysis, usage
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse LLM JSON", error=str(exc))
            return self._fallback_analysis(natural_language), usage
        except Exception as exc:
            logger.error("LLM analysis failed", error=str(exc), exc_info=True)
            return self._fallback_analysis(natural_language), usage

    @staticmethod
    def _format_platforms(data_agents: List[Dict[str, Any]]) -> str:
        platforms: Dict[str, List[str]] = {}
        for agent in data_agents:
            platform = agent.get("platform") or "unknown"
            platforms.setdefault(platform, []).append(agent.get("agent_name") or agent.get("agent_id", "unknown"))
        return "\n".join(
            f"- {platform}: {', '.join(names)}"
            for platform, names in platforms.items()
        )

    @staticmethod
    def _fallback_analysis(query: str) -> QueryAnalysis:
        lower_query = query.lower()
        requires_error = any(word in lower_query for word in ["error", "fail", "exception", "issue", "problem"])
        return QueryAnalysis(
            intent="diagnostic",
            complexity="moderate",
            required_platforms=["kql"],
            confidence=0.5,
            reasoning="Fallback analysis (LLM unavailable or invalid output)",
            entities={"tables": [], "fields": [], "time_range": "unknown"},
            knowledge_requirements={
                "requires_sop": True,
                "requires_error_knowledge": requires_error,
                "requires_schema": False
            },
            query_characteristics={
                "is_aggregation": False,
                "is_time_series": False,
                "has_joins": False,
                "complexity_factors": []
            }
        )
