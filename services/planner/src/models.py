"""Planner agent data models."""

from typing import Dict, List, Literal, Optional, Any
from pydantic import BaseModel, Field


class QueryAnalysis(BaseModel):
    """Structured query analysis from LLM."""

    intent: Literal["diagnostic", "monitoring", "security", "analytics", "investigation"]
    complexity: Literal["simple", "moderate", "complex"]
    required_platforms: List[Literal["kql", "spl", "sql"]]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

    entities: Dict[str, Any] = Field(default_factory=dict)
    knowledge_requirements: Dict[str, bool] = Field(default_factory=dict)
    query_characteristics: Dict[str, Any] = Field(default_factory=dict)


class SelectedAgents(BaseModel):
    """Agents selected for workflow."""

    knowledge_agents: List[str] = Field(default_factory=list)
    data_agents: Dict[str, str] = Field(default_factory=dict)
    reasoning: str = Field(default="")


class LLMUsage(BaseModel):
    """LLM usage metrics for cost tracking."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    model: Optional[str] = None
