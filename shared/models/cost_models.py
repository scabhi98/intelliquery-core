"""Cost tracking related data models."""

from datetime import datetime
from typing import Dict, Any, Optional, List, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class AgentCostInfo(BaseModel):
    """Cost information from a single A2A agent execution."""
    
    agent_id: str = Field(..., description="Agent identifier")
    agent_type: str = Field(..., description="Agent type (planner, knowledge, data)")
    llm_calls: int = Field(0, description="Number of LLM calls made")
    llm_tokens: int = Field(0, description="Total LLM tokens used")
    mcp_calls: int = Field(0, description="Number of MCP tool calls")
    embedding_calls: int = Field(0, description="Number of embedding operations")
    execution_time_ms: float = Field(..., description="Total execution time")
    cost_usd: float = Field(..., description="Estimated cost in USD")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelUsage(BaseModel):
    """Track individual model invocation usage."""
    
    usage_id: UUID = Field(default_factory=uuid4)
    journey_id: UUID = Field(..., description="User journey/session ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Model information
    model_type: Literal["llm", "embedding"] = Field(...)
    model_name: str = Field(..., description="e.g., llama3, nomic-embed-text")
    
    # Usage metrics
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)
    
    # Compute metrics
    latency_ms: float = Field(..., description="Response time in milliseconds")
    compute_units: float = Field(..., description="Normalized compute cost")
    
    # Cost calculation
    estimated_cost_usd: float = Field(..., description="Estimated cost in USD")
    
    # Context
    service_name: str = Field(..., description="Service that made the call")
    operation: str = Field(..., description="e.g., query_generation, semantic_search")
    metadata: Optional[Dict[str, Any]] = Field(None)
    
    class Config:
        json_schema_extra = {
            "example": {
                "journey_id": "550e8400-e29b-41d4-a716-446655440000",
                "model_type": "llm",
                "model_name": "llama3",
                "prompt_tokens": 150,
                "completion_tokens": 75,
                "total_tokens": 225,
                "latency_ms": 1234.5,
                "compute_units": 1.2345,
                "estimated_cost_usd": 0.002,
                "service_name": "query-generator",
                "operation": "query_generation"
            }
        }


class JourneyCostSummary(BaseModel):
    """Aggregate cost summary for a complete journey with A2A and MCP tracking."""
    
    journey_id: UUID
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Aggregate metrics
    total_llm_calls: int = Field(default=0)
    total_embedding_calls: int = Field(default=0)
    total_mcp_calls: int = Field(default=0, description="Total MCP tool invocations")
    total_tokens: int = Field(default=0)
    total_compute_units: float = Field(default=0.0)
    total_cost_usd: float = Field(default=0.0)
    
    # Per-agent breakdown (A2A agents)
    cost_by_agent: Dict[str, AgentCostInfo] = Field(
        default_factory=dict,
        description="Cost breakdown by A2A agent"
    )
    
    # Per-service type breakdown
    cost_by_service_type: Dict[str, float] = Field(
        default_factory=dict,
        description="Cost by service type (llm, mcp, embedding)"
    )
    
    # Per-model breakdown
    cost_by_model: Dict[str, float] = Field(default_factory=dict)
    
    # Detailed usage records (optional, for detailed analysis)
    usage_records: List[ModelUsage] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "journey_id": "550e8400-e29b-41d4-a716-446655440000",
                "start_time": "2026-02-04T10:00:00Z",
                "end_time": "2026-02-04T10:05:30Z",
                "total_llm_calls": 4,
                "total_embedding_calls": 2,
                "total_mcp_calls": 3,
                "total_tokens": 1250,
                "total_compute_units": 6.5,
                "total_cost_usd": 0.014,
                "cost_by_agent": {
                    "planner_agent": {
                        "agent_id": "planner_agent",
                        "agent_type": "planner",
                        "llm_calls": 1,
                        "llm_tokens": 300,
                        "cost_usd": 0.002
                    },
                    "sop_knowledge_agent": {
                        "agent_id": "sop_knowledge_agent",
                        "agent_type": "knowledge",
                        "llm_calls": 0,
                        "mcp_calls": 1,
                        "cost_usd": 0.003
                    },
                    "kql_data_agent": {
                        "agent_id": "kql_data_agent",
                        "agent_type": "data",
                        "llm_calls": 1,
                        "mcp_calls": 1,
                        "cost_usd": 0.008
                    }
                },
                "cost_by_service_type": {
                    "llm": 0.010,
                    "mcp": 0.003,
                    "embedding": 0.001
                }
            }
        }
