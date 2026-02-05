"""Query-related data models."""

from datetime import datetime
from typing import Optional, Literal, Dict, Any, List
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

from shared.models.cost_models import JourneyCostSummary
from shared.models.workflow_models import WorkflowPlan
from shared.models.a2a_models import KnowledgeContext, QueryResult


class QueryRequest(BaseModel):
    """Request model for query generation."""
    
    request_id: UUID = Field(default_factory=uuid4)
    natural_language: str = Field(
        ...,
        description="User's natural language query",
        min_length=1,
        max_length=2000
    )
    platform: Literal["kql", "spl", "sql"] = Field(
        ...,
        description="Target query platform"
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional context from SOP or previous queries"
    )
    schema_hints: Optional[Dict[str, Any]] = Field(
        None,
        description="GraphRAG schema information"
    )
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('natural_language')
    def validate_query(cls, v):
        if len(v.strip()) == 0:
            raise ValueError("Query cannot be empty")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "natural_language": "Show all failed login attempts in the last hour",
                "platform": "kql",
                "context": {"table": "SigninLogs"},
                "user_id": "user@example.com"
            }
        }


class QueryResponse(BaseModel):
    """Response model for comprehensive query workflow with A2A agents."""
    
    request_id: UUID = Field(..., description="Original request ID")
    journey_id: UUID = Field(..., description="Journey identifier")
    
    # Workflow information
    workflow_plan: Optional[WorkflowPlan] = Field(None, description="Execution plan from Planner")
    
    # Knowledge context from knowledge providers
    knowledge_context: Dict[str, KnowledgeContext] = Field(
        default_factory=dict,
        description="Context from knowledge provider agents (sop, errors, wiki, etc.)"
    )
    
    # Query results from data agents
    queries: Dict[str, str] = Field(
        default_factory=dict,
        description="Generated queries by platform"
    )
    query_results: Dict[str, QueryResult] = Field(
        default_factory=dict,
        description="Detailed query results by platform"
    )
    
    # Execution results (if queries were executed)
    results: Optional[Dict[str, Dict[str, Any]]] = Field(
        None,
        description="Execution results by platform"
    )
    
    # Aggregated confidence and metadata
    overall_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall confidence across all agents"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings from all agents"
    )
    
    # Cost information
    cost_summary: Optional[JourneyCostSummary] = Field(
        None,
        description="Journey-level cost summary"
    )
    
    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (agents_used, timing, etc.)"
    )
    generation_time_ms: Optional[float] = Field(
        ...,
        description="Time taken to generate query in milliseconds"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "123e4567-e89b-12d3-a456-426614174000",
                "query": "SigninLogs | where TimeGenerated > ago(1h) | where ResultType != 0",
                "platform": "kql",
                "confidence": 0.95,
                "explanation": "Query filters sign-in logs for failures in the last hour",
                "optimizations_applied": ["time_filter_optimization"],
                "warnings": [],
                "generation_time_ms": 245.5
            }
        }


class QueryValidation(BaseModel):
    """Query validation results."""
    
    is_valid: bool = Field(..., description="Whether query is valid")
    syntax_errors: List[str] = Field(
        default_factory=list,
        description="List of syntax errors"
    )
    semantic_errors: List[str] = Field(
        default_factory=list,
        description="List of semantic errors"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-critical warnings"
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Optimization suggestions"
    )
    estimated_performance: Optional[str] = Field(
        None,
        description="Estimated query performance (fast/medium/slow)"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "is_valid": True,
                "syntax_errors": [],
                "semantic_errors": [],
                "warnings": ["Query may return large result set"],
                "suggestions": ["Consider adding LIMIT clause"],
                "estimated_performance": "fast"
            }
        }
