"""Workflow and state management models."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any, List
from uuid import UUID

from pydantic import BaseModel, Field


class WorkflowState(str, Enum):
    """Workflow state enumeration."""
    
    PENDING = "pending"
    DISCOVERING = "discovering"
    PLANNING = "planning"
    GATHERING_KNOWLEDGE = "gathering_knowledge"
    GENERATING_QUERIES = "generating_queries"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"


class StateTransition(BaseModel):
    """Record of a state transition."""
    
    from_state: WorkflowState = Field(..., description="Previous state")
    to_state: WorkflowState = Field(..., description="New state")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    reason: Optional[str] = Field(None, description="Reason for transition")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "from_state": "pending",
                "to_state": "discovering",
                "timestamp": "2026-02-05T10:30:00Z",
                "reason": "Starting agent discovery"
            }
        }


class WorkflowError(BaseModel):
    """Record of an error during workflow execution."""
    
    error_type: str = Field(..., description="Exception type")
    message: str = Field(..., description="Error message")
    state: WorkflowState = Field(..., description="State when error occurred")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    stack_trace: Optional[str] = Field(None, description="Stack trace if available")
    retry_count: int = Field(0, description="Number of retries attempted")
    is_transient: bool = Field(False, description="Whether error is retryable")


class WorkflowStep(BaseModel):
    """Individual step in workflow plan."""
    
    step_id: str = Field(..., description="Unique step identifier")
    agent_type: str = Field(..., description="Type of agent (planner, knowledge, data)")
    agent_id: Optional[str] = Field(None, description="Specific agent instance ID")
    action: str = Field(..., description="Action to perform")
    depends_on: List[str] = Field(default_factory=list, description="Step IDs this depends on")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = Field(60, description="Timeout for this step")
    
    class Config:
        json_schema_extra = {
            "example": {
                "step_id": "step_1",
                "agent_type": "knowledge",
                "agent_id": "sop_knowledge_agent",
                "action": "search_procedures",
                "depends_on": [],
                "parameters": {"query": "failed login", "top_k": 5}
            }
        }


class WorkflowPlan(BaseModel):
    """Execution plan created by Planner agent."""
    
    plan_id: str = Field(..., description="Unique plan identifier")
    created_by: str = Field(..., description="Planner agent that created this")
    steps: List[WorkflowStep] = Field(..., description="Ordered workflow steps")
    estimated_duration_seconds: Optional[int] = Field(None)
    estimated_cost_usd: Optional[float] = Field(None)
    parallel_execution_groups: List[List[str]] = Field(
        default_factory=list,
        description="Groups of step IDs that can run in parallel"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "plan_id": "plan_123",
                "created_by": "workflow_planner_agent",
                "steps": [
                    {
                        "step_id": "step_1",
                        "agent_type": "knowledge",
                        "action": "search_procedures"
                    }
                ],
                "estimated_duration_seconds": 15,
                "estimated_cost_usd": 0.01
            }
        }
