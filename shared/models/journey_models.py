"""Journey context models for tracking workflow state."""

from datetime import datetime
from typing import Optional, Dict, List, Any
from uuid import UUID

from pydantic import BaseModel, Field

from shared.models.workflow_models import WorkflowState, StateTransition, WorkflowError, WorkflowPlan
from shared.models.query_models import QueryRequest
from shared.models.cost_models import JourneyCostSummary, AgentCostInfo
from shared.models.a2a_models import A2AAgentDescriptor, KnowledgeContext, QueryResult


class JourneyContext(BaseModel):
    """Thread-safe container for journey state across workflow."""
    
    # Journey identification
    journey_id: UUID = Field(..., description="Unique journey identifier")
    request: QueryRequest = Field(..., description="Original query request")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    
    # Workflow state
    current_state: WorkflowState = Field(
        default=WorkflowState.PENDING,
        description="Current workflow state"
    )
    state_history: List[StateTransition] = Field(
        default_factory=list,
        description="History of state transitions"
    )
    
    # Discovered agents
    discovered_agents: Dict[str, List[A2AAgentDescriptor]] = Field(
        default_factory=dict,
        description="Agents discovered by type (planner, knowledge, data)"
    )
    
    # Workflow plan
    workflow_plan: Optional[WorkflowPlan] = Field(
        None,
        description="Execution plan from Planner agent"
    )
    
    # Knowledge contexts from knowledge providers
    knowledge_contexts: Dict[str, KnowledgeContext] = Field(
        default_factory=dict,
        description="Knowledge contexts by source (sop, errors, wiki)"
    )
    
    # Query results from data agents
    query_results: Dict[str, QueryResult] = Field(
        default_factory=dict,
        description="Query results by platform (kql, spl, sql)"
    )
    
    # Cost tracking
    cost_summary: JourneyCostSummary = Field(
        default_factory=lambda: JourneyCostSummary(
            journey_id=UUID('00000000-0000-0000-0000-000000000000'),
            start_time=datetime.utcnow()
        ),
        description="Journey-level cost aggregation"
    )
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None)
    
    # Error tracking
    errors: List[WorkflowError] = Field(
        default_factory=list,
        description="All errors encountered during journey"
    )
    last_error: Optional[WorkflowError] = Field(
        None,
        description="Most recent error"
    )
    retry_count: int = Field(0, description="Number of retries attempted")
    
    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional journey metadata"
    )
    
    def add_state_transition(
        self,
        new_state: WorkflowState,
        reason: Optional[str] = None
    ) -> None:
        """Add a state transition to history."""
        transition = StateTransition(
            from_state=self.current_state,
            to_state=new_state,
            timestamp=datetime.utcnow(),
            reason=reason
        )
        self.state_history.append(transition)
        self.current_state = new_state
        self.updated_at = datetime.utcnow()
    
    def add_agent_cost(self, agent_cost: AgentCostInfo) -> None:
        """Add cost information from an agent."""
        # Add to cost by agent
        self.cost_summary.cost_by_agent[agent_cost.agent_id] = agent_cost
        
        # Update aggregates
        self.cost_summary.total_llm_calls += agent_cost.llm_calls
        self.cost_summary.total_mcp_calls += agent_cost.mcp_calls
        self.cost_summary.total_tokens += agent_cost.llm_tokens
        self.cost_summary.total_cost_usd += agent_cost.cost_usd
        
        # Update service type breakdown
        if agent_cost.llm_calls > 0:
            current_llm_cost = self.cost_summary.cost_by_service_type.get("llm", 0.0)
            # Estimate LLM portion of cost
            llm_cost = agent_cost.cost_usd * (agent_cost.llm_calls / max(1, agent_cost.llm_calls + agent_cost.mcp_calls))
            self.cost_summary.cost_by_service_type["llm"] = current_llm_cost + llm_cost
        
        if agent_cost.mcp_calls > 0:
            current_mcp_cost = self.cost_summary.cost_by_service_type.get("mcp", 0.0)
            # Estimate MCP portion of cost
            mcp_cost = agent_cost.cost_usd * (agent_cost.mcp_calls / max(1, agent_cost.llm_calls + agent_cost.mcp_calls))
            self.cost_summary.cost_by_service_type["mcp"] = current_mcp_cost + mcp_cost
        
        self.updated_at = datetime.utcnow()
    
    def add_error(self, error: WorkflowError) -> None:
        """Add an error to the journey context."""
        self.errors.append(error)
        self.last_error = error
        self.updated_at = datetime.utcnow()
    
    def mark_completed(self) -> None:
        """Mark journey as completed."""
        self.current_state = WorkflowState.COMPLETED
        self.completed_at = datetime.utcnow()
        self.cost_summary.end_time = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def mark_failed(self, reason: Optional[str] = None) -> None:
        """Mark journey as failed."""
        self.current_state = WorkflowState.FAILED
        self.completed_at = datetime.utcnow()
        self.cost_summary.end_time = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        if reason:
            self.metadata["failure_reason"] = reason
    
    class Config:
        # Allow arbitrary types for default_factory lambda
        arbitrary_types_allowed = True
