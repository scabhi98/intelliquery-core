"""Integration tests for Phase 2 A2A orchestration workflow."""

import pytest
from uuid import uuid4

from shared.models.query_models import QueryRequest, QueryResponse
from shared.models.workflow_models import WorkflowState
from shared.models.journey_models import JourneyContext
from services.core_engine.src.orchestrator import A2AOrchestrator


@pytest.fixture
def orchestrator_config():
    """Create test orchestrator configuration."""
    return {
        "max_retries": 3,
        "agent_timeout_seconds": 30,
        "planner_agent_url": "http://localhost:9000",
        "sop_knowledge_url": "http://localhost:9010",
        "kql_data_url": "http://localhost:9020",
        "spl_data_url": "http://localhost:9021",
        "sql_data_url": "http://localhost:9022",
    }


@pytest.fixture
def orchestrator(orchestrator_config):
    """Create orchestrator instance."""
    return A2AOrchestrator(orchestrator_config)


@pytest.fixture
def sample_query_request():
    """Create sample query request."""
    return QueryRequest(
        natural_language="Show failed login attempts in the last hour",
        platform="kql",
        context={},
        user_id="test_user",
        session_id="test_session"
    )


@pytest.mark.asyncio
async def test_journey_initialization(orchestrator, sample_query_request):
    """Test journey context initialization."""
    journey_id = uuid4()
    
    context = await orchestrator._initialize_journey(
        sample_query_request,
        journey_id
    )
    
    assert context.journey_id == journey_id
    assert context.current_state == WorkflowState.PENDING
    assert context.request == sample_query_request
    assert journey_id in orchestrator.active_journeys


@pytest.mark.asyncio
async def test_agent_discovery(orchestrator, sample_query_request):
    """Test A2A agent discovery."""
    journey_id = uuid4()
    context = await orchestrator._initialize_journey(
        sample_query_request,
        journey_id
    )
    
    await orchestrator._discover_required_agents(context)
    
    assert "planner" in context.discovered_agents
    assert "knowledge" in context.discovered_agents
    assert "data" in context.discovered_agents
    
    assert len(context.discovered_agents["planner"]) > 0
    assert len(context.discovered_agents["knowledge"]) > 0
    assert len(context.discovered_agents["data"]) > 0
    
    # Verify correct data agent for platform
    data_agent = context.discovered_agents["data"][0]
    assert data_agent.platform == "kql"


@pytest.mark.asyncio
async def test_planner_invocation(orchestrator, sample_query_request):
    """Test workflow planner invocation."""
    journey_id = uuid4()
    context = await orchestrator._initialize_journey(
        sample_query_request,
        journey_id
    )
    
    await orchestrator._discover_required_agents(context)
    await orchestrator._invoke_planner_agent(context)
    
    assert context.workflow_plan is not None
    assert len(context.workflow_plan.steps) > 0
    assert context.workflow_plan.created_by == "mock_workflow_planner"
    
    # Verify cost tracking
    assert len(context.cost_summary.cost_by_agent) > 0
    assert "mock_workflow_planner" in context.cost_summary.cost_by_agent


@pytest.mark.asyncio
async def test_knowledge_agent_invocation(orchestrator, sample_query_request):
    """Test knowledge provider agent invocation."""
    journey_id = uuid4()
    context = await orchestrator._initialize_journey(
        sample_query_request,
        journey_id
    )
    
    await orchestrator._discover_required_agents(context)
    await orchestrator._invoke_planner_agent(context)
    await orchestrator._invoke_knowledge_agents(context)
    
    assert len(context.knowledge_contexts) > 0
    assert "sop" in context.knowledge_contexts
    
    sop_knowledge = context.knowledge_contexts["sop"]
    assert sop_knowledge.source == "sop"
    assert sop_knowledge.agent_id == "mock_sop_knowledge"
    assert sop_knowledge.relevance_score > 0


@pytest.mark.asyncio
async def test_data_agent_invocation(orchestrator, sample_query_request):
    """Test data agent invocation for query generation."""
    journey_id = uuid4()
    context = await orchestrator._initialize_journey(
        sample_query_request,
        journey_id
    )
    
    await orchestrator._discover_required_agents(context)
    await orchestrator._invoke_planner_agent(context)
    await orchestrator._invoke_knowledge_agents(context)
    await orchestrator._invoke_data_agents(context)
    
    assert len(context.query_results) > 0
    assert "kql" in context.query_results
    
    query_result = context.query_results["kql"]
    assert query_result.platform == "kql"
    assert query_result.query is not None
    assert len(query_result.query) > 0
    assert query_result.confidence > 0


@pytest.mark.asyncio
async def test_complete_workflow(orchestrator, sample_query_request):
    """Test complete end-to-end workflow orchestration."""
    journey_id = uuid4()
    
    response = await orchestrator.orchestrate_query_workflow(
        request=sample_query_request,
        journey_id=journey_id
    )
    
    # Verify response structure
    assert isinstance(response, QueryResponse)
    assert response.journey_id == journey_id
    assert response.workflow_plan is not None
    
    # Verify knowledge context
    assert len(response.knowledge_context) > 0
    
    # Verify queries generated
    assert len(response.queries) > 0
    assert "kql" in response.queries
    
    # Verify cost tracking
    assert response.cost_summary is not None
    assert response.cost_summary.total_cost_usd > 0
    assert len(response.cost_summary.cost_by_agent) > 0
    
    # Verify metadata
    assert "agents_used" in response.metadata
    assert "workflow_plan_id" in response.metadata
    
    # Verify journey cleanup
    assert journey_id not in orchestrator.active_journeys


@pytest.mark.asyncio
async def test_workflow_state_transitions(orchestrator, sample_query_request):
    """Test workflow state machine transitions."""
    journey_id = uuid4()
    
    context = await orchestrator._initialize_journey(
        sample_query_request,
        journey_id
    )
    
    # Initial state
    assert context.current_state == WorkflowState.PENDING
    
    # Discovery
    context.add_state_transition(WorkflowState.DISCOVERING, reason="Test")
    assert context.current_state == WorkflowState.DISCOVERING
    assert len(context.state_history) == 2  # PENDING + DISCOVERING
    
    # Planning
    context.add_state_transition(WorkflowState.PLANNING, reason="Test")
    assert context.current_state == WorkflowState.PLANNING
    
    # Knowledge gathering
    context.add_state_transition(WorkflowState.GATHERING_KNOWLEDGE, reason="Test")
    assert context.current_state == WorkflowState.GATHERING_KNOWLEDGE
    
    # Query generation
    context.add_state_transition(WorkflowState.GENERATING_QUERIES, reason="Test")
    assert context.current_state == WorkflowState.GENERATING_QUERIES
    
    # Validation
    context.add_state_transition(WorkflowState.VALIDATING, reason="Test")
    assert context.current_state == WorkflowState.VALIDATING
    
    # Completion
    context.mark_completed()
    assert context.current_state == WorkflowState.COMPLETED
    assert context.completed_at is not None


@pytest.mark.asyncio
async def test_cost_aggregation(orchestrator, sample_query_request):
    """Test cost tracking and aggregation."""
    journey_id = uuid4()
    
    response = await orchestrator.orchestrate_query_workflow(
        request=sample_query_request,
        journey_id=journey_id
    )
    
    cost_summary = response.cost_summary
    
    # Verify agent-level costs
    assert len(cost_summary.cost_by_agent) >= 3  # Planner + Knowledge + Data
    
    # Verify call counts
    assert cost_summary.total_llm_calls > 0
    assert cost_summary.total_mcp_calls > 0  # From knowledge agents
    
    # Verify cost calculation
    assert cost_summary.total_cost_usd > 0
    sum_by_agent = sum(cost_summary.cost_by_agent.values())
    assert abs(sum_by_agent - cost_summary.total_cost_usd) < 0.001  # Float precision


@pytest.mark.asyncio
async def test_multiple_platforms(orchestrator):
    """Test query generation for different platforms."""
    platforms = ["kql", "spl", "sql"]
    
    for platform in platforms:
        journey_id = uuid4()
        request = QueryRequest(
            natural_language="Show application errors",
            platform=platform,
            user_id="test_user"
        )
        
        response = await orchestrator.orchestrate_query_workflow(
            request=request,
            journey_id=journey_id
        )
        
        assert platform in response.queries
        assert response.query_results[platform].platform == platform


@pytest.mark.asyncio
async def test_error_keyword_detection(orchestrator):
    """Test that error keywords trigger error knowledge agent."""
    journey_id = uuid4()
    request = QueryRequest(
        natural_language="Show database connection errors and failures",
        platform="kql",
        user_id="test_user"
    )
    
    context = await orchestrator._initialize_journey(request, journey_id)
    await orchestrator._discover_required_agents(context)
    await orchestrator._invoke_planner_agent(context)
    
    # Check that workflow plan includes error knowledge step
    error_steps = [
        step for step in context.workflow_plan.steps
        if step.agent_id == "mock_error_knowledge"
    ]
    
    assert len(error_steps) > 0, "Error knowledge agent should be included for error-related queries"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
