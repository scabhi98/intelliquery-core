"""Core A2A Orchestrator for coordinating agents."""

import asyncio
from typing import Dict, List, Optional
from uuid import UUID
import structlog

from shared.models.journey_models import JourneyContext
from shared.models.query_models import QueryRequest, QueryResponse
from shared.models.workflow_models import WorkflowState, WorkflowError
from shared.models.a2a_models import A2AAgentDescriptor, A2ATaskRequest, A2ATaskResponse, KnowledgeContext
from shared.models.cost_models import AgentCostInfo


logger = structlog.get_logger()


class A2AOrchestrator:
    """
    Thin orchestration layer coordinating A2A agents via protocol.
    
    This class is responsible for:
    - Discovering available A2A agents (Planner, Knowledge Providers, Data Agents)
    - Coordinating agent invocations based on workflow plan
    - Aggregating results and costs from all agents
    - Managing journey lifecycle and state transitions
    """
    
    def __init__(self, config: dict):
        """
        Initialize A2A Orchestrator.
        
        Args:
            config: Configuration dictionary with agent endpoints and settings
        """
        self.config = config
        self.logger = logger.bind(component="A2AOrchestrator")
        
        # Active journeys (in-memory for Phase 2, could be Redis for production)
        self.active_journeys: Dict[UUID, JourneyContext] = {}
        
        # A2A client will be initialized properly in Phase 4
        self.a2a_client = None  # Placeholder for a2a-sdk-python client
        
        # Configuration
        self.max_retries = config.get("max_retries", 3)
        self.agent_timeout = config.get("agent_timeout_seconds", 60)
        
        self.logger.info("A2AOrchestrator initialized", config=config)
    
    async def orchestrate_query_workflow(
        self,
        request: QueryRequest,
        journey_id: UUID
    ) -> QueryResponse:
        """
        Main entry point - orchestrates complete query workflow via A2A agents.
        
        Workflow:
        1. Initialize journey context
        2. Discover required agents
        3. Invoke Planner agent
        4. Invoke Knowledge Provider agents (parallel)
        5. Invoke Data agents (based on plan)
        6. Aggregate results and costs
        7. Return comprehensive response
        
        Args:
            request: Query request from user
            journey_id: Unique journey identifier
            
        Returns:
            QueryResponse with results from all agents
            
        Raises:
            Exception: If workflow fails after all retries
        """
        self.logger.info(
            "Starting query workflow orchestration",
            journey_id=str(journey_id),
            query=request.natural_language[:100]
        )
        
        try:
            # Step 1: Initialize journey
            context = await self._initialize_journey(request, journey_id)
            
            # Step 2: Discover agents
            context.add_state_transition(
                WorkflowState.DISCOVERING,
                reason="Discovering A2A agents"
            )
            await self._discover_required_agents(context)
            
            # Step 3: Invoke Planner agent
            context.add_state_transition(
                WorkflowState.PLANNING,
                reason="Creating workflow plan"
            )
            await self._invoke_planner_agent(context)
            
            # Step 4: Invoke Knowledge Provider agents (parallel)
            context.add_state_transition(
                WorkflowState.GATHERING_KNOWLEDGE,
                reason="Gathering knowledge from providers"
            )
            await self._invoke_knowledge_agents(context)
            
            # Step 5: Invoke Data agents
            context.add_state_transition(
                WorkflowState.GENERATING_QUERIES,
                reason="Generating and executing queries"
            )
            await self._invoke_data_agents(context)
            
            # Step 6: Validate and aggregate
            context.add_state_transition(
                WorkflowState.VALIDATING,
                reason="Validating results"
            )
            await self._validate_results(context)
            
            # Step 7: Compile response
            response = await self._compile_response(context)
            
            # Mark journey as completed
            context.mark_completed()
            
            self.logger.info(
                "Query workflow completed successfully",
                journey_id=str(journey_id),
                total_cost=context.cost_summary.total_cost_usd
            )
            
            return response
            
        except Exception as e:
            self.logger.error(
                "Query workflow failed",
                journey_id=str(journey_id),
                error=str(e),
                exc_info=True
            )
            
            # Mark journey as failed
            if journey_id in self.active_journeys:
                context = self.active_journeys[journey_id]
                context.mark_failed(reason=str(e))
            
            raise
        
        finally:
            # Cleanup
            if journey_id in self.active_journeys:
                del self.active_journeys[journey_id]
    
    async def _initialize_journey(
        self,
        request: QueryRequest,
        journey_id: UUID
    ) -> JourneyContext:
        """Initialize journey context."""
        context = JourneyContext(
            journey_id=journey_id,
            request=request,
            user_id=request.user_id,
            session_id=request.session_id
        )
        
        # Update cost summary with journey_id
        context.cost_summary.journey_id = journey_id
        
        self.active_journeys[journey_id] = context
        
        self.logger.info(
            "Journey initialized",
            journey_id=str(journey_id),
            state=context.current_state
        )
        
        return context
    
    async def _discover_required_agents(
        self,
        context: JourneyContext
    ) -> None:
        """
        Discover available A2A agents.
        
        In Phase 2 (mock mode), this returns configured agent endpoints.
        In Phase 4, this will query A2A registry for live agent discovery.
        """
        self.logger.info("Discovering A2A agents", journey_id=str(context.journey_id))
        
        # Phase 2: Use configured endpoints
        planner_agents = [
            A2AAgentDescriptor(
                agent_id="mock_workflow_planner",
                agent_type="planner",
                agent_name="Mock Workflow Planner",
                endpoint=self.config.get("planner_agent_url", "http://localhost:9000"),
                capabilities=["create_plan"],
                version="1.0.0-mock"
            )
        ]
        
        knowledge_agents = [
            A2AAgentDescriptor(
                agent_id="mock_sop_knowledge",
                agent_type="knowledge",
                agent_name="Mock SOP Knowledge Agent",
                endpoint=self.config.get("sop_knowledge_url", "http://localhost:9010"),
                capabilities=["retrieve_sop_context"],
                version="1.0.0-mock"
            )
        ]
        
        # Discover data agents based on requested platform
        platform = context.request.platform
        data_agent_map = {
            "kql": A2AAgentDescriptor(
                agent_id="mock_kql_data",
                agent_type="data",
                agent_name="Mock KQL Data Agent",
                endpoint=self.config.get("kql_data_url", "http://localhost:9020"),
                capabilities=["generate_and_execute"],
                platform="kql",
                version="1.0.0-mock"
            ),
            "spl": A2AAgentDescriptor(
                agent_id="mock_spl_data",
                agent_type="data",
                agent_name="Mock SPL Data Agent",
                endpoint=self.config.get("spl_data_url", "http://localhost:9021"),
                capabilities=["generate_and_execute"],
                platform="spl",
                version="1.0.0-mock"
            ),
            "sql": A2AAgentDescriptor(
                agent_id="mock_sql_data",
                agent_type="data",
                agent_name="Mock SQL Data Agent",
                endpoint=self.config.get("sql_data_url", "http://localhost:9022"),
                capabilities=["generate_and_execute"],
                platform="sql",
                version="1.0.0-mock"
            )
        }
        
        data_agents = [data_agent_map.get(platform)] if platform in data_agent_map else []
        
        # Store discovered agents in context
        context.discovered_agents = {
            "planner": planner_agents,
            "knowledge": knowledge_agents,
            "data": data_agents
        }
        
        self.logger.info(
            "Agents discovered",
            journey_id=str(context.journey_id),
            planner_count=len(planner_agents),
            knowledge_count=len(knowledge_agents),
            data_count=len(data_agents)
        )
        
        # Validate minimum requirements
        if not planner_agents:
            raise Exception("No Planner agents available")
        if not knowledge_agents:
            self.logger.warning("No Knowledge Provider agents available")
        if not data_agents:
            raise Exception(f"No Data agents available for platform: {platform}")
    
    async def _invoke_planner_agent(
        self,
        context: JourneyContext
    ) -> None:
        """
        Invoke Planner agent to create workflow plan.
        
        Phase 2: Calls mock planner that returns a simple plan.
        Phase 4: Invokes real A2A planner agent.
        """
        planner = context.discovered_agents["planner"][0]
        
        self.logger.info(
            "Invoking Planner agent",
            journey_id=str(context.journey_id),
            agent_id=planner.agent_id
        )
        
        # Phase 2: Mock invocation (will be replaced with real A2A call)
        # For now, create a simple mock plan
        from shared.models.workflow_models import WorkflowPlan, WorkflowStep
        
        workflow_plan = WorkflowPlan(
            plan_id=f"plan_{context.journey_id}",
            created_by=planner.agent_id,
            steps=[
                WorkflowStep(
                    step_id="step_1",
                    agent_type="knowledge",
                    agent_id="mock_sop_knowledge",
                    action="retrieve_sop_context",
                    parameters={
                        "query": context.request.natural_language,
                        "top_k": 5
                    }
                ),
                WorkflowStep(
                    step_id="step_2",
                    agent_type="data",
                    agent_id=context.discovered_agents["data"][0].agent_id,
                    action="generate_and_execute",
                    depends_on=["step_1"],
                    parameters={
                        "platform": context.request.platform,
                        "execute": False  # Phase 2: Don't execute, just generate
                    }
                )
            ],
            estimated_duration_seconds=15,
            estimated_cost_usd=0.01
        )
        
        context.workflow_plan = workflow_plan
        
        # Mock cost info from planner
        planner_cost = AgentCostInfo(
            agent_id=planner.agent_id,
            agent_type="planner",
            llm_calls=1,
            llm_tokens=300,
            mcp_calls=0,
            embedding_calls=0,
            execution_time_ms=150.0,
            cost_usd=0.002
        )
        context.add_agent_cost(planner_cost)
        
        self.logger.info(
            "Workflow plan created",
            journey_id=str(context.journey_id),
            steps_count=len(workflow_plan.steps),
            estimated_cost=workflow_plan.estimated_cost_usd
        )
    
    async def _invoke_knowledge_agents(
        self,
        context: JourneyContext
    ) -> None:
        """
        Invoke Knowledge Provider agents in parallel.
        
        Phase 2: Calls mock knowledge agents.
        Phase 4: Invokes real A2A knowledge agents via MCP tools.
        """
        knowledge_agents = context.discovered_agents.get("knowledge", [])
        
        if not knowledge_agents:
            self.logger.warning("No knowledge agents to invoke")
            return
        
        self.logger.info(
            "Invoking Knowledge Provider agents",
            journey_id=str(context.journey_id),
            agent_count=len(knowledge_agents)
        )
        
        # Phase 2: Mock knowledge context
        from shared.models.sop_models import Procedure
        
        sop_knowledge = KnowledgeContext(
            source="sop",
            agent_id="mock_sop_knowledge",
            content={
                "procedures": [
                    {
                        "id": "sop_001",
                        "title": "Incident Response: Failed Logins",
                        "description": "Procedure for investigating failed login attempts",
                        "steps": [
                            "Check SigninLogs table for failed authentication events",
                            "Identify user accounts with multiple failures",
                            "Verify if failures are from known IP ranges",
                            "Lock account if suspicious activity detected"
                        ],
                        "relevant_tables": ["SigninLogs", "AuditLogs"],
                        "tags": ["security", "authentication", "incident"]
                    }
                ]
            },
            relevance_score=0.85
        )
        
        context.knowledge_contexts["sop"] = sop_knowledge
        
        # Mock cost from knowledge agent
        knowledge_cost = AgentCostInfo(
            agent_id="mock_sop_knowledge",
            agent_type="knowledge",
            llm_calls=0,
            llm_tokens=0,
            mcp_calls=1,  # MCP call to ChromaDB
            embedding_calls=1,
            execution_time_ms=450.0,
            cost_usd=0.003
        )
        context.add_agent_cost(knowledge_cost)
        
        self.logger.info(
            "Knowledge contexts retrieved",
            journey_id=str(context.journey_id),
            sources=list(context.knowledge_contexts.keys())
        )
    
    async def _invoke_data_agents(
        self,
        context: JourneyContext
    ) -> None:
        """
        Invoke Data agents for query generation/execution.
        
        Phase 2: Calls mock data agents that return template queries.
        Phase 4: Invokes real A2A data agents with LLM + GraphRAG + MCP execution.
        """
        data_agents = context.discovered_agents.get("data", [])
        
        if not data_agents:
            raise Exception("No data agents available")
        
        platform = context.request.platform
        data_agent = data_agents[0]  # Use first (and only in Phase 2) data agent
        
        self.logger.info(
            "Invoking Data agent",
            journey_id=str(context.journey_id),
            agent_id=data_agent.agent_id,
            platform=platform
        )
        
        # Phase 2: Mock query result
        query_templates = {
            "kql": f"SigninLogs\n| where TimeGenerated > ago(1h)\n| where Status == 'Failure'\n| project TimeGenerated, UserPrincipalName, IPAddress, Location\n| take 100\n// Query generated for: {context.request.natural_language[:50]}",
            "spl": f"index=main sourcetype=authentication Status=\"Failure\" earliest=-1h\n| stats count by user, src_ip\n| sort -count\n# Query generated for: {context.request.natural_language[:50]}",
            "sql": f"SELECT timestamp, username, ip_address, status\nFROM authentication_logs\nWHERE status = 'FAILURE'\n  AND timestamp > NOW() - INTERVAL '1 HOUR'\nORDER BY timestamp DESC\nLIMIT 100\n-- Query generated for: {context.request.natural_language[:50]}"
        }
        
        query = query_templates.get(platform, f"-- Platform {platform} not implemented in mock")
        
        from shared.models.a2a_models import QueryResult
        
        query_result = QueryResult(
            platform=platform,
            agent_id=data_agent.agent_id,
            query=query,
            confidence=0.75,  # Mock confidence
            explanation=f"Mock query for {platform} platform based on template",
            optimizations=["mock_template_based"],
            data=None,  # Phase 2: No execution
            row_count=None,
            execution_time_ms=None,
            cost_info={
                "llm_tokens": 450,
                "mcp_calls": 0,  # Not executing in Phase 2
                "cost_usd": 0.005
            },
            warnings=[],
            metadata={"mock": True, "phase": 2}
        )
        
        context.query_results[platform] = query_result
        
        # Add data agent cost
        data_cost = AgentCostInfo(
            agent_id=data_agent.agent_id,
            agent_type="data",
            llm_calls=1,
            llm_tokens=450,
            mcp_calls=0,
            embedding_calls=0,
            execution_time_ms=780.0,
            cost_usd=0.005
        )
        context.add_agent_cost(data_cost)
        
        self.logger.info(
            "Query generated",
            journey_id=str(context.journey_id),
            platform=platform,
            confidence=query_result.confidence
        )
    
    async def _validate_results(
        self,
        context: JourneyContext
    ) -> None:
        """Validate all agent results."""
        self.logger.info(
            "Validating results",
            journey_id=str(context.journey_id)
        )
        
        # Basic validation
        if not context.workflow_plan:
            raise Exception("No workflow plan created")
        
        if not context.query_results:
            raise Exception("No queries generated")
        
        # Check confidence thresholds
        for platform, result in context.query_results.items():
            if result.confidence < 0.5:
                context.metadata.setdefault("warnings", []).append(
                    f"Low confidence for {platform}: {result.confidence}"
                )
        
        self.logger.info(
            "Validation completed",
            journey_id=str(context.journey_id),
            queries_count=len(context.query_results)
        )
    
    async def _compile_response(
        self,
        context: JourneyContext
    ) -> QueryResponse:
        """Compile comprehensive response from all agent results."""
        self.logger.info(
            "Compiling response",
            journey_id=str(context.journey_id)
        )
        
        # Extract queries
        queries = {
            platform: result.query
            for platform, result in context.query_results.items()
        }
        
        # Calculate overall confidence
        confidences = [r.confidence for r in context.query_results.values()]
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Compile metadata
        metadata = {
            "agents_used": list(context.cost_summary.cost_by_agent.keys()),
            "workflow_plan_id": context.workflow_plan.plan_id if context.workflow_plan else None,
            "execution_time_ms": (context.updated_at - context.created_at).total_seconds() * 1000,
            "state_transitions": len(context.state_history)
        }
        metadata.update(context.metadata)
        
        response = QueryResponse(
            request_id=context.request.request_id,
            journey_id=context.journey_id,
            workflow_plan=context.workflow_plan,
            knowledge_context=context.knowledge_contexts,
            queries=queries,
            query_results=context.query_results,
            results=None,  # Phase 2: No execution results
            overall_confidence=overall_confidence,
            warnings=metadata.get("warnings", []),
            cost_summary=context.cost_summary,
            metadata=metadata,
            generation_time_ms=0
        )
        
        self.logger.info(
            "Response compiled",
            journey_id=str(context.journey_id),
            overall_confidence=overall_confidence,
            total_cost=context.cost_summary.total_cost_usd
        )
        
        return response
