"""Core A2A Orchestrator for coordinating agents."""

import asyncio
from typing import Any, Dict, List, Optional
from uuid import UUID

import httpx
import structlog

from shared.models.journey_models import JourneyContext
from shared.models.query_models import QueryRequest, QueryResponse
from shared.models.workflow_models import WorkflowState
from shared.models.a2a_models import (
    A2AAgentDescriptor,
    A2ATaskRequest,
    A2ATaskResponse,
    KnowledgeContext,
    QueryResult,
)
from shared.models.workflow_models import WorkflowPlan, WorkflowStep
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
        
        # Configuration
        self.max_retries = config.get("max_retries", 3)
        self.agent_timeout = config.get("agent_timeout_seconds", 60)
        self.protocol_interface_url = config.get("protocol_interface_url", "http://localhost:8001")
        self.max_concurrent_agents = config.get("max_concurrent_agents", 5)
        
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
        
        This queries the Protocol Interface agent registry for live discovery.
        """
        self.logger.info("Discovering A2A agents", journey_id=str(context.journey_id))

        agents = await self._discover_agents_via_protocol(context)

        planner_agents = [a for a in agents if a.agent_type == "planner"]
        knowledge_agents = [a for a in agents if a.agent_type == "knowledge"]

        platform = context.request.platform
        if platform:
            data_agents = [
                a for a in agents
                if a.agent_type == "data" and (a.platform is None or a.platform == platform)
            ]
        else:
            data_agents = [a for a in agents if a.agent_type == "data"]
        
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
        
        Invokes planner agent via Protocol Interface.
        """
        planner = self._select_agent_by_capability(
            context.discovered_agents.get("planner", []),
            required_capability="create_plan"
        )
        
        self.logger.info(
            "Invoking Planner agent",
            journey_id=str(context.journey_id),
            agent_id=planner.agent_id
        )
        
        # Build agent capabilities summary for planner
        available_agents = {
            "planner": [
                {"agent_id": a.agent_id, "agent_name": a.agent_name, "capabilities": a.capabilities}
                for a in context.discovered_agents.get("planner", [])
            ],
            "knowledge": [
                {"agent_id": a.agent_id, "agent_name": a.agent_name, "capabilities": a.capabilities, "knowledge_domain": a.metadata.get("knowledge_domain")}
                for a in context.discovered_agents.get("knowledge", [])
            ],
            "data": [
                {"agent_id": a.agent_id, "agent_name": a.agent_name, "capabilities": a.capabilities, "platform": a.platform}
                for a in context.discovered_agents.get("data", [])
            ]
        }
        
        # Build execution context summary
        execution_context = {
            "journey_id": str(context.journey_id),
            "current_state": context.current_state.value,
            "state_history_count": len(context.state_history),
            "available_agents": available_agents,
            "user_context": context.request.context,
            "schema_hints": context.request.schema_hints
        }
        
        task = A2ATaskRequest(
            journey_id=context.journey_id,
            agent_id=planner.agent_id,
            action="create_plan",
            parameters={
                "natural_language": context.request.natural_language,
                "execution_context": execution_context
            },
            timeout_seconds=self.agent_timeout
        )

        response = await self._invoke_agent_task(planner, task)
        if response.status != "success":
            raise Exception(response.error_message or "Planner agent failed")

        plan_payload = None
        if response.result:
            plan_payload = response.result.get("plan") or response.result.get("workflow_plan")

        if not plan_payload:
            raise Exception("Planner response missing workflow plan")

        context.workflow_plan = WorkflowPlan(**plan_payload)
        self._record_agent_cost("planner", planner.agent_id, response, context)
        
        self.logger.info(
            "Workflow plan created",
            journey_id=str(context.journey_id),
            steps_count=len(context.workflow_plan.steps),
            estimated_cost=context.workflow_plan.estimated_cost_usd
        )
    
    async def _invoke_knowledge_agents(
        self,
        context: JourneyContext
    ) -> None:
        """
        Invoke Knowledge Provider agents in parallel.
        
        Invokes knowledge agents via Protocol Interface.
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
        
        if not context.workflow_plan:
            raise Exception("Workflow plan not available before knowledge gathering")

        knowledge_steps = [
            step for step in context.workflow_plan.steps
            if step.agent_type == "knowledge"
        ]

        semaphore = asyncio.Semaphore(self.max_concurrent_agents)
        tasks = [
            self._invoke_knowledge_step(semaphore, context, step, knowledge_agents)
            for step in knowledge_steps
        ]

        results = await asyncio.gather(*tasks)

        for knowledge_context in results:
            if knowledge_context:
                context.knowledge_contexts[knowledge_context.source] = knowledge_context
        
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
        
        Invokes data agents via Protocol Interface.
        """
        data_agents = context.discovered_agents.get("data", [])
        
        if not data_agents:
            raise Exception("No data agents available")
        
        if not context.workflow_plan:
            raise Exception("Workflow plan not available before data generation")

        data_steps = [
            step for step in context.workflow_plan.steps
            if step.agent_type == "data"
        ]

        semaphore = asyncio.Semaphore(self.max_concurrent_agents)
        tasks = [
            self._invoke_data_step(semaphore, context, step, data_agents)
            for step in data_steps
        ]

        results = await asyncio.gather(*tasks)

        for query_result in results:
            if query_result:
                context.query_results[query_result.platform] = query_result
        
        self.logger.info(
            "Query generated",
            journey_id=str(context.journey_id),
            platforms=list(context.query_results.keys()),
            results_count=len(context.query_results)
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

    async def _discover_agents_via_protocol(
        self,
        context: JourneyContext
    ) -> List[A2AAgentDescriptor]:
        """Discover agents via the Protocol Interface registry."""
        url = f"{self.protocol_interface_url}/a2a/agents/discover"
        async with httpx.AsyncClient(timeout=self.agent_timeout) as client:
            response = await client.get(url)

        if response.status_code != 200:
            raise Exception(
                f"Agent discovery failed: {response.status_code} - {response.text}"
            )

        payload = response.json()
        agents = [A2AAgentDescriptor(**item) for item in payload]

        self.logger.info(
            "Agents discovered via protocol",
            journey_id=str(context.journey_id),
            total_agents=len(agents)
        )
        return agents

    async def _invoke_agent_task(
        self,
        agent: A2AAgentDescriptor,
        task: A2ATaskRequest
    ) -> A2ATaskResponse:
        """Invoke an agent via Protocol Interface."""
        url = f"{self.protocol_interface_url}/a2a/task/invoke"
        payload = {
            "agent": agent.model_dump(mode="json"),
            "task": task.model_dump(mode="json")
        }

        async with httpx.AsyncClient(timeout=self.agent_timeout) as client:
            response = await client.post(url, json=payload)

        if response.status_code != 200:
            raise Exception(
                f"A2A task invocation failed: {response.status_code} - {response.text}"
            )

        return A2ATaskResponse(**response.json())

    async def _invoke_knowledge_step(
        self,
        semaphore: asyncio.Semaphore,
        context: JourneyContext,
        step: WorkflowStep,
        agents: List[A2AAgentDescriptor]
    ) -> Optional[KnowledgeContext]:
        """Invoke a single knowledge step."""
        async with semaphore:
            agent = self._select_agent_for_step(step, agents)

            parameters = dict(step.parameters or {})
            parameters.setdefault("query", context.request.natural_language)

            task = A2ATaskRequest(
                journey_id=context.journey_id,
                agent_id=agent.agent_id,
                action=step.action,
                parameters=parameters,
                timeout_seconds=step.timeout_seconds or self.agent_timeout
            )

            response = await self._invoke_agent_task(agent, task)
            if response.status != "success":
                raise Exception(response.error_message or "Knowledge agent failed")

            knowledge_payload = None
            if response.result:
                knowledge_payload = response.result.get("knowledge_context")

            if not knowledge_payload:
                raise Exception("Knowledge response missing context")

            knowledge_context = KnowledgeContext(**knowledge_payload)
            self._record_agent_cost("knowledge", agent.agent_id, response, context)
            return knowledge_context

    async def _invoke_data_step(
        self,
        semaphore: asyncio.Semaphore,
        context: JourneyContext,
        step: WorkflowStep,
        agents: List[A2AAgentDescriptor]
    ) -> Optional[QueryResult]:
        """Invoke a single data step."""
        async with semaphore:
            agent = self._select_agent_for_step(step, agents)

            knowledge_payload = {
                source: kc.model_dump(mode="json")
                for source, kc in context.knowledge_contexts.items()
            }

            parameters = dict(step.parameters or {})
            if context.request.platform:
                parameters.setdefault("platform", context.request.platform)
            parameters.setdefault("natural_language", context.request.natural_language)
            parameters.setdefault("knowledge_context", knowledge_payload)

            task = A2ATaskRequest(
                journey_id=context.journey_id,
                agent_id=agent.agent_id,
                action=step.action,
                parameters=parameters,
                timeout_seconds=step.timeout_seconds or self.agent_timeout
            )

            response = await self._invoke_agent_task(agent, task)
            if response.status != "success":
                raise Exception(response.error_message or "Data agent failed")

            result_payload = None
            if response.result:
                result_payload = response.result.get("query_result")

            if not result_payload:
                raise Exception("Data response missing query result")

            query_result = QueryResult(**result_payload)
            self._record_agent_cost("data", agent.agent_id, response, context)
            return query_result

    def _select_agent_by_capability(
        self,
        agents: List[A2AAgentDescriptor],
        required_capability: str
    ) -> A2AAgentDescriptor:
        """Select an agent that supports a required capability."""
        if not agents:
            raise Exception("No agents available")

        for agent in agents:
            if required_capability in agent.capabilities:
                return agent

        return agents[0]

    def _select_agent_for_step(
        self,
        step: WorkflowStep,
        agents: List[A2AAgentDescriptor]
    ) -> A2AAgentDescriptor:
        """Select an agent for a workflow step."""
        if step.agent_id:
            for agent in agents:
                if agent.agent_id == step.agent_id:
                    return agent

        for agent in agents:
            if step.action in agent.capabilities:
                return agent

        if agents:
            return agents[0]

        raise Exception(f"No agent available for step {step.step_id}")

    def _record_agent_cost(
        self,
        agent_type: str,
        agent_id: str,
        response: A2ATaskResponse,
        context: JourneyContext
    ) -> None:
        """Record cost info from agent response."""
        cost_info = response.cost_info or {}
        agent_cost = AgentCostInfo(
            agent_id=agent_id,
            agent_type=agent_type,
            llm_calls=cost_info.get("llm_calls", 0),
            llm_tokens=cost_info.get("llm_tokens", 0),
            mcp_calls=cost_info.get("mcp_calls", 0),
            embedding_calls=cost_info.get("embedding_calls", 0),
            execution_time_ms=cost_info.get("execution_time_ms", response.execution_time_ms),
            cost_usd=cost_info.get("cost_usd", 0.0),
            metadata={"task_id": str(response.task_id)}
        )
        context.add_agent_cost(agent_cost)
