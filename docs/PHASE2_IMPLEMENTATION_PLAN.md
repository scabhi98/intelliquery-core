# Phase 2: Core Engine Implementation Plan

**Version**: 1.0  
**Date**: February 5, 2026  
**Status**: Design Phase - Awaiting Review

---

## Executive Summary

Phase 2 focuses on implementing the Core Engine Service with a CrewAI-based orchestration system. This plan defines:
1. Class hierarchy and architecture
2. Orchestration algorithm and workflow state machine
3. Service interaction patterns and flow diagrams
4. Integration points with all dependent services

**Key Principle**: Interface-first design with mock services enables validation of orchestration logic before real service implementation.

---

## 1. Class Hierarchy & Architecture

### 1.1 Core Classes

```
LexiQueryCore (Thin Orchestration Layer)
    ├── MCPServer (MCP Protocol Server - External Interface)
    ├── A2AOrchestrator (A2A Agent Discovery & Invocation)
    ├── WorkflowStateMachine (Workflow State Management)
    ├── CostAggregator (Journey-level Cost Aggregation)
    └── ErrorHandler (Retry & Error Logic)

┌────────────────────────────────────────────────────────┐
│           A2A AGENT ECOSYSTEM (External)               │
└────────────────────────────────────────────────────────┘

Planner Agents (A2A) - Planning & Orchestration Logic
    ├── WorkflowPlannerAgent
    ├── MultiStepPlannerAgent
    └── AdaptivePlannerAgent

Knowledge Provider Agents (A2A) - Knowledge Retrieval
    ├── SOPKnowledgeAgent (connects to ChromaDB via MCP)
    ├── KnownErrorAgent (connects to Error DB via MCP)
    ├── WikiKnowledgeAgent (connects to Wiki via MCP)
    ├── DocumentationAgent (connects to Docs via MCP)
    └── RunbookAgent (connects to Runbook DB via MCP)

Data Agents (A2A) - Platform-Specific Query Generation
    ├── KQLDataAgent (Azure Log Analytics specialist)
    ├── SPLDataAgent (Splunk specialist)
    ├── SQLDataAgent (SQL databases specialist)
    └── PromeQLDataAgent (Prometheus specialist)

┌────────────────────────────────────────────────────────┐
│         MCP TOOL SERVERS (Service Layer)               │
└────────────────────────────────────────────────────────┘

MCP Tools accessed by A2A Agents:
    ├── ChromaDB_MCP_Server (Vector DB for SOPs/Docs)
    ├── ErrorDB_MCP_Server (Known Error Database)
    ├── AzureLogAnalytics_MCP_Server (KQL Executor)
    ├── Splunk_MCP_Server (SPL Executor)
    └── SQL_MCP_Server (SQL Executor)

CacheServiceInterface (Abstract)
    └── MockCacheService (Phase 2 Implementation)
         └── RealCacheService (Phase 4 Implementation)

DataConnectorInterface (Abstract)
    └── MockDataConnector (Phase 2 Implementation)
         └── RealDataConnector (Phase 4 Implementation)

ProtocolHandlerInterface (Abstract)
    └── MockProtocolHandler (Phase 2 Implementation)
         └── RealProtocolHandler (Phase 4 Implementation)
```

---

## 2. Detailed Class Specifications

### 2.1 A2AOrchestrator

**Purpose**: Thin orchestration layer coordinating A2A agents via protocol

**Attributes**:
```python
class A2AOrchestrator:
    # Configuration
    config: OrchestrationConfig
    state_machine: WorkflowStateMachine
    cost_aggregator: CostAggregator
    
    # A2A Protocol Client
    a2a_client: A2AClient  # From a2a-sdk-python
    a2a_registry: A2ARegistry  # Agent discovery
    
    # Required Agent Types (discovered at runtime)
    planner_agents: List[A2AAgentDescriptor]
    knowledge_agents: List[A2AAgentDescriptor]
    data_agents: List[A2AAgentDescriptor]
    
    # State
    active_journeys: Dict[UUID, JourneyContext]
    
    # Logging
    logger: structlog.BoundLogger
```

**Key Methods**:
```python
async def orchestrate_query_workflow(
    request: QueryRequest,
    journey_id: UUID
) -> QueryResponse:
    """Main entry point - delegates to A2A agents."""
    
async def discover_required_agents() -> Dict[str, List[A2AAgentDescriptor]]:
    """Discover available Planner, Knowledge Provider, and Data agents."""
    
async def invoke_planner_agent(
    request: QueryRequest,
    journey_id: UUID
) -> WorkflowPlan:
    """Invoke Planner agent to create execution plan."""
    
async def invoke_knowledge_agents(
    plan: WorkflowPlan,
    journey_id: UUID
) -> Dict[str, KnowledgeContext]:
    """Invoke relevant Knowledge Provider agents (SOPs, Errors, Wikis)."""
    
async def invoke_data_agents(
    plan: WorkflowPlan,
    knowledge_context: Dict[str, KnowledgeContext],
    journey_id: UUID
) -> Dict[str, QueryResult]:
    """Invoke platform-specific Data agents for query generation/execution."""
    
async def aggregate_results(
    plan: WorkflowPlan,
    knowledge_context: Dict,
    query_results: Dict
) -> QueryResponse:
    """Aggregate all agent results into final response."""
    
async def _handle_orchestration_error(
    error: Exception,
    context: JourneyContext,
    retry_count: int
) -> None:
    """Error handling with exponential backoff."""
```

---

### 2.2 WorkflowStateMachine

**Purpose**: Manages workflow state transitions and validations

**States**:
```
    ┌─────────────────────────────────────────────────┐
    │                WORKFLOW STATES                  │
    └─────────────────────────────────────────────────┘
    
    PENDING
       ↓
    ANALYZING (SOP Context Retrieval)
       ↓
    GENERATING (Query Generation)
       ↓
    VALIDATING (Query Validation)
       ↓
    OPTIMIZING (Query Optimization)
       ↓
    COMPLETED / FAILED
```

**Attributes**:
```python
class WorkflowStateMachine:
    current_state: WorkflowState
    state_history: List[StateTransition]
    journey_id: UUID
    started_at: datetime
    
    # State Handlers
    state_handlers: Dict[WorkflowState, StateHandler]
    
    # Transition Rules
    valid_transitions: Dict[WorkflowState, List[WorkflowState]]
    timeout_handlers: Dict[WorkflowState, Coroutine]
```

**Key Methods**:
```python
async def transition(
    new_state: WorkflowState,
    context: Dict[str, Any]
) -> bool:
    """Validate and execute state transition."""
    
async def handle_state_entry(state: WorkflowState) -> None:
    """Execute entry actions for state."""
    
async def handle_state_exit(state: WorkflowState) -> None:
    """Execute exit actions for state."""
    
async def rollback_to_state(state: WorkflowState) -> None:
    """Rollback workflow to previous state."""
    
def get_state_duration(state: WorkflowState) -> timedelta:
    """Get time spent in state."""
```

---

### 2.3 ServiceRegistry

**Purpose**: Service discovery, dependency injection, and service lifecycle management

**Attributes**:
```python
class ServiceRegistry:
    # Registered Services
    sop_engine: SOPEngineInterface
    query_generator: QueryGeneratorInterface
    cache_service: CacheServiceInterface
    data_connector: DataConnectorInterface
    protocol_handler: ProtocolHandlerInterface
    
    # Health Status
    service_health: Dict[str, ServiceHealthStatus]
    
    # Configuration
    service_urls: Dict[str, str]
    retry_policy: RetryPolicy
    timeout_policy: TimeoutPolicy
```

**Key Methods**:
```python
def register_service(
    service_name: str,
    service_instance: Any,
    health_check_url: Optional[str] = None
) -> None:
    """Register service with health check."""
    
def get_service(
    service_name: str
) -> Any:
    """Resolve service from registry."""
    
async def check_service_health(
    service_name: str
) -> ServiceHealthStatus:
    """Perform health check on service."""
    
async def initialize_all_services() -> None:
    """Initialize all registered services."""
    
async def shutdown_all_services() -> None:
    """Graceful shutdown of all services."""
    
def get_circuit_breaker(
    service_name: str
) -> CircuitBreaker:
    """Get circuit breaker for service."""
```

---

### 2.4 JourneyContext

**Purpose**: Thread-safe container for journey state across workflow

**Attributes**:
```python
class JourneyContext:
    journey_id: UUID
    request: QueryRequest
    user_id: Optional[str]
    session_id: Optional[str]
    
    # Workflow State
    current_state: WorkflowState
    state_history: List[StateTransition]
    
    # Data Pipeline
    sop_procedures: List[Procedure] = []
    analysis_result: Optional[AnalysisResult] = None
    generated_query: Optional[str] = None
    validation_result: Optional[QueryValidation] = None
    
    # Cost Tracking
    cost_summary: JourneyCostSummary
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    
    # Error Tracking
    errors: List[WorkflowError] = []
    last_error: Optional[WorkflowError] = None
```

---

### 2.5 A2A Agent Specifications

#### 2.5.1 Planner Agents (A2A)

**Agent**: WorkflowPlannerAgent
**Protocol**: A2A SDK Server
**Expertise**: Workflow planning, task decomposition, agent selection
**Location**: Independent microservice

**Responsibilities**:
- Analyze user's natural language query
- Determine required knowledge sources (SOPs, wikis, error DBs)
- Identify target data platforms (KQL, SPL, SQL)
- Create step-by-step execution plan
- Estimate cost and complexity

**A2A Interface**:
```python
class WorkflowPlannerAgent:
    @a2a_task
    async def create_plan(
        natural_language: str,
        user_context: dict,
        journey_id: str
    ) -> WorkflowPlan:
        # Returns structured plan with steps, agent assignments
        pass
```

**Internal Tools**: Uses own LLM for planning logic

---

#### 2.5.2 Knowledge Provider Agents (A2A)

**Agent**: SOPKnowledgeAgent
**Protocol**: A2A SDK Server
**Expertise**: SOP procedures, business processes, compliance
**Location**: Independent microservice

**Responsibilities**:
- Search SOP documents using semantic search
- Extract relevant procedures and steps
- Identify compliance requirements
- Map business context to technical requirements

**MCP Tool Integration**:
```python
class SOPKnowledgeAgent:
    mcp_client: MCPClient  # Connects to ChromaDB MCP server
    
    @a2a_task
    async def retrieve_sop_context(
        query: str,
        filters: dict,
        journey_id: str
    ) -> SOPContext:
        # Uses MCP to query ChromaDB vector store
        results = await self.mcp_client.call_tool(
            server="chromadb",
            tool="semantic_search",
            arguments={
                "collection": "sop_procedures",
                "query": query,
                "top_k": 5
            }
        )
        return SOPContext(procedures=results)
```

---

**Agent**: KnownErrorAgent
**Protocol**: A2A SDK Server
**Expertise**: Error patterns, resolutions, troubleshooting
**Location**: Independent microservice

**MCP Tool Integration**:
- Connects to ErrorDB MCP server
- Searches known error database
- Returns error patterns and resolutions

---

**Agent**: WikiKnowledgeAgent
**Protocol**: A2A SDK Server
**Expertise**: Documentation, how-to guides, architecture
**Location**: Independent microservice

**MCP Tool Integration**:
- Connects to Wiki MCP server (Confluence, Notion, etc.)
- Semantic search across documentation
- Returns relevant articles and guides

---

#### 2.5.3 Data Agents (A2A) - Query Generation & Execution

**Agent**: KQLDataAgent
**Protocol**: A2A SDK Server
**Expertise**: KQL syntax, Azure Log Analytics schema, query optimization
**Location**: Independent microservice

**Responsibilities**:
- Generate KQL queries from natural language + context
- Optimize queries for Azure Log Analytics performance
- Execute queries using MCP connection to Azure
- Return results with cost information

**MCP Tool Integration**:
```python
class KQLDataAgent:
    mcp_client: MCPClient  # Connects to Azure Log Analytics MCP server
    llm_client: OllamaClient  # Uses Ollama for query generation
    graphrag: GraphRAG  # Azure schema knowledge
    
    @a2a_task
    async def generate_and_execute(
        natural_language: str,
        sop_context: dict,
        schema_hints: dict,
        journey_id: str,
        execute: bool = False
    ) -> QueryResult:
        # Generate query using LLM + GraphRAG
        query = await self._generate_kql(natural_language, sop_context)
        
        # Execute using MCP if requested
        if execute:
            results = await self.mcp_client.call_tool(
                server="azure_log_analytics",
                tool="execute_query",
                arguments={
                    "workspace_id": "...",
                    "query": query,
                    "timespan": "PT1H"
                }
            )
            return QueryResult(query=query, data=results)
        
        return QueryResult(query=query)
```

---

**Agent**: SPLDataAgent
**Protocol**: A2A SDK Server
**Expertise**: SPL syntax, Splunk data models, search optimization
**Location**: Independent microservice

**MCP Tool Integration**:
- Connects to Splunk MCP server
- Generates SPL queries using specialized LLM prompts
- Executes searches via MCP tool
- Returns results with performance metrics

---

**Agent**: SQLDataAgent
**Protocol**: A2A SDK Server
**Expertise**: SQL dialects, database schemas, join optimization  
**Location**: Independent microservice

**MCP Tool Integration**:
- Connects to SQL database MCP servers
- Supports multiple dialects (PostgreSQL, MySQL, T-SQL)
- Generates and executes SQL queries
- Returns results with execution plans

---

### 2.6 MCP Server Specifications

#### 2.6.1 ChromaDB MCP Server

**Purpose**: Provide MCP tool interface to ChromaDB vector database
**Used By**: SOPKnowledgeAgent, WikiKnowledgeAgent, DocumentationAgent
**Location**: Separate MCP server microservice

**Tools**:
```python
@mcp_tool
async def semantic_search(
    collection: str,
    query: str,
    top_k: int = 5,
    filters: dict = None
) -> List[dict]:
    """Semantic search in vector database."""
    
@mcp_tool
async def add_documents(
    collection: str,
    documents: List[dict],
    metadatas: List[dict]
) -> dict:
    """Add documents to collection."""
```

---

#### 2.6.2 Azure Log Analytics MCP Server

**Purpose**: Execute KQL queries against Azure Log Analytics
**Used By**: KQLDataAgent
**Location**: Separate MCP server microservice

**Tools**:
```python
@mcp_tool
async def execute_query(
    workspace_id: str,
    query: str,
    timespan: str = "PT1H"
) -> dict:
    """Execute KQL query and return results."""
    
@mcp_tool
async def get_schema(
    workspace_id: str,
    table: str = None
) -> dict:
    """Get workspace schema information."""
```

---

#### 2.6.3 Splunk MCP Server

**Purpose**: Execute SPL searches against Splunk
**Used By**: SPLDataAgent
**Location**: Separate MCP server microservice

**Capabilities**:
- Deep understanding of Azure log schemas (SigninLogs, SecurityEvents, etc.)
- KQL-specific optimizations (summarize, join strategies)
- Time range optimization for Azure retention policies
- Integration with GraphRAG for Azure schema knowledge

**A2A Interface**:
```python
class KQLQueryGeneratorA2A:
    @a2a_task
    async def generate_kql_query(
        natural_language: str,
        sop_context: dict,
        schema_hints: dict,
        journey_id: str
    ) -> dict:
        # Returns: {query, confidence, explanation, optimizations}
```

#### 2.6.2 SPLQueryGenerator (A2A Agent)

**Platform**: Splunk Enterprise, Splunk Cloud
**Expertise**: SPL syntax, Splunk data models, search optimization
**Location**: Separate microservice with A2A server endpoint

**Capabilities**:
- Splunk Common Information Model (CIM) knowledge
- SPL command pipeline optimization
- Index-time vs search-time field extraction
- Integration with GraphRAG for Splunk schema knowledge

**A2A Interface**:
```python
class SPLQueryGeneratorA2A:
    @a2a_task
    async def generate_spl_query(
        natural_language: str,
        sop_context: dict,
        schema_hints: dict,
        journey_id: str
    ) -> dict:
        # Returns: {query, confidence, explanation, optimizations}
```

#### 2.6.3 SQLQueryGenerator (A2A Agent)

**Platform**: PostgreSQL, MySQL, SQL Server, Oracle
**Expertise**: SQL dialects, database schemas, join optimization
**Location**: Separate microservice with A2A server endpoint

**Capabilities**:
- Multi-dialect SQL support (ANSI, T-SQL, PL/SQL)
- Index-aware optimization
- Join strategy selection
- Integration with GraphRAG for database schema knowledge

**A2A Interface**:
```python
class SQLQueryGeneratorA2A:
    @a2a_task
    async def generate_sql_query(
        natural_language: str,
        sop_context: dict,
        schema_hints: dict,
        db_dialect: str,
        journey_id: str
    ) -> dict:
        # Returns: {query, confidence, explanation, optimizations}
```

---

## 2.7 A2A Protocol Architecture Benefits

### Why A2A for Query Generators?

**1. Platform Expertise Isolation**
- Each query generator is a specialist in its platform (KQL, SPL, SQL)
- Independent development cycles per platform
- Can use different LLM models/prompts optimized for each platform
- Schema knowledge can be deeply integrated per platform

**2. Scalability & Independence**
- Query generators can scale independently based on demand
- KQL generator can be deployed closer to Azure regions
- SPL generator can be co-located with Splunk infrastructure
- Different teams can own different generators

**3. A2A Protocol Advantages**
- Standard agent-to-agent communication
- Built-in task routing and discovery
- Protocol handles serialization, versioning
- Natural fit for LLM-based agents
- Cost information flows through protocol

**4. Reduced Core Engine Complexity**
- Core engine doesn't need platform-specific knowledge
- QueryRouter agent only needs to know WHAT platform, not HOW to generate
- Platform logic stays with platform specialists
- Easier to add new platforms (just deploy new A2A agent)

**5. Testing & Mocking**
- Each query generator can be tested independently
- Mock A2A agents are simple REST endpoints
- Can swap real/mock generators without changing core engine
- Integration tests are cleaner

### A2A Communication Flow

```
Core Engine                           A2A Protocol                    Query Generator Agent
    │                                       │                                  │
    │ 1. Route to KQL generator            │                                  │
    │─────────────────────────────────────▶│                                  │
    │                                       │                                  │
    │                                       │ 2. Discover KQL agent           │
    │                                       │─────────────────────────────────▶│
    │                                       │                                  │
    │                                       │ 3. Agent endpoint               │
    │                                       │◀─────────────────────────────────│
    │                                       │                                  │
    │                                       │ 4. Invoke generate_query task   │
    │                                       │─────────────────────────────────▶│
    │                                       │                                  │
    │                                       │                                  │ 5. LLM processing
    │                                       │                                  │    + GraphRAG
    │                                       │                                  │    + KQL optimization
    │                                       │                                  │
    │                                       │ 6. Query + metadata + cost      │
    │                                       │◀─────────────────────────────────│
    │                                       │                                  │
    │ 7. A2A response                      │                                  │
    │◀─────────────────────────────────────│                                  │
    │                                       │                                  │
    │ 8. Validate & use query              │                                  │
    │                                       │                                  │
```

---

## 3. Orchestration Algorithm

### 3.1 Main Orchestration Flow

```
INPUT: QueryRequest + JourneyID (via MCP or REST)
    ↓
[1] INITIALIZE JOURNEY CONTEXT
    - Create JourneyContext
    - Initialize state machine → PENDING
    - Register journey in active_journeys
    ↓
[2] DISCOVER REQUIRED AGENTS
    State Transition: PENDING → DISCOVERING
    │
    ├─ A2A Registry Query:
    │  - Find available Planner agents
    │  - Find available Knowledge Provider agents (SOP, Wiki, Error)
    │  - Find available Data agents (KQL, SPL, SQL)
    │
    ├─ Validate minimum requirements:
    │  - At least 1 Planner agent
    │  - At least 1 Knowledge Provider agent
    │  - At least 1 Data agent matching platform
    │
    └─ Store agent descriptors in context
    ↓
[3] INVOKE PLANNER AGENT (A2A)
    State Transition: DISCOVERING → PLANNING
    │
    ├─ A2A Request to WorkflowPlannerAgent:
    │  - natural_language: user query
    │  - user_context: session info
    │  - available_agents: discovered agents
    │  - journey_id: for tracking
    │
    ├─ Planner Agent Processing:
    │  - Uses own LLM to analyze query
    │  - Determines required knowledge sources
    │  - Selects target platforms
    │  - Creates execution plan
    │  - Tracks cost internally
    │
    └─ A2A Response: WorkflowPlan
       - steps: ["fetch_sop", "fetch_errors", "generate_kql"]
       - agent_assignments: {step → agent_id}
       - estimated_cost: {...}
    ↓
[4] INVOKE KNOWLEDGE PROVIDER AGENTS (A2A - Parallel)
    State Transition: PLANNING → GATHERING_KNOWLEDGE
    │
    ├─ For each knowledge source in plan:
    │  │
    │  ├─ A2A Request to SOPKnowledgeAgent:
    │  │  - query: user query
    │  │  - journey_id: tracking
    │  │  Agent uses MCP to query ChromaDB
    │  │  Returns: SOPContext with procedures
    │  │
    │  ├─ A2A Request to KnownErrorAgent:
    │  │  - query: user query
    │  │  Agent uses MCP to query ErrorDB
    │  │  Returns: ErrorContext with known issues
    │  │
    │  └─ A2A Request to WikiKnowledgeAgent:
    │     - query: user query
    │     Agent uses MCP to search Wiki
    │     Returns: WikiContext with articles
    │
    └─ Aggregate all knowledge contexts
    ↓
[5] INVOKE DATA AGENTS (A2A - Based on Plan)
    State Transition: GATHERING_KNOWLEDGE → GENERATING_QUERIES
    │
    ├─ For each data platform in plan:
    │  │
    │  ├─ A2A Request to KQLDataAgent (if platform=kql):
    │  │  - natural_language: user query
    │  │  - sop_context: from SOPKnowledgeAgent
    │  │  - error_context: from KnownErrorAgent
    │  │  - wiki_context: from WikiKnowledgeAgent
    │  │  - schema_hints: from plan
    │  │  - execute: true (optional - execute and return data)
    │  │  - journey_id: tracking
    │  │  
    │  │  Agent Processing:
    │  │  - Uses own LLM to generate KQL
    │  │  - GraphRAG for Azure schema
    │  │  - MCP call to Azure LA for execution (if execute=true)
    │  │  - Tracks cost internally
    │  │  
    │  │  Returns: QueryResult {query, data, confidence, cost_info}
    │  │
    │  ├─ A2A Request to SPLDataAgent (if platform=spl):
    │  │  Similar flow, uses MCP to Splunk
    │  │
    │  └─ A2A Request to SQLDataAgent (if platform=sql):
    │     Similar flow, uses MCP to SQL database
    │
    └─ Aggregate query results from all platforms
    ↓
[6] AGGREGATE & VALIDATE
    State Transition: GENERATING_QUERIES → VALIDATING
    │
    ├─ Validate all agent responses:
    │  ├─ Check knowledge context completeness
    │  ├─ Verify queries generated successfully
    │  ├─ Check confidence thresholds
    │  └─ Basic security validation
    │
    └─ Output: ValidationResult with status
    ↓
[7] AGGREGATE COSTS FROM ALL AGENTS
    │
    ├─ Collect cost_info from each A2A response:
    │  ├─ PlannerAgent: cost_info
    │  ├─ SOPKnowledgeAgent: cost_info (LLM + MCP embedding)
    │  ├─ KnownErrorAgent: cost_info
    │  ├─ WikiKnowledgeAgent: cost_info
    │  ├─ KQLDataAgent: cost_info (LLM + MCP execution)
    │  └─ Other agents...
    │
    ├─ Aggregate to journey level:
    │  - total_llm_calls
    │  - total_mcp_calls
    │  - total_tokens
    │  - total_cost_usd
    │  - cost_by_agent
    │  - cost_by_service (LLM vs MCP)
    │
    └─ Store in JourneyCostSummary
    ↓
[8] COMPILE RESPONSE
    State Transition: VALIDATING → COMPLETED
    │
    ├─ Aggregate all agent outputs:
    │  - workflow_plan from Planner
    │  - knowledge_contexts from Knowledge Providers
    │  - query_results from Data Agents
    │  - journey_cost_summary
    │
    ├─ Format as QueryResponse:
    │  - queries: {platform → query}
    │  - results: {platform → data} (if executed)
    │  - context: merged knowledge
    │  - cost_summary: aggregated costs
    │  - metadata: journey_id, timestamps, agent_info
    │
    └─ Output: QueryResponse
    ↓
[9] CLEANUP & RETURN
    - Remove journey from active_journeys
    - Log completion with metrics
    - Publish cost summary to CostTracker
    - Return QueryResponse (via MCP or REST)

ERROR HANDLING (at any step):
    ├─ Catch Exception
    ├─ Log error with journey context
    ├─ Check retry_count
    │  ├─ If retry_count < MAX_RETRIES
    │  │   └─ Exponential backoff → Retry from step [2]
    │  └─ Else
    │      └─ State Transition → FAILED
    │      └─ Return error response
    ├─ Update JourneyContext.errors
    └─ Cleanup resources

TIMEOUT HANDLING (per state):
    ├─ ANALYZING: 30 second timeout for SOP search
    ├─ GENERATING: 60 second timeout for LLM generation
    ├─ VALIDATING: 10 second timeout for validation
    └─ On timeout: Transition to FAILED, log timeout
```

### 3.2 Cost Tracking Integration

Cost is tracked at every LLM and embedding operation:

```
[ANALYZING Phase]
    ├─ SOPEngine.search_procedures()
    │  └─ Track embedding operation (nomic-embed-text)
    │     - tokens, latency, estimated_cost
    └─ Store in JourneyContext.cost_summary

[GENERATING Phase]
    ├─ SOP Analyst Agent LLM call
    │  └─ Track LLM operation (llama3)
    │     - prompt_tokens, completion_tokens, latency, cost
    │
    ├─ Query Engineer Agent LLM call
    │  └─ Track LLM operation (llama3)
    │     - prompt_tokens, completion_tokens, latency, cost
    │
    └─ Aggregate in JourneyContext.cost_summary

[VALIDATING Phase]
    └─ Query Validator Agent (if uses LLM)
       └─ Track LLM operation if validation requires LLM

[CLEANUP]
    └─ Send final cost_summary to CostTrackerService
       - Journey aggregate
       - Per-service breakdown
       - Per-model breakdown
```

---

## 4. Service Interaction Patterns

### 4.1 Service Call Pattern with Circuit Breaker

```python
class ServiceCallPattern:
    """
    Pattern for all service interactions with resilience.
    
    Flow:
        Check Circuit Breaker
            ├─ OPEN: Fail fast
            ├─ HALF_OPEN: Allow one call
            └─ CLOSED: Proceed to retry loop
        ↓
        Retry Loop (exponential backoff):
            └─ for attempt in range(MAX_RETRIES):
                ├─ Try service call with timeout
                ├─ On success: Record, return
                ├─ On transient error: Backoff and retry
                ├─ On permanent error: Break loop
                └─ Update circuit breaker state
        ↓
        Track Cost:
            └─ Record latency and errors
        ↓
        Return Result or Error
    """
```

**Code Structure**:
```python
async def call_service_with_resilience(
    service_name: str,
    operation: Callable,
    *args,
    **kwargs
) -> Any:
    # Get circuit breaker
    cb = service_registry.get_circuit_breaker(service_name)
    
    if cb.is_open():
        raise ServiceUnavailableException(f"{service_name} is unavailable")
    
    # Retry loop
    for attempt in range(MAX_RETRIES):
        try:
            result = await asyncio.wait_for(
                operation(*args, **kwargs),
                timeout=timeout_policy.get_timeout(service_name)
            )
            cb.record_success()
            return result
        except TransientException as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(backoff(attempt))
                continue
            cb.record_failure()
            raise
        except PermanentException as e:
            cb.record_failure()
            raise
```

---

### 4.2 SOP Engine Integration

```
QueryOrchestrator
    │
    ├─ journeyID, natural_language
    │
    ▼
[ServiceRegistry.get_service("sop_engine")]
    │
    ├─ Check circuit breaker
    ├─ Timeout: 30 seconds
    ├─ Max retries: 3
    │
    ▼
SOPEngineInterface.search_procedures(
    query=request.natural_language,
    journey_id=journey_id,
    top_k=5
)
    │
    ├─ Track embedding cost
    │  └─ embedding_model="nomic-embed-text"
    │  └─ tokens, latency_ms, cost_usd
    │
    ▼
Returns: List[Procedure]
    │
    ├─ Store in JourneyContext.sop_procedures
    ├─ Log retrieval count and confidence
    │
    ▼
Continue to SOP Analyst Agent
```

---

### 4.3 Query Generator Integration (via A2A Protocol)

```
QueryRouter Agent (via CrewAI)
    │
    ├─ Input: AnalysisResult + Platform
    │
    ▼
[CrewAI Agent determines routing]
    │
    ├─ Platform: "kql" → KQLQueryGenerator
    ├─ Platform: "spl" → SPLQueryGenerator
    ├─ Platform: "sql" → SQLQueryGenerator
    │
    ▼
[A2A Client discovers and invokes agent]
    │
    ├─ A2A Discovery:
    │  └─ Find service endpoint for platform generator
    │
    ├─ A2A Request Payload:
    │  {
    │    "natural_language": "...",
    │    "sop_context": {...},
    │    "schema_hints": {...},
    │    "journey_id": "uuid"
    │  }
    │
    ├─ Protocol: A2A SDK (a2a-sdk-python)
    ├─ Timeout: 60 seconds
    ├─ Retry: 3 attempts on transient errors
    │
    ▼
[Query Generator A2A Agent executes]
    │
    ├─ Platform-Specific Processing:
    │  ├─ Uses specialized Ollama LLM prompts
    │  ├─ GraphRAG schema enhancement
    │  ├─ Platform-specific optimization rules
    │  └─ Tracks cost internally
    │
    ▼
A2A Response:
    │
    ├─ {
    │    "query": "generated platform query",
    │    "confidence": 0.87,
    │    "explanation": "...",
    │    "optimizations": [...],
    │    "cost_info": {...}
    │  }
    │
    ▼
Core Engine:
    │
    ├─ Store in JourneyContext.generated_query
    ├─ Aggregate cost from A2A response
    ├─ Include confidence score
    │
    ▼
Pass to Response Validation
```

---

## 5. Interaction Flow Diagrams

### 5.1 Complete A2A + MCP Workflow Sequence

```
USER/CLIENT
  │
  ├─ MCP Request OR REST POST /api/v1/query
  │  {"natural_language": "...", "platform": "kql"}
  │
  ▼
LexiQueryCore (MCP Server + A2A Orchestrator)
  │
  ├─────────────────────────────────────────────────────┐
  │  [STEP 1] Initialize Journey                        │
  │  ├─ Create JourneyContext                           │
  │  ├─ State: PENDING → DISCOVERING                    │
  │  └─ Initialize cost aggregation                     │
  └─────────────────────────────────────────────────────┘
           │
           ▼
  ├─────────────────────────────────────────────────────┐
  │  [STEP 2] Discover A2A Agents                       │
  │  ├─ Query A2A Registry                              │
  │  ├─ Find: Planner agents                            │
  │  ├─ Find: Knowledge Provider agents                 │
  │  ├─ Find: Data agents (KQL, SPL, SQL)               │
  │  └─ Validate minimum requirements met               │
  └─────────────────────────────────────────────────────┘
           │
           ▼
  ┌────────────────────────────────────────────────────┐
  │  WorkflowPlannerAgent (A2A)                        │
  │  ┌──────────────────────────────────────────────┐  │
  │  │ [STEP 3] Create Execution Plan               │  │
  │  │ State: DISCOVERING → PLANNING                │  │
  │  │                                               │  │
  │  │ A2A Input:                                    │  │
  │  │ - natural_language: user query                │  │
  │  │ - user_context: {}                           │  │
  │  │ - available_agents: {planner, knowledge,     │  │
  │  │                      data}                    │  │
  │  │                                               │  │
  │  │ Internal Processing:                          │  │
  │  │ ├─ LLM analyzes query intent                 │  │
  │  │ ├─ Determines knowledge sources needed       │  │
  │  │ ├─ Selects target platforms                  │  │
  │  │ ├─ Creates step-by-step plan                 │  │
  │  │ └─ Tracks cost                               │  │
  │  │                                               │  │
  │  │ A2A Output: WorkflowPlan                     │  │
  │  │ {                                             │  │
  │  │   steps: [                                    │  │
  │  │     {agent: "sop_knowledge", action:          │  │
  │  │      "search_procedures"},                    │  │
  │  │     {agent: "error_knowledge", action:        │  │
  │  │      "search_known_errors"},                  │  │
  │  │     {agent: "kql_data", action:               │  │
  │  │      "generate_and_execute"}                  │  │
  │  │   ],                                          │  │
  │  │   estimated_cost: {...},                      │  │
  │  │   cost_info: {...}                            │  │
  │  │ }                                             │  │
  │  └──────────────────────────────────────────────┘  │
  └────────────────────────────────────────────────────┘
           │
           ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  Knowledge Provider Agents (A2A) - PARALLEL EXECUTION        │
  │  State: PLANNING → GATHERING_KNOWLEDGE                       │
  │                                                               │
  │  ┌────────────────────┐  ┌────────────────────┐              │
  │  │ SOPKnowledgeAgent  │  │ KnownErrorAgent    │              │
  │  │ (A2A)              │  │ (A2A)              │              │
  │  │                    │  │                    │              │
  │  │ A2A Input:         │  │ A2A Input:         │              │
  │  │ - query: "..."     │  │ - query: "..."     │              │
  │  │ - journey_id       │  │ - journey_id       │              │
  │  │                    │  │                    │              │
  │  │ ┌────────────────┐ │  │ ┌────────────────┐ │              │
  │  │ │ MCP Client     │ │  │ │ MCP Client     │ │              │
  │  │ │ ↓              │ │  │ │ ↓              │ │              │
  │  │ │ ChromaDB_MCP   │ │  │ │ ErrorDB_MCP    │ │              │
  │  │ │ - semantic_    │ │  │ │ - search_      │ │              │
  │  │ │   search()     │ │  │ │   errors()     │ │              │
  │  │ │ - embedding    │ │  │ │ - pattern_     │ │              │
  │  │ │   (nomic-      │ │  │ │   match()      │ │              │
  │  │ │    embed-text) │ │  │ │                │ │              │
  │  │ └────────────────┘ │  │ └────────────────┘ │              │
  │  │                    │  │                    │              │
  │  │ A2A Output:        │  │ A2A Output:        │              │
  │  │ SOPContext         │  │ ErrorContext       │              │
  │  │ {procedures: [...],│  │ {errors: [...],    │              │
  │  │  cost_info: {...}} │  │  cost_info: {...}} │              │
  │  └────────────────────┘  └────────────────────┘              │
  └──────────────────────────────────────────────────────────────┘
           │
           │ Aggregate knowledge contexts
           │
           ▼
  ┌────────────────────────────────────────────────────┐
  │  Data Agents (A2A) - Platform-Specific             │
  │  State: GATHERING_KNOWLEDGE → GENERATING_QUERIES   │
  │                                                     │
  │  ┌──────────────────────────────────────────────┐  │
  │  │ KQLDataAgent (A2A)                           │  │
  │  │                                               │  │
  │  │ A2A Input:                                    │  │
  │  │ - natural_language: user query                │  │
  │  │ - sop_context: from SOPKnowledgeAgent        │  │
  │  │ - error_context: from KnownErrorAgent        │  │
  │  │ - schema_hints: {}                           │  │
  │  │ - execute: true                              │  │
  │  │ - journey_id                                  │  │
  │  │                                               │  │
  │  │ Internal Processing:                          │  │
  │  │ ├─ LLM generates KQL query                   │  │
  │  │ │  - Uses specialized KQL prompts            │  │
  │  │ │  - GraphRAG for Azure schema              │  │
  │  │ │  - Context from knowledge agents          │  │
  │  │ ├─ Track LLM cost                            │  │
  │  │ │                                             │  │
  │  │ ├─ MCP Client                                │  │
  │  │ │  ↓                                          │  │
  │  │ │  AzureLogAnalytics_MCP                     │  │
  │  │ │  - execute_query(query)                   │  │
  │  │ │  - Returns: data rows                     │  │
  │  │ └─ Track MCP execution cost                  │  │
  │  │                                               │  │
  │  │ A2A Output: QueryResult                      │  │
  │  │ {                                             │  │
  │  │   query: "SigninLogs | where ...",          │  │
  │  │   data: [{row1}, {row2}, ...],              │  │
  │  │   confidence: 0.87,                          │  │
  │  │   explanation: "...",                        │  │
  │  │   cost_info: {                               │  │
  │  │     llm_tokens: 450,                         │  │
  │  │     mcp_calls: 1,                            │  │
  │  │     latency_ms: 780,                         │  │
  │  │     cost_usd: 0.008                          │  │
  │  │   }                                           │  │
  │  │ }                                             │  │
  │  └──────────────────────────────────────────────┘  │
  └────────────────────────────────────────────────────┘
           │
           ▼
  ├─────────────────────────────────────────────────────┐
  │  [STEP 7] Aggregate Costs from All Agents           │
  │  State: GENERATING_QUERIES → VALIDATING            │
  │                                                      │
  │  JourneyCostSummary:                                │
  │  ├─ PlannerAgent: $0.002                           │
  │  ├─ SOPKnowledgeAgent: $0.003 (LLM + MCP embed)    │
  │  ├─ KnownErrorAgent: $0.001                        │
  │  ├─ KQLDataAgent: $0.008 (LLM + MCP execute)       │
  │  │                                                  │
  │  └─ TOTAL: $0.014                                  │
  │     - LLM calls: 4                                 │
  │     - MCP calls: 3                                 │
  │     - Total tokens: 1,250                          │
  └─────────────────────────────────────────────────────┘
           │
           ▼
  ├─────────────────────────────────────────────────────┐
  │  [STEP 8] Compile Response                          │
  │  State: VALIDATING → COMPLETED                      │
  │                                                      │
  │  QueryResponse:                                     │
  │  {                                                   │
  │    journey_id: "uuid",                             │
  │    workflow_plan: {...},                           │
  │    knowledge_context: {                            │
  │      sop: SOPContext,                              │
  │      errors: ErrorContext                          │
  │    },                                               │
  │    queries: {                                       │
  │      kql: "SigninLogs | where ..."                │
  │    },                                               │
  │    results: {                                       │
  │      kql: {data: [...], row_count: 42}            │
  │    },                                               │
  │    cost_summary: JourneyCostSummary,               │
  │    metadata: {                                      │
  │      agents_used: ["planner", "sop", "kql"],      │
  │      execution_time_ms: 2340                       │
  │    }                                                │
  │  }                                                   │
  └─────────────────────────────────────────────────────┘
           │
           ▼
USER/CLIENT
  │
  └─ Receives comprehensive response with queries,
     data, context, and cost breakdown
```

### 5.2 MCP Communication Pattern (Agent → Tool Server)

```
A2A Agent                       MCP Client                   MCP Tool Server
(e.g., KQLDataAgent)                                        (e.g., Azure LA MCP)
    │                               │                                │
    │ Need to execute query        │                                │
    │                               │                                │
    │ 1. Call MCP tool             │                                │
    │──────────────────────────────▶│                                │
    │   tool="execute_query"        │                                │
    │   args={query, workspace}     │                                │
    │                               │                                │
    │                               │ 2. MCP Request                 │
    │                               │───────────────────────────────▶│
    │                               │   {jsonrpc, method, params}    │
    │                               │                                │
    │                               │                                │ 3. Execute
    │                               │                                │    - Auth
    │                               │                                │    - Call Azure API
    │                               │                                │    - Process results
    │                               │                                │
    │                               │ 4. MCP Response                │
    │                               │◀───────────────────────────────│
    │                               │   {result: {...}}              │
    │                               │                                │
    │ 5. Return data                │                                │
    │◀──────────────────────────────│                                │
    │   {data, row_count, cost}     │                                │
    │                               │                                │
```

### 5.3 A2A Communication Pattern (Core → Agent)

```
Core Orchestrator              A2A Protocol                  A2A Agent
                                                            (e.g., SOPKnowledgeAgent)
    │                               │                                │
    │ 1. Discover agent             │                                │
    │──────────────────────────────▶│                                │
    │   agent_type="knowledge"      │                                │
    │   capability="sop_search"     │                                │
    │                               │                                │
    │ 2. Agent endpoint             │                                │
    │◀──────────────────────────────│                                │
    │   url, capabilities, schema   │                                │
    │                               │                                │
    │ 3. Invoke task                │                                │
    │──────────────────────────────▶│                                │
    │   task="retrieve_sop_context" │                                │
    │   params={query, journey_id}  │                                │
    │                               │                                │
    │                               │ 4. Task request                │
    │                               │───────────────────────────────▶│
    │                               │                                │
    │                               │                                │ 5. Execute
    │                               │                                │    - MCP call
    │                               │                                │    - LLM processing
    │                               │                                │    - Cost tracking
    │                               │                                │
    │                               │ 6. Task result + cost_info     │
    │                               │◀───────────────────────────────│
    │                               │                                │
    │ 7. Result + cost              │                                │
    │◀──────────────────────────────│                                │
    │                               │                                │
```

---

## 6. Mock Service Specifications

### 5.1 Complete Query Workflow Sequence

```
USER
  │
  ├─ POST /api/v1/query
  │  {"natural_language": "...", "platform": "kql"}
  │
  ▼
FastAPI Controller
  │
  ├─ Generate journey_id
  ├─ Create QueryRequest
  │
  ▼
QueryOrchestrator.orchestrate_query_workflow()
  │
  ├─────────────────────────────────────────────────────┐
  │  [STEP 1] Initialize Journey                        │
  │  ├─ Create JourneyContext                           │
  │  ├─ State: PENDING                                  │
  │  └─ Initialize cost tracking                        │
  └─────────────────────────────────────────────────────┘
           │
           ▼
  ├─────────────────────────────────────────────────────┐
  │  [STEP 2] Fetch SOP Context                         │
  │  ├─ State: PENDING → ANALYZING                      │
  │  ├─ Call SOPEngine.search_procedures()              │
  │  │  └─ Track embedding cost                         │
  │  ├─ Timeout: 30 seconds                             │
  │  └─ Retry: 3 attempts with exponential backoff      │
  └─────────────────────────────────────────────────────┘
           │
           ├─ Success: Store procedures in context
           │
           ├─ Failure (transient): Retry
           │
           └─ Failure (permanent): Fail journey
           │
           ▼
  ├─────────────────────────────────────────────────────┐
  │  [STEP 3] Execute SOP Analyst Agent                 │
  │  ├─ CrewAI agent task                               │
  │  ├─ Input: QueryRequest + Procedures                │
  │  ├─ LLM: Ollama (llama3)                            │
  │  ├─ Track LLM cost                                  │
  │  ├─ Timeout: 60 seconds                             │
  │  └─ Output: AnalysisResult                          │
  └─────────────────────────────────────────────────────┘
           │
           ├─ Analyze requirements
           ├─ Extract relevant fields
           ├─ Identify constraints
           │
           ▼
  ├─────────────────────────────────────────────────────┐
  │  [STEP 4] Execute Query Engineer Agent              │
  │  ├─ State: ANALYZING → GENERATING                   │
  │  ├─ CrewAI agent task                               │
  │  ├─ Input: AnalysisResult + Platform                │
  │  ├─ LLM: Ollama (llama3) with schema hints          │
  │  ├─ Track LLM cost                                  │
  │  ├─ Timeout: 60 seconds                             │
  │  └─ Output: Generated Query String                  │
  └─────────────────────────────────────────────────────┘
           │
           ├─ Generate platform-specific query
           ├─ Apply optimizations
           ├─ Include explanation
           │
           ▼
  ├─────────────────────────────────────────────────────┐
  │  [STEP 5] Execute Query Validator Agent             │
  │  ├─ State: GENERATING → VALIDATING                  │
  │  ├─ CrewAI agent task                               │
  │  ├─ Input: Generated Query + Platform Rules         │
  │  ├─ Operations: DETERMINISTIC (no LLM)              │
  │  │  ├─ Syntax validation                            │
  │  │  ├─ Security checks                              │
  │  │  └─ Performance estimates                        │
  │  ├─ Timeout: 10 seconds                             │
  │  └─ Output: QueryValidation                         │
  └─────────────────────────────────────────────────────┘
           │
           ├─ Validate syntax
           ├─ Check for vulnerabilities
           ├─ Estimate execution cost
           │
           ▼
  ├─────────────────────────────────────────────────────┐
  │  [STEP 6] Compile Response                          │
  │  ├─ State: VALIDATING → OPTIMIZING → COMPLETED     │
  │  ├─ Aggregate all results                           │
  │  ├─ Include cost breakdown                          │
  │  └─ Format as QueryResponse                         │
  └─────────────────────────────────────────────────────┘
           │
           ▼
  ├─────────────────────────────────────────────────────┐
  │  [STEP 7] Cleanup & Return                          │
  │  ├─ Send cost summary to CostTrackerService         │
  │  ├─ Log metrics                                     │
  │  ├─ Remove from active journeys                     │
  │  └─ Return QueryResponse to controller              │
  └─────────────────────────────────────────────────────┘
           │
           ▼
FastAPI Controller
  │
  ├─ HTTP 200 OK
  ├─ QueryResponse JSON
  │
  ▼
USER
  │
  └─ Receives generated query with metadata
```

### 5.2 Error Handling Flow

```
Error occurs at any step
    │
    ▼
Log error with full context
    │
    ├─ journey_id
    ├─ current_state
    ├─ step name
    ├─ error message
    └─ stack trace
    │
    ▼
Classify error
    │
    ├─ TransientException (network, timeout)
    │  └─ Retryable
    │
    ├─ PermanentException (invalid input, config)
    │  └─ Not retryable
    │
    └─ UnknownException
       └─ Retry once, then fail
    │
    ▼
Check retry count
    │
    ├─ If retries < MAX_RETRIES and transient
    │  │
    │  ├─ Calculate backoff: 2^attempt * 1 second (max 30s)
    │  ├─ Update circuit breaker (if service-specific)
    │  ├─ Log retry attempt
    │  │
    │  └─ RETRY from appropriate step
    │
    └─ Else (max retries or permanent error)
       │
       ├─ Update JourneyContext.last_error
       ├─ Add to JourneyContext.errors list
       ├─ State Transition: → FAILED
       │
       └─ Return ErrorResponse
          └─ Include error details + logs for debugging
```

---

## 6. Mock Service Specifications

### 6.1 MockSOPEngine

**Purpose**: Return pre-defined procedures for testing

**Data**:
```python
MOCK_PROCEDURES = [
    Procedure(
        id="sop_001",
        title="Incident Response: Service Outage",
        description="...",
        steps=["Step 1", "Step 2"],
        relevant_tables=["SigninLogs", "AuditLogs"],
        tags=["incident", "outage"]
    ),
    Procedure(
        id="sop_002",
        title="Security Investigation: Suspicious Activity",
        relevant_tables=["SecurityEvents", "NetworkTraffic"],
        tags=["security", "investigation"]
    ),
    # ... more procedures
]
```

**Behavior**:
- `search_procedures()`: Keyword match against procedure titles/tags, return top-k
- `process_sop()`: Return mock ProcessedSOP with fixed IDs
- Latency: 100-500ms to simulate network delay
- Cost tracking: Enabled (mock tokens and latency)

### 6.2 MockQueryGeneratorA2A

**Purpose**: Mock A2A agent that returns template queries based on platform
**Protocol**: A2A SDK server endpoint
**Location**: Separate mock service per platform

**Templates**:
```python
QUERY_TEMPLATES = {
    "kql": "SigninLogs | where TimeGenerated > ago({time}) | where Status == 'Failure'",
    "spl": "source=\"main\" Status=\"Failure\" | stats count by user",
    "sql": "SELECT * FROM logs WHERE status='Failure' AND timestamp > NOW() - INTERVAL 1 HOUR"
}
```

**Behavior**:
- A2A endpoint: `/a2a/generate_query`
- Input: A2A request with natural_language, sop_context, schema_hints
- Processing:
  - Return platform template + user's query embedded
  - Add simulated latency (200-800ms)
  - Mock cost tracking (100-500 tokens)
- Response:
  ```python
  {
    "query": "<platform query>",
    "confidence": 0.75,
    "explanation": "Generated from template",
    "optimizations": ["basic_template"],
    "cost_info": {
      "tokens": 250,
      "latency_ms": 450,
      "estimated_cost_usd": 0.003
    }
  }
  ```
- Mock services:
  - MockKQLGeneratorA2A (port 9001)
  - MockSPLGeneratorA2A (port 9002)
  - MockSQLGeneratorA2A (port 9003)

### 6.3 MockCacheService

**Purpose**: In-memory caching for mock phase

**Behavior**:
- `get()`: Return cached value or None
- `set()`: Store in Dict with TTL
- `delete()`: Remove entry
- Simple TTL expiration (1 hour default)

### 6.4 MockDataConnector

**Purpose**: Synthetic data responses

**Behavior**:
- `execute_query()`: Return mock result set with 5-10 rows
- Schema: Fixed columns based on platform
- Latency: 200-1000ms simulation

### 6.5 MockProtocolHandler

**Purpose**: Minimal MCP/A2A simulation

**Behavior**:
- `handle_request()`: Echo back request data
- Minimal validation
- Mock response with success status

---

## 7. Configuration & Dependency Injection

### 7.1 Configuration Model

```python
class OrchestrationConfig:
    """Core engine configuration."""
    
    # Timeouts
    sop_search_timeout_seconds: int = 30
    llm_generation_timeout_seconds: int = 60
    query_validation_timeout_seconds: int = 10
    default_timeout_seconds: int = 30
    
    # Retry Policy
    max_retries: int = 3
    initial_backoff_seconds: int = 1
    max_backoff_seconds: int = 30
    backoff_multiplier: float = 2.0
    
    # Transient Errors (Retryable)
    transient_errors: List[Type[Exception]] = [
        TimeoutError,
        ConnectionError,
        ServiceUnavailableException
    ]
    
    # CrewAI Config
    crewai_verbose: bool = True
    crewai_memory: bool = True
    llm_model: str = "llama3"
    embedding_model: str = "nomic-embed-text"
    ollama_host: str = "http://localhost:11434"
    
    # MCP Server Configuration
    mcp_server_port: int = 8000
    mcp_server_name: str = "lexiquery-core"
    
    # A2A Protocol Config
    a2a_registry_url: str = "http://a2a-registry:7000"  # Agent discovery
    a2a_timeout_seconds: int = 60
    a2a_discovery_enabled: bool = True
    a2a_retry_attempts: int = 3
    
    # Required Agent Minimums
    require_planner_agent: bool = True
    require_knowledge_agent: bool = True  # At least 1
    require_data_agent: bool = True  # At least 1 matching platform
    
    # A2A Agent Endpoints (can be discovered or configured)
    # Planner Agents
    planner_agents: List[str] = [
        "http://workflow-planner-a2a:9000"
    ]
    
    # Knowledge Provider Agents
    knowledge_agents: Dict[str, str] = {
        "sop": "http://sop-knowledge-a2a:9010",
        "errors": "http://error-knowledge-a2a:9011",
        "wiki": "http://wiki-knowledge-a2a:9012",
        "docs": "http://docs-knowledge-a2a:9013"
    }
    
    # Data Agents (Platform-specific)
    data_agents: Dict[str, str] = {
        "kql": "http://kql-data-a2a:9020",
        "spl": "http://spl-data-a2a:9021",
        "sql": "http://sql-data-a2a:9022"
    }
    
    # MCP Tool Servers (used by agents, not directly by core)
    mcp_tool_servers: Dict[str, str] = {
        "chromadb": "http://chromadb-mcp:8100",
        "errordb": "http://errordb-mcp:8101",
        "azure_la": "http://azure-la-mcp:8102",
        "splunk": "http://splunk-mcp:8103",
        "sql": "http://sql-mcp:8104"
    }
    
    # Mode
    use_mocks: bool = True  # Phase 2 default
    
    class Config:
        env_file = ".env"
        env_prefix = "CORE_ENGINE_"
```

### 7.2 Dependency Injection Setup

```python
class DIContainer:
    """Dependency injection container."""
    
    def __init__(self, config: OrchestrationConfig):
        self.config = config
        self._services: Dict[str, Any] = {}
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize all service instances."""
        if self.config.use_mocks:
            # Phase 2: Use mocks
            self._services["sop_engine"] = MockSOPEngine()
            self._services["query_generator"] = MockQueryGenerator()
            self._services["cache_service"] = MockCacheService()
            self._services["data_connector"] = MockDataConnector()
            self._services["protocol_handler"] = MockProtocolHandler()
        else:
            # Phase 4: Use real services
            self._services["sop_engine"] = RealSOPEngine(
                url=self.config.sop_engine_url
            )
            # ... etc
        
        self._services["cost_tracker"] = RedisCostTracker()
        self._services["service_registry"] = ServiceRegistry(
            services=self._services,
            config=self.config
        )
        self._services["state_machine"] = WorkflowStateMachine()
        self._services["orchestrator"] = QueryOrchestrator(
            config=self.config,
            service_registry=self._services["service_registry"],
            cost_tracker=self._services["cost_tracker"],
            state_machine=self._services["state_machine"]
        )
    
    def get_service(self, service_name: str) -> Any:
        return self._services[service_name]
    
    def get_orchestrator(self) -> QueryOrchestrator:
        return self._services["orchestrator"]
```

---

## 8. Testing Strategy for Phase 2

### 8.1 Unit Tests

```
tests/
├── test_query_orchestrator.py
│   ├── test_orchestrate_query_workflow_success()
│   ├── test_orchestrate_with_sop_context()
│   ├── test_error_handling_with_retry()
│   └── test_cost_tracking()
│
├── test_workflow_state_machine.py
│   ├── test_valid_state_transitions()
│   ├── test_invalid_transitions_rejected()
│   ├── test_state_timeout_handling()
│   └── test_state_history_tracking()
│
├── test_service_registry.py
│   ├── test_service_registration()
│   ├── test_service_health_checks()
│   ├── test_circuit_breaker_open()
│   └── test_service_resolution()
│
└── test_crewai_agents.py
    ├── test_sop_analyst_task()
    ├── test_query_engineer_task()
    ├── test_query_validator_task()
    └── test_agent_collaboration()
```

### 8.2 Integration Tests

```
integration_tests/
├── test_full_workflow.py
│   ├── test_query_to_response_success()
│   ├── test_workflow_with_sop_context()
│   ├── test_error_recovery()
│   └── test_concurrent_journeys()
│
└── test_mock_services.py
    ├── test_mock_sop_engine()
    ├── test_mock_query_generator()
    ├── test_mock_cache_service()
    └── test_mock_data_connector()
```

### 8.3 Test Scenarios

**Success Path**:
- User submits valid query → Receives generated query response

**SOP Context Path**:
- User submits incident query → Relevant SOPs retrieved → Query generated using SOP context

**Error Recovery Path**:
- Service timeout occurs → Automatic retry → Success

**Validation Failure Path**:
- Generated query fails validation → Error response with details

**Concurrent Journeys**:
- Multiple concurrent requests → All processed independently with proper isolation

---

## 9. Integration Points with Other Services

### 9.1 SOP Engine Contract

**Orchestrator → SOP Engine**:
```python
request = {
    "query": "failed login attempts",
    "journey_id": "uuid",
    "top_k": 5
}

response = {
    "procedures": [
        {
            "id": "sop_001",
            "title": "Incident Response",
            "steps": [...],
            "relevant_tables": [...]
        }
    ],
    "search_confidence": 0.85
}
```

**Expectation**: 30 second timeout, retry 3 times on transient errors

### 9.2 Query Generator Contract (A2A Protocol)

**Orchestrator → Query Generator** (via A2A SDK):
```python
# A2A Request
request = {
    "task": "generate_query",
    "parameters": {
        "natural_language": "show failed logins",
        "platform": "kql",
        "sop_context": [
            {
                "procedure_id": "sop_001",
                "relevant_tables": ["SigninLogs"],
                "fields": ["Status", "UserPrincipalName"]
            }
        ],
        "schema_hints": {
            "tables": ["SigninLogs"],
            "columns": [...],
            "relationships": [...]
        },
        "journey_id": "uuid"
    }
}

# A2A Response from platform-specific generator
response = {
    "status": "success",
    "result": {
        "query": "SigninLogs | where TimeGenerated > ago(1h) | where Status == 'Failure' | project TimeGenerated, UserPrincipalName, IPAddress",
        "confidence": 0.87,
        "explanation": "Query retrieves failed sign-in attempts from the last hour using KQL best practices",
        "optimizations": [
            "time_filter_first",
            "project_limiting_columns"
        ],
        "estimated_cost": {
            "scan_gb": 0.5,
            "execution_time_estimate": "1-3 seconds"
        },
        "cost_info": {
            "llm_tokens": 450,
            "latency_ms": 780,
            "estimated_cost_usd": 0.005
        }
    }
}
```

**Expectation**: 60 second timeout, retry 3 times on transient A2A errors

### 9.3 Cache Service Contract

**Orchestrator → Cache Service**:
```python
# Check cache before SOP search
get_request = {
    "key": "query_hash",
    "journey_id": "uuid"
}

# Store result after completion
set_request = {
    "key": "query_hash",
    "value": response,
    "ttl_seconds": 3600
}
```

### 9.4 Cost Tracker Contract

**Orchestrator → Cost Tracker**:
```python
# Send journey cost summary on completion
cost_record = {
    "journey_id": "uuid",
    "total_llm_calls": 2,
    "total_embedding_calls": 1,
    "total_tokens": 1250,
    "total_cost_usd": 0.015,
    "cost_by_service": {...},
    "cost_by_model": {...}
}
```

---

## 10. Metrics & Monitoring

### 10.1 Key Metrics

```
Orchestrator Metrics:
├── journey_initiated_total          (counter)
├── journey_completed_total          (counter)
├── journey_failed_total             (counter)
├── journey_duration_seconds         (histogram)
├── state_transition_total           (counter)
│
Agent Metrics:
├── agent_task_executed_total        (counter)
├── agent_task_duration_seconds      (histogram)
├── agent_task_errors_total          (counter)
│
Service Call Metrics:
├── service_call_total               (counter, labeled by service)
├── service_call_duration_seconds    (histogram)
├── service_call_errors_total        (counter)
│
Cost Metrics:
├── cost_total_usd                   (gauge)
├── tokens_used_total                (counter)
├── llm_calls_total                  (counter)
└── embedding_calls_total            (counter)
```

### 10.2 Health Check Endpoints

```
GET /health
├─ Overall service health
└─ Dependencies status

GET /health/services
├─ SOP Engine: healthy/unhealthy
├─ Query Generator: mock/healthy/unhealthy
├─ Cache Service: mock/healthy/unhealthy
└─ Cost Tracker: healthy/unhealthy
```

---

## 11. Rollout Strategy

### Phase 2a: Local Development (Week 3)
- Implement core orchestrator
- Build mock services
- Unit tests ✓
- Integration tests ✓
- Manual smoke tests ✓

### Phase 2b: Docker Validation (Week 4)
- Containerize all services
- docker-compose local orchestration
- End-to-end tests
- Performance profiling

### Phase 2c: Validation & Documentation (Week 5)
- Complete test coverage
- API documentation
- Orchestration flow documentation
- Developer guide

---

## 12. Success Criteria

- ✅ Core orchestrator successfully orchestrates all mocked services
- ✅ CrewAI agents execute tasks with proper LLM integration
- ✅ State machine transitions correctly through workflow
- ✅ Cost tracking records all LLM and embedding operations
- ✅ Error handling with retry logic works end-to-end
- ✅ Docker containers for core engine and mocks build and run
- ✅ Integration tests pass with >80% code coverage
- ✅ Complete E2E workflow from request to response works
- ✅ Concurrent journey handling works correctly
- ✅ Service health checks functional

---

## Appendix A: Algorithm Pseudocode

```python
ALGORITHM QueryWorkflowOrchestration(request: QueryRequest, journey_id: UUID)

    // Initialize
    context ← CreateJourneyContext(request, journey_id)
    state ← PENDING
    retry_count ← 0
    MAX_RETRIES ← 3
    
    TRY
        
        // Phase 1: Fetch SOP Context
        state ← ANALYZING
        procedures ← CallWithRetry(
            sop_engine.search_procedures(request.natural_language),
            max_retries=3,
            timeout=30s
        )
        context.sop_procedures ← procedures
        context.cost_summary.add(embedding_cost)
        
        // Phase 2: SOP Analysis
        analysis ← sop_analyst_agent.execute(
            query=request.natural_language,
            sop_context=procedures
        )
        context.analysis_result ← analysis
        context.cost_summary.add(llm_cost)
        
        // Phase 3: Query Generation
        state ← GENERATING
        generated_query ← query_engineer_agent.execute(
            analysis=analysis,
            platform=request.platform,
            schema_hints=request.schema_hints
        )
        context.generated_query ← generated_query
        context.cost_summary.add(llm_cost)
        
        // Phase 4: Query Validation
        state ← VALIDATING
        validation ← query_validator_agent.execute(
            query=generated_query,
            platform=request.platform
        )
        context.validation_result ← validation
        
        IF validation.is_valid THEN
            state ← OPTIMIZING
            state ← COMPLETED
            response ← CompileQueryResponse(context)
        ELSE
            RAISE QueryValidationException(validation.errors)
        ENDIF
        
        // Cleanup
        cost_tracker.track_journey(context.cost_summary)
        RETURN response
        
    CATCH TransientException AS error
        
        IF retry_count < MAX_RETRIES THEN
            retry_count ← retry_count + 1
            backoff ← ExponentialBackoff(retry_count)
            SLEEP(backoff)
            GOTO TRY  // Retry from Phase 1
        ELSE
            GOTO ERROR_HANDLING
        ENDIF
        
    CATCH PermanentException AS error
        GOTO ERROR_HANDLING
        
    ERROR_HANDLING:
        state ← FAILED
        context.last_error ← error
        context.errors.append(error)
        error_response ← CompileErrorResponse(context)
        cost_tracker.track_journey(context.cost_summary)
        RETURN error_response
        
END ALGORITHM
```

---

## Appendix B: State Transition Diagram

```
                    ┌─────────────┐
                    │   PENDING   │ (Initial State)
                    └──────┬──────┘
                           │
                           │ initialize()
                           │
                    ┌──────▼──────┐
                    │ ANALYZING   │ (Fetch SOP context)
                    └──────┬──────┘
                           │
                   ┌───────┴───────┐
                   │               │
          (success)│               │(error, retry)
                   │               │
                   │    ┌──────────┘
                   │    │
                   ▼    ▼
            ┌─────────────────┐
            │    GENERATING   │ (Query generation)
            └────────┬────────┘
                     │
             ┌───────┴───────┐
             │               │
    (success)│               │(error, retry)
             │               │
             │    ┌──────────┘
             │    │
             ▼    ▼
        ┌──────────────────┐
        │    VALIDATING    │ (Query validation)
        └────────┬─────────┘
                 │
         ┌───────┴───────┐
         │               │
(success)│               │(validation fails)
         │               │
         ▼               ▼
    ┌─────────┐    ┌───────────┐
    │OPTIMIZING    │   FAILED   │
    └─────┬───────┘└───────────┘
          │
          │ finalize()
          │
    ┌─────▼──────────┐
    │   COMPLETED    │
    └────────────────┘
```

---

**Document Status**: Ready for Review and Architecture Approval

**Next Steps**:
1. Review orchestration logic and algorithm
2. Validate state machine design
3. Approve service interaction patterns
4. Begin implementation of core classes
