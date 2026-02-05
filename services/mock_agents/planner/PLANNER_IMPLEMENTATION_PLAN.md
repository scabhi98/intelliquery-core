# Planner Agent - Real Implementation Plan

**Version**: 1.0  
**Date**: February 6, 2026  
**Status**: Planning Phase  
**Objective**: Replace mock Planner agent with LLM-powered intelligent workflow planning

---

## Executive Summary

The Planner Agent is a critical A2A agent responsible for creating intelligent workflow plans based on:
- User's natural language query
- Available agent capabilities (knowledge providers and data agents)
- Execution context (schema hints, user context, journey state)

This plan outlines the transition from the current mock implementation to a production-ready LLM-powered planner using **Ollama** for local, privacy-first inference.

---

## Current State Analysis

### Mock Implementation (Phase 2)

Located in: `services/mock_agents/planner/src/main.py`

**Current Behavior**:
1. Receives A2A task with `create_plan` action
2. Uses simple keyword matching for decision making
3. Always includes SOP knowledge step
4. Conditionally adds error knowledge if query contains error keywords
5. Adds single data agent step based on platform parameter
6. Returns predefined workflow plan

**Limitations**:
- No intelligent query understanding
- Platform must be specified by user (not auto-detected)
- No multi-platform query support
- No optimization of agent selection
- No reasoning about query complexity
- Fixed workflow patterns

---

## Input Context Analysis

From `orchestrator.py` → `_invoke_planner_agent()`, the planner receives:

### A2ATaskRequest Parameters

```python
{
    "natural_language": str,  # User's query
    "execution_context": {
        "journey_id": str,
        "current_state": str,
        "state_history_count": int,
        "available_agents": {
            "planner": [
                {
                    "agent_id": str,
                    "agent_name": str,
                    "capabilities": List[str]
                }
            ],
            "knowledge": [
                {
                    "agent_id": str,
                    "agent_name": str,
                    "capabilities": List[str],
                    "knowledge_domain": str  # "sop", "errors", "wiki", etc.
                }
            ],
            "data": [
                {
                    "agent_id": str,
                    "agent_name": str,
                    "capabilities": List[str],
                    "platform": str  # "kql", "spl", "sql"
                }
            ]
        },
        "user_context": Optional[Dict],
        "schema_hints": Optional[Dict]
    }
}
```

### Key Insights

1. **Platform Selection**: User does NOT specify platform - planner must decide which data agents to use
2. **Agent Selection**: Planner must choose appropriate knowledge providers based on query
3. **Multi-Platform**: Planner can choose multiple data agents (e.g., both KQL and SQL)
4. **Schema Awareness**: Can leverage schema_hints for better planning
5. **User Context**: Can use historical context for better decisions

---

## Architecture Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Planner Agent (Port 9000)                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │         FastAPI A2A Endpoint                    │    │
│  │  POST /a2a/task (create_plan, optimize_plan)   │    │
│  └───────────────────┬────────────────────────────┘    │
│                      │                                   │
│  ┌───────────────────▼────────────────────────────┐    │
│  │         PlannerService                          │    │
│  │  - Query Analyzer                               │    │
│  │  - Agent Selector                               │    │
│  │  - Plan Generator                               │    │
│  │  - Cost Estimator                               │    │
│  └───────────────────┬────────────────────────────┘    │
│                      │                                   │
│  ┌───────────────────▼────────────────────────────┐    │
│  │      LLM Inference Engine (Ollama)             │    │
│  │  - Query Understanding (llama3)                 │    │
│  │  - Platform Detection                           │    │
│  │  - Agent Capability Matching                    │    │
│  │  - Workflow Step Sequencing                     │    │
│  └───────────────────┬────────────────────────────┘    │
│                      │                                   │
│  ┌───────────────────▼────────────────────────────┐    │
│  │        Plan Optimizer                           │    │
│  │  - Dependency Resolution                        │    │
│  │  - Parallel Execution Grouping                  │    │
│  │  - Cost Optimization                            │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### 1. Query Analyzer
**Purpose**: Understand query intent, complexity, and requirements

**Responsibilities**:
- Extract query intent (diagnostic, monitoring, security, etc.)
- Identify required data platforms (logs, metrics, database)
- Detect query complexity level (simple, moderate, complex)
- Extract entities (tables, fields, time ranges)
- Identify required knowledge domains

**LLM Prompt Pattern**:
```
Analyze this query and extract structured information:
Query: "{natural_language}"

Available schema hints: {schema_hints}
User context: {user_context}

Return JSON:
{
  "intent": "diagnostic|monitoring|security|analytics",
  "complexity": "simple|moderate|complex",
  "required_platforms": ["kql", "spl", "sql"],
  "entities": ["SigninLogs", "Users", ...],
  "time_scope": "1 hour|24 hours|7 days|custom",
  "requires_sop": true|false,
  "requires_error_knowledge": true|false,
  "query_type": "aggregation|filter|join|time_series"
}
```

#### 2. Agent Selector
**Purpose**: Match query requirements to available agents

**Responsibilities**:
- Filter knowledge agents by domain relevance
- Select data agents by platform requirements
- Validate agent capabilities match needs
- Handle missing agents gracefully

**Selection Logic**:
```python
def select_knowledge_agents(
    query_analysis: QueryAnalysis,
    available_agents: List[AgentDescriptor]
) -> List[str]:
    """Select knowledge agents based on query needs."""
    selected = []
    
    # Always include SOP if available
    if query_analysis.requires_sop:
        sop_agents = [a for a in available_agents 
                     if a.knowledge_domain == "sop"]
        if sop_agents:
            selected.append(sop_agents[0].agent_id)
    
    # Add error knowledge if needed
    if query_analysis.requires_error_knowledge:
        error_agents = [a for a in available_agents 
                       if a.knowledge_domain == "errors"]
        if error_agents:
            selected.append(error_agents[0].agent_id)
    
    return selected
```

#### 3. Plan Generator
**Purpose**: Create WorkflowPlan with ordered steps

**Responsibilities**:
- Create knowledge gathering steps
- Create data query generation steps
- Establish dependencies between steps
- Set timeouts and parameters
- Estimate costs and duration

**Step Creation Pattern**:
```python
def create_workflow_steps(
    query_analysis: QueryAnalysis,
    selected_agents: SelectedAgents
) -> List[WorkflowStep]:
    """Generate workflow steps."""
    steps = []
    
    # Phase 1: Knowledge gathering (parallel)
    for agent_id in selected_agents.knowledge:
        step = WorkflowStep(
            step_id=f"knowledge_{len(steps)}",
            agent_type="knowledge",
            agent_id=agent_id,
            action="retrieve_context",
            depends_on=[],  # No dependencies
            parameters={
                "query": query_analysis.natural_language,
                "top_k": 5,
                "domain": agent_domain
            },
            timeout_seconds=30
        )
        steps.append(step)
    
    # Phase 2: Data query generation (depends on knowledge)
    knowledge_step_ids = [s.step_id for s in steps]
    for platform in query_analysis.required_platforms:
        agent = selected_agents.get_data_agent(platform)
        step = WorkflowStep(
            step_id=f"data_{platform}",
            agent_type="data",
            agent_id=agent.agent_id,
            action="generate_query",
            depends_on=knowledge_step_ids,  # Wait for knowledge
            parameters={
                "platform": platform,
                "query_analysis": query_analysis.dict(),
                "use_knowledge": True
            },
            timeout_seconds=60
        )
        steps.append(step)
    
    return steps
```

#### 4. Plan Optimizer
**Purpose**: Optimize workflow for efficiency

**Responsibilities**:
- Identify parallel execution groups
- Minimize total execution time
- Reduce cost where possible
- Validate dependency chains

**Optimization Logic**:
```python
def optimize_plan(plan: WorkflowPlan) -> WorkflowPlan:
    """Optimize workflow plan."""
    # Group steps by dependencies for parallel execution
    parallel_groups = []
    completed_steps = set()
    
    while len(completed_steps) < len(plan.steps):
        # Find steps with no unmet dependencies
        ready_steps = [
            s for s in plan.steps
            if s.step_id not in completed_steps
            and all(dep in completed_steps for dep in s.depends_on)
        ]
        
        if ready_steps:
            parallel_groups.append([s.step_id for s in ready_steps])
            completed_steps.update(s.step_id for s in ready_steps)
    
    plan.parallel_execution_groups = parallel_groups
    return plan
```

---

## LLM Integration Strategy

### Ollama Setup

**Model**: `llama3` (or `llama3.1` for larger context)  
**Endpoint**: `http://localhost:11434`  
**Purpose**: Query analysis and intelligent decision making

### Prompt Engineering

#### Query Analysis Prompt

```python
QUERY_ANALYSIS_PROMPT = """You are a query planning expert for a multi-platform log and database query system.

Analyze the following user query and provide structured analysis:

USER QUERY:
{natural_language}

AVAILABLE PLATFORMS:
{platforms_summary}

SCHEMA HINTS:
{schema_hints}

USER CONTEXT:
{user_context}

Analyze this query and return ONLY valid JSON with the following structure:
{{
  "intent": "<diagnostic|monitoring|security|analytics|investigation>",
  "complexity": "<simple|moderate|complex>",
  "required_platforms": ["<kql|spl|sql>"],
  "confidence": <0.0-1.0>,
  "reasoning": "<brief explanation>",
  "entities": {{
    "tables": ["<table names>"],
    "fields": ["<field names>"],
    "time_range": "<description>"
  }},
  "knowledge_requirements": {{
    "requires_sop": <true|false>,
    "requires_error_knowledge": <true|false>,
    "requires_schema": <true|false>
  }},
  "query_characteristics": {{
    "is_aggregation": <true|false>,
    "is_time_series": <true|false>,
    "has_joins": <true|false>,
    "complexity_factors": ["<factor1>", "<factor2>"]
  }}
}}

Think step by step:
1. What is the user trying to accomplish?
2. Which data platforms contain the relevant data?
3. What knowledge sources would help (SOPs, error patterns)?
4. How complex is this query?

Return ONLY the JSON, no other text."""

#### Platform Selection Prompt

PLATFORM_SELECTION_PROMPT = """You are an expert at selecting the right data platform for queries.

QUERY INTENT: {intent}
ENTITIES: {entities}
AVAILABLE DATA AGENTS:
{data_agents_summary}

Select which data platforms should be used for this query.

KQL (Azure Log Analytics):
- Best for: Azure logs, security events, application telemetry
- Tables: SigninLogs, AuditLogs, AppTraces, etc.

SPL (Splunk):
- Best for: Enterprise logs, infrastructure monitoring
- Indexes: main, security, network, etc.

SQL (Database):
- Best for: Structured business data, relational queries
- Tables: Users, Orders, Transactions, etc.

Return JSON:
{{
  "selected_platforms": ["<platform>"],
  "reasoning": "<why each platform was selected>",
  "confidence": <0.0-1.0>
}}

Return ONLY the JSON."""
```

### LLM Client Implementation

```python
import ollama
import json
from typing import Dict, Any

class PlannerLLMClient:
    """LLM client for planner agent."""
    
    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://localhost:11434"
    ):
        self.model = model
        ollama.set_host(host)
    
    async def analyze_query(
        self,
        natural_language: str,
        execution_context: Dict[str, Any]
    ) -> QueryAnalysis:
        """Use LLM to analyze query requirements."""
        
        # Build prompt with context
        prompt = QUERY_ANALYSIS_PROMPT.format(
            natural_language=natural_language,
            platforms_summary=self._format_platforms(
                execution_context.get("available_agents", {}).get("data", [])
            ),
            schema_hints=json.dumps(
                execution_context.get("schema_hints", {}), 
                indent=2
            ),
            user_context=json.dumps(
                execution_context.get("user_context", {}), 
                indent=2
            )
        )
        
        # Call Ollama
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            format="json"  # Request JSON response
        )
        
        # Parse JSON response
        try:
            analysis_data = json.loads(response['response'])
            return QueryAnalysis(**analysis_data)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response", error=str(e))
            # Fallback to basic analysis
            return self._fallback_analysis(natural_language)
    
    def _format_platforms(self, data_agents: List[Dict]) -> str:
        """Format available platforms for prompt."""
        platforms = {}
        for agent in data_agents:
            platform = agent.get("platform", "unknown")
            if platform not in platforms:
                platforms[platform] = []
            platforms[platform].append(agent.get("agent_name", "unnamed"))
        
        return "\n".join(
            f"- {platform}: {', '.join(agents)}"
            for platform, agents in platforms.items()
        )
    
    def _fallback_analysis(self, query: str) -> QueryAnalysis:
        """Fallback analysis if LLM fails."""
        return QueryAnalysis(
            intent="diagnostic",
            complexity="moderate",
            required_platforms=["kql"],
            confidence=0.5,
            reasoning="Fallback analysis (LLM unavailable)",
            entities={"tables": [], "fields": [], "time_range": "unknown"},
            knowledge_requirements={
                "requires_sop": True,
                "requires_error_knowledge": "error" in query.lower(),
                "requires_schema": False
            },
            query_characteristics={
                "is_aggregation": False,
                "is_time_series": False,
                "has_joins": False,
                "complexity_factors": []
            }
        )
```

---

## Data Models

### Query Analysis Model

```python
from pydantic import BaseModel, Field
from typing import List, Literal, Dict

class QueryAnalysis(BaseModel):
    """Structured query analysis from LLM."""
    
    intent: Literal[
        "diagnostic", 
        "monitoring", 
        "security", 
        "analytics", 
        "investigation"
    ]
    complexity: Literal["simple", "moderate", "complex"]
    required_platforms: List[Literal["kql", "spl", "sql"]]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    
    entities: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Extracted entities (tables, fields, time_range)"
    )
    
    knowledge_requirements: Dict[str, bool] = Field(
        default_factory=dict,
        description="Which knowledge sources are needed"
    )
    
    query_characteristics: Dict[str, any] = Field(
        default_factory=dict,
        description="Query type characteristics"
    )
```

### Selected Agents Model

```python
class SelectedAgents(BaseModel):
    """Agents selected for workflow."""
    
    knowledge_agents: List[str] = Field(
        default_factory=list,
        description="Knowledge agent IDs"
    )
    data_agents: Dict[str, str] = Field(
        default_factory=dict,
        description="Platform -> Agent ID mapping"
    )
    reasoning: str = Field(
        ...,
        description="Why these agents were selected"
    )
```

---

## Service Structure

```
services/mock_agents/planner/
├── src/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app and A2A endpoints
│   ├── config.py                  # Pydantic settings
│   ├── service.py                 # Main PlannerService class
│   ├── models.py                  # Data models (QueryAnalysis, etc.)
│   ├── prompts.py                 # LLM prompt templates
│   ├── llm_client.py              # Ollama LLM client
│   ├── query_analyzer.py          # Query analysis logic
│   ├── agent_selector.py          # Agent selection logic
│   ├── plan_generator.py          # Workflow plan generation
│   ├── plan_optimizer.py          # Plan optimization
│   └── cost_estimator.py          # Cost estimation logic
├── tests/
│   ├── __init__.py
│   ├── test_service.py            # Service tests
│   ├── test_query_analyzer.py    # Analyzer tests
│   ├── test_plan_generator.py    # Generator tests
│   └── test_integration.py       # End-to-end tests
├── requirements.txt
├── Dockerfile
└── PLANNER_IMPLEMENTATION_PLAN.md  # This file
```

---

## Implementation Steps

### Phase 1: Core Infrastructure (Days 1-2)

**Tasks**:
1. [ ] Create data models (`models.py`)
   - QueryAnalysis
   - SelectedAgents
   - PlannerConfig
2. [ ] Set up LLM client (`llm_client.py`)
   - Ollama integration
   - Prompt formatting
   - JSON parsing with fallback
3. [ ] Create prompt templates (`prompts.py`)
   - Query analysis prompt
   - Platform selection prompt
   - Agent selection prompt
4. [ ] Update configuration (`config.py`)
   - Ollama settings
   - Model selection
   - Timeout configurations

**Deliverables**:
- Models defined and validated
- LLM client can call Ollama
- Prompts are structured and tested

### Phase 2: Query Analysis (Days 3-4)

**Tasks**:
1. [ ] Implement QueryAnalyzer class
   - LLM-based query understanding
   - Entity extraction
   - Platform detection
   - Fallback logic
2. [ ] Add platform selection logic
   - Match query intent to platforms
   - Consider available agents
3. [ ] Add knowledge requirement detection
   - SOP relevance
   - Error pattern matching
4. [ ] Write unit tests

**Deliverables**:
- QueryAnalyzer working with real queries
- Platform detection accurate (>80%)
- Test coverage >90%

### Phase 3: Agent Selection (Day 5)

**Tasks**:
1. [ ] Implement AgentSelector class
   - Filter by capability
   - Filter by domain/platform
   - Prioritize by quality metrics
2. [ ] Add agent matching logic
   - Knowledge domain matching
   - Platform compatibility
3. [ ] Handle missing agents gracefully
4. [ ] Write unit tests

**Deliverables**:
- AgentSelector selects appropriate agents
- Handles edge cases (no agents available)
- Test coverage >90%

### Phase 4: Plan Generation (Days 6-7)

**Tasks**:
1. [ ] Implement PlanGenerator class
   - Create knowledge steps
   - Create data steps
   - Set dependencies
   - Add parameters
2. [ ] Implement PlanOptimizer
   - Parallel execution grouping
   - Dependency validation
   - Cost optimization
3. [ ] Implement CostEstimator
   - Estimate step duration
   - Estimate LLM costs
   - Estimate total cost
4. [ ] Write unit tests

**Deliverables**:
- PlanGenerator creates valid WorkflowPlan
- Plans are optimized for parallelism
- Cost estimates are reasonable

### Phase 5: Service Integration (Days 8-9)

**Tasks**:
1. [ ] Implement PlannerService class
   - Orchestrate all components
   - Handle errors gracefully
   - Track costs
2. [ ] Update FastAPI endpoints
   - `/a2a/task` with create_plan
   - `/a2a/task` with optimize_plan
   - Add logging and metrics
3. [ ] Add structured logging
   - Query analysis logs
   - Agent selection logs
   - Plan generation logs
4. [ ] Write integration tests

**Deliverables**:
- Complete PlannerService
- A2A endpoints working
- End-to-end tests passing

### Phase 6: Testing & Optimization (Day 10)

**Tasks**:
1. [ ] Integration testing with Core Engine
   - Test with various queries
   - Validate workflow plans
   - Check cost tracking
2. [ ] Performance optimization
   - LLM response caching
   - Prompt optimization
   - Reduce latency
3. [ ] Error handling improvements
   - Retry logic
   - Fallback strategies
4. [ ] Documentation
   - API documentation
   - Deployment guide
   - Usage examples

**Deliverables**:
- Planner integrated with Core Engine
- Performance benchmarks met (<2s avg)
- Complete documentation

---

## Testing Strategy

### Unit Tests

```python
# tests/test_query_analyzer.py

import pytest
from src.query_analyzer import QueryAnalyzer
from src.llm_client import PlannerLLMClient

@pytest.fixture
def analyzer():
    llm_client = PlannerLLMClient(model="llama3")
    return QueryAnalyzer(llm_client)

@pytest.mark.asyncio
async def test_analyze_simple_query(analyzer):
    """Test analyzing a simple query."""
    query = "Show failed login attempts"
    context = {
        "available_agents": {
            "data": [
                {"platform": "kql", "agent_id": "kql_agent"}
            ]
        }
    }
    
    analysis = await analyzer.analyze(query, context)
    
    assert analysis.intent in ["security", "diagnostic"]
    assert "kql" in analysis.required_platforms
    assert analysis.knowledge_requirements["requires_sop"]
    assert analysis.confidence > 0.5

@pytest.mark.asyncio
async def test_multi_platform_detection(analyzer):
    """Test detecting multi-platform queries."""
    query = "Compare database user table with Azure AD signin logs"
    context = {
        "available_agents": {
            "data": [
                {"platform": "kql", "agent_id": "kql_agent"},
                {"platform": "sql", "agent_id": "sql_agent"}
            ]
        }
    }
    
    analysis = await analyzer.analyze(query, context)
    
    assert len(analysis.required_platforms) == 2
    assert "kql" in analysis.required_platforms
    assert "sql" in analysis.required_platforms
```

### Integration Tests

```python
# tests/test_integration.py

import pytest
from httpx import AsyncClient
from src.main import app

@pytest.mark.asyncio
async def test_create_plan_endpoint():
    """Test end-to-end plan creation via A2A endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        request = {
            "task_id": "test_task_1",
            "agent_id": "planner_agent",
            "action": "create_plan",
            "parameters": {
                "natural_language": "Show all error logs from last 24 hours",
                "execution_context": {
                    "available_agents": {
                        "knowledge": [
                            {
                                "agent_id": "sop_knowledge",
                                "capabilities": ["retrieve_context"],
                                "knowledge_domain": "sop"
                            }
                        ],
                        "data": [
                            {
                                "agent_id": "kql_data",
                                "capabilities": ["generate_query"],
                                "platform": "kql"
                            }
                        ]
                    }
                }
            }
        }
        
        response = await client.post("/a2a/task", json=request)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert "plan" in data["result"]
        
        plan = data["result"]["plan"]
        assert len(plan["steps"]) > 0
        assert any(s["agent_type"] == "knowledge" for s in plan["steps"])
        assert any(s["agent_type"] == "data" for s in plan["steps"])
```

---

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Average Plan Generation Time | < 2 seconds | 95th percentile |
| LLM Call Latency | < 1.5 seconds | Per call |
| Plan Accuracy | > 85% | Manual validation |
| Platform Detection Accuracy | > 90% | Automated tests |
| Agent Selection Accuracy | > 85% | Manual validation |
| Cost per Plan | < $0.01 USD | Ollama compute cost |

---

## Cost Tracking

### LLM Token Usage

```python
class CostTracker:
    """Track LLM costs for planner operations."""
    
    COST_PER_1K_TOKENS = 0.001  # Local Ollama estimate
    
    async def track_llm_call(
        self,
        prompt: str,
        response: str,
        model: str
    ) -> Dict[str, Any]:
        """Track LLM call costs."""
        prompt_tokens = self._count_tokens(prompt)
        response_tokens = self._count_tokens(response)
        total_tokens = prompt_tokens + response_tokens
        
        cost_usd = (total_tokens / 1000) * self.COST_PER_1K_TOKENS
        
        return {
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
            "model": model
        }
```

---

## Deployment

### Docker Configuration

```dockerfile
# services/mock_agents/planner/Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY shared/ /shared/

# Environment variables
ENV PYTHONPATH=/app:/shared
ENV OLLAMA_HOST=http://localhost:11434
ENV OLLAMA_MODEL=llama3

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9000/health || exit 1

# Run
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "9000"]
```

### Environment Variables

```bash
# Planner Agent Configuration
PLANNER_AGENT_ID=workflow_planner
PLANNER_AGENT_NAME=Workflow Planner Agent
PLANNER_VERSION=2.0.0

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3
OLLAMA_TIMEOUT=30

# Performance
PLANNER_MAX_PARALLEL_AGENTS=5
PLANNER_DEFAULT_TIMEOUT=60
PLANNER_ENABLE_CACHING=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=/app/logs/planner-agent.log
```

---

## Migration Strategy

### Transition from Mock to Real

1. **Parallel Deployment** (Week 1)
   - Deploy real planner on different port (9001)
   - Keep mock running on 9000
   - Test real planner with sample queries

2. **A/B Testing** (Week 2)
   - Route 10% of traffic to real planner
   - Compare results with mock
   - Monitor performance and accuracy

3. **Gradual Rollout** (Week 3)
   - Increase to 50% traffic
   - Monitor for issues
   - Gather user feedback

4. **Full Migration** (Week 4)
   - Route 100% to real planner
   - Deprecate mock implementation
   - Update documentation

---

## Success Criteria

- [ ] Plan generation accuracy > 85% (manual validation)
- [ ] Platform detection accuracy > 90% (automated tests)
- [ ] Average latency < 2 seconds (95th percentile)
- [ ] Cost per plan < $0.01 USD
- [ ] Zero downtime during migration
- [ ] Integration tests passing 100%
- [ ] Documentation complete
- [ ] Core Engine integration successful

---

## Dependencies

### Required Services
- Ollama (llama3 model) on port 11434
- Protocol Interface (for agent registry)
- Core Engine (for integration testing)

### Python Packages
```
fastapi>=0.100.0
pydantic>=2.0
ollama>=0.1.0
httpx>=0.24.0
structlog>=23.1.0
tiktoken>=0.5.0
uvicorn>=0.24.0
```

---

## Risks and Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| LLM response inconsistency | High | Medium | Add fallback logic, structured prompts |
| Ollama unavailable | High | Low | Health checks, graceful degradation |
| Slow LLM inference | Medium | Medium | Caching, prompt optimization |
| Wrong platform selection | Medium | Low | Extensive testing, confidence thresholds |
| Cost overruns | Low | Low | Local Ollama, cost tracking |

---

## Future Enhancements

### Phase 5+ Improvements

1. **Learning from Execution**
   - Track plan success rates
   - Learn from failures
   - Improve prompts based on feedback

2. **Multi-Step Reasoning**
   - Chain-of-thought prompting
   - Better complex query handling
   - Improved cost estimation

3. **User Feedback Integration**
   - Allow users to rate plans
   - Incorporate feedback into planning
   - Build preference models

4. **Advanced Optimization**
   - ML-based cost prediction
   - Historical query patterns
   - Workload-aware scheduling

---

## Conclusion

This implementation plan provides a clear path from the current mock Planner agent to a production-ready LLM-powered intelligent planner. The design leverages **Ollama for local, privacy-first LLM inference** while maintaining the A2A protocol compatibility established in Phase 2.

**Key Differentiators**:
- ✅ No platform specified by user - planner decides intelligently
- ✅ Multi-platform query support (can select multiple data agents)
- ✅ Context-aware planning (schema hints, user context)
- ✅ LLM-powered query understanding
- ✅ Cost-optimized workflow generation
- ✅ Fully local - no external API calls

**Estimated Timeline**: 10 working days  
**Team Size**: 1-2 developers  
**Complexity**: Moderate (LLM integration, prompt engineering)
