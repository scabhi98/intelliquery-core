# Phase 2 Implementation Complete

## Overview

Phase 2 implements the **A2A + MCP Hybrid Architecture** with a thin orchestration layer coordinating external agents. All components are fully functional with mock implementations ready for Phase 4 enhancement with real LLMs and MCP tools.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Core Engine (Port 8000)                  │
│                   A2A Orchestrator + MCP Server              │
└──────────────┬──────────────────────────────────────────────┘
               │
               │ A2A Protocol
               │
       ┌───────┴────────┬────────────┬──────────────┐
       │                │            │              │
       ▼                ▼            ▼              ▼
┌─────────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐
│   Planner   │  │Knowledge │  │Knowledge│  │   Data   │
│   Agent     │  │Provider  │  │Provider │  │  Agents  │
│  (9000)     │  │  SOP     │  │  Error  │  │KQL/SPL   │
│             │  │ (9010)   │  │ (9011)  │  │SQL       │
│             │  │          │  │         │  │9020-9022 │
└─────────────┘  └──────────┘  └─────────┘  └──────────┘
```

## Components Implemented

### 1. Core Engine Service (Port 8000)
**Location**: `services/core-engine/src/`

**Key Files**:
- `orchestrator.py` - A2AOrchestrator with complete workflow logic
- `main.py` - FastAPI service with MCP server interface
- `config.py` - Configuration management

**Capabilities**:
- Agent discovery and invocation
- Workflow orchestration (9-step process)
- Journey-level cost tracking
- State machine management
- MCP server interface (Phase 2 placeholder)

**API Endpoints**:
- `POST /api/v1/query` - Process query through A2A workflow
- `GET /api/v1/journey/{id}` - Get journey context
- `GET /api/v1/journey/{id}/cost` - Get cost summary
- `POST /mcp/tools/list` - List MCP tools
- `POST /mcp/tools/call` - Execute MCP tool

### 2. Mock Planner Agent (Port 9000)
**Location**: `services/mock-agents/planner/src/`

**Capabilities**:
- Analyzes natural language queries
- Creates workflow plans with knowledge + data steps
- Detects error keywords to include error knowledge agent
- Returns estimated costs and durations

**A2A Actions**:
- `create_plan` - Generate workflow plan
- `optimize_plan` - Optimize existing plan (Phase 4)

### 3. Mock Knowledge Provider Agents

#### SOP Knowledge Agent (Port 9010)
**Location**: `services/mock-agents/knowledge/src/main.py:app_sop`

**Capabilities**:
- Retrieves relevant SOP procedures
- Keyword-based matching (Phase 2)
- Returns table/column hints for query generation
- Mock data includes authentication, performance, database SOPs

**A2A Actions**:
- `retrieve_sop_context` - Get relevant procedures
- `search_procedures` - Search by tags

#### Error Knowledge Agent (Port 9011)
**Location**: `services/mock-agents/knowledge/src/main.py:app_error`

**Capabilities**:
- Retrieves known error patterns
- Returns resolution steps and query suggestions
- Mock data includes auth errors, network errors

**A2A Actions**:
- `retrieve_error_context` - Get relevant errors

### 4. Mock Data Agents

#### KQL Data Agent (Port 9020)
**Location**: `services/mock-agents/data/src/main.py:app_kql`

**Capabilities**:
- Generates Azure Log Analytics KQL queries
- Uses SOP table hints
- Template-based generation (Phase 2)
- Returns confidence scores

#### SPL Data Agent (Port 9021)
**Location**: `services/mock-agents/data/src/main.py:app_spl`

**Capabilities**:
- Generates Splunk SPL queries
- Index and sourcetype selection
- Stats and aggregations

#### SQL Data Agent (Port 9022)
**Location**: `services/mock-agents/data/src/main.py:app_sql`

**Capabilities**:
- Generates standard SQL queries
- Time-based filtering
- GROUP BY aggregations

**A2A Actions** (all data agents):
- `generate_and_execute` - Generate (and optionally execute) query

## Data Models

### Workflow Models
**Location**: `shared/models/workflow_models.py`

- `WorkflowState` - 8-state enum (PENDING → COMPLETED/FAILED)
- `StateTransition` - State change tracking
- `WorkflowError` - Error tracking with retry support
- `WorkflowStep` - Individual workflow step
- `WorkflowPlan` - Complete execution plan from Planner

### A2A Protocol Models
**Location**: `shared/models/a2a_models.py`

- `A2AAgentDescriptor` - Agent discovery/registration
- `A2ATaskRequest` - Task invocation request
- `A2ATaskResponse` - Task response with cost info
- `KnowledgeContext` - Knowledge from provider agents
- `QueryResult` - Query generation/execution result
- `MCPToolCall` - MCP tool invocation tracking

### Journey Models
**Location**: `shared/models/journey_models.py`

- `JourneyContext` - Complete journey state with:
  - State history
  - Cost aggregation
  - Agent results
  - Error tracking
  - Helper methods for state management

### Cost Models
**Location**: `shared/models/cost_models.py`

- `AgentCostInfo` - Per-agent cost breakdown
- `JourneyCostSummary` - Journey-level aggregation with:
  - Cost by agent
  - Cost by service type
  - LLM/MCP call counts
  - Token usage

## Running Phase 2

### Local Development (Recommended)

Phase 2 uses a **single root-level virtual environment** for easy VS Code integration and development.

**1. Setup Development Environment**:
```powershell
# Run the setup script to create .venv at root and install all dependencies
.\scripts\setup_envs.ps1

# This will:
# - Create .venv/ at project root
# - Install shared dependencies
# - Install all service dependencies
# - Build wheel distributions (optional)
```

**2. Start All Services (Easy Mode)**:
```powershell
# Start all 7 services at once in separate windows
.\scripts\run_services.ps1

# Check service status
.\scripts\run_services.ps1 -Status

# Stop all services
.\scripts\run_services.ps1 -Stop
```

**3. Start Services Manually (Individual Control)**:
```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Start services using Python module syntax
python -m uvicorn services.core_engine.src.main:app --port 8000 --reload
python -m uvicorn services.mock_agents.planner.src.main:app --port 9000 --reload
python -m uvicorn services.mock_agents.knowledge.src.main:app_sop --port 9010 --reload
python -m uvicorn services.mock_agents.knowledge.src.main:app_error --port 9011 --reload
python -m uvicorn services.mock_agents.data.src.main:app_kql --port 9020 --reload
python -m uvicorn services.mock_agents.data.src.main:app_spl --port 9021 --reload
python -m uvicorn services.mock_agents.data.src.main:app_sql --port 9022 --reload
```

**4. VS Code Integration**:
```powershell
# VS Code will automatically detect .venv at root
# Select interpreter: Ctrl+Shift+P -> "Python: Select Interpreter"
# Choose: .\.venv\Scripts\python.exe

# All services share the same environment - no conflicts!
```

### Docker Compose (Recommended)

**1. Build All Services**:
```powershell
docker-compose build
```

**2. Start All Services**:
```powershell
docker-compose up -d
```

**3. View Logs**:
```powershell
docker-compose logs -f core-engine
```

**4. Stop All Services**:
```powershell
docker-compose down
```

## Testing Phase 2

### Manual API Testing

**Test Query Processing**:
```powershell
curl -X POST http://localhost:8000/api/v1/query `
  -H "Content-Type: application/json" `
  -d '{
    "natural_language": "Show failed login attempts in the last hour",
    "platform": "kql",
    "user_id": "test_user"
  }'
```

**Expected Response**:
```json
{
  "journey_id": "uuid",
  "workflow_plan": {
    "plan_id": "plan_uuid",
    "steps": [...],
    "estimated_cost_usd": 0.01
  },
  "knowledge_context": {
    "sop": {...}
  },
  "queries": {
    "kql": "SigninLogs\n| where TimeGenerated > ago(1h)\n..."
  },
  "query_results": {
    "kql": {
      "platform": "kql",
      "query": "...",
      "confidence": 0.78
    }
  },
  "overall_confidence": 0.78,
  "cost_summary": {
    "total_cost_usd": 0.016,
    "cost_by_agent": {
      "mock_workflow_planner": 0.002,
      "mock_sop_knowledge": 0.003,
      "mock_kql_data": 0.006
    }
  }
}
```

### Automated Tests

**Run Integration Tests**:
```powershell
cd services/core-engine
pytest tests/test_integration.py -v
```

**Test Coverage**:
- Journey initialization
- Agent discovery
- Planner invocation
- Knowledge agent invocation
- Data agent invocation
- Complete workflow orchestration
- State machine transitions
- Cost aggregation
- Multi-platform support

## Workflow Execution

### Complete Journey Flow

1. **PENDING** → User submits query
2. **DISCOVERING** → Orchestrator discovers A2A agents
3. **PLANNING** → Planner agent creates workflow plan
4. **GATHERING_KNOWLEDGE** → Knowledge agents retrieve context (parallel)
5. **GENERATING_QUERIES** → Data agents generate platform queries
6. **VALIDATING** → Orchestrator validates results
7. **COMPLETED** → Return comprehensive response

### Cost Tracking

**Per-Agent Granularity**:
- LLM calls and token usage
- MCP calls (ChromaDB, Azure LA, Splunk, etc.)
- Embedding operations
- Execution time

**Journey-Level Aggregation**:
- Total cost across all agents
- Cost breakdown by agent
- Cost breakdown by service type (planner/knowledge/data)

## Phase 2 vs Phase 4

### Phase 2 (Current - Mock Mode)
- ✅ Template-based query generation
- ✅ Keyword matching for knowledge retrieval
- ✅ Hardcoded agent endpoints
- ✅ No actual query execution
- ✅ No real LLM usage
- ✅ No MCP tool servers

### Phase 4 (Future - Production)
- 🔄 Ollama LLM for intelligent generation
- 🔄 ChromaDB + embeddings for semantic search
- 🔄 GraphRAG for schema understanding
- 🔄 A2A SDK for dynamic agent discovery
- 🔄 Real MCP servers (ChromaDB, Azure LA, Splunk)
- 🔄 Actual query execution
- 🔄 Learning from successful queries

## Directory Structure

```
services/
├── core-engine/
│   ├── src/
│   │   ├── main.py           # FastAPI + MCP server
│   │   ├── orchestrator.py   # A2A orchestration logic
│   │   └── config.py         # Configuration
│   ├── tests/
│   │   └── test_integration.py
│   ├── Dockerfile
│   └── requirements.txt
├── mock-agents/
│   ├── planner/
│   │   ├── src/main.py       # Workflow planner
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── knowledge/
│   │   ├── src/main.py       # SOP + Error agents
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── data/
│       ├── src/main.py       # KQL/SPL/SQL agents
│       ├── Dockerfile
│       └── requirements.txt
shared/
├── models/
│   ├── workflow_models.py    # State machine
│   ├── a2a_models.py         # A2A protocol
│   ├── journey_models.py     # Journey tracking
│   ├── cost_models.py        # Cost tracking
│   ├── query_models.py       # Query request/response
│   └── ...
├── interfaces/
│   └── ...
└── utils/
    ├── logging_config.py     # Structured logging
    └── exceptions.py         # Exception hierarchy
```

## Key Features Demonstrated

✅ **A2A Protocol**: Agents communicate through standardized interface
✅ **Cost Transparency**: Every operation tracked with journey-level aggregation
✅ **State Machine**: Clear workflow progression with history
✅ **Agent Discovery**: Orchestrator discovers required agents
✅ **Knowledge Integration**: SOP and error knowledge inform query generation
✅ **Multi-Platform**: KQL, SPL, SQL query generation
✅ **Error Handling**: Comprehensive exception hierarchy
✅ **Logging**: Structured JSON logs with context
✅ **Dockerized**: All services containerized and orchestrated

## Next Steps

After Phase 2 validation:
1. Review architecture and mock behavior
2. Validate cost tracking accuracy
3. Test with various query types
4. Proceed to Phase 3: Real MCP servers
5. Proceed to Phase 4: Ollama + GraphRAG integration

## Troubleshooting

**Port Conflicts**:
```powershell
# Check ports in use
netstat -ano | findstr "8000 9000 9010 9011 9020 9021 9022"
```

**Service Not Starting**:
```powershell
# Check logs
docker-compose logs [service-name]
```

**Import Errors**:
```powershell
# Ensure PYTHONPATH includes project root
$env:PYTHONPATH="J:\projects\engine-core"
```

## Contact

For questions or issues with Phase 2 implementation, refer to:
- Architecture: `docs/PHASE2_IMPLEMENTATION_PLAN.md`
- Copilot Instructions: `.github/copilot-instructions.md`
