# LexiQuery Project Setup Plan

**Version**: 1.0  
**Date**: February 4, 2026  
**Status**: Initial Planning

---

## 1. Executive Summary

This document outlines the development plan for LexiQuery, focusing on interface-first design and core engine implementation. The architecture follows a microservices pattern with isolated Python virtual environments for each service, designed for independent containerization and deployment.

---

## 2. Project Architecture Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    API Gateway Layer                         │
│              (MCP & A2A Protocol Handlers)                   │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────────────┐
│                    Core Engine                               │
│         (Orchestration & Business Logic)                     │
└──────┬────────────┬────────────┬────────────┬───────────────┘
       │            │            │            │
   ┌───▼───┐   ┌───▼───┐   ┌───▼───┐   ┌───▼────┐
   │  SOP  │   │ Query │   │Cache/ │   │  Data  │
   │Engine │   │  Gen  │   │Learn  │   │Connec- │
   │       │   │       │   │       │   │  tors  │
   └───────┘   └───────┘   └───────┘   └────────┘
```

### 2.2 Microservices Breakdown

1. **Core Engine Service** (Priority 1 - Full Implementation)
   - Orchestration logic with CrewAI
   - Agent-based business process coordination
   - Interface definitions

2. **Protocol Interface Service** (Priority 2 - Mock then Real)
   - MCP server and client implementation
   - A2A SDK for Python integration
   - Abstract authorization provider (Google/GCP/Azure)
   - OAuth2 OBO flow

3. **SOP Engine Service** (Priority 3 - Mock then Real)
   - Document parsing
   - ChromaDB vector database integration
   - GraphRAG for schema understanding
   - Ollama embeddings for semantic search

4. **Query Generator Service** (Priority 3 - Mock then Real)
   - Multi-platform query generation with Ollama
   - GraphRAG for schema-aware queries
   - KQL, SPL, SQL support
   - CrewAI agents for query optimization

5. **Caching & Learning Service** (Priority 3 - Mock then Real)
   - Pattern storage
   - PII elimination
   - Performance optimization

6. **Data Connector Service** (Priority 3 - Mock then Real)
   - Azure Log Analytics connector
   - Splunk connector
   - SQL connector

---

## 3. Directory Structure

```
engine-core/
├── docs/
│   ├── architecture/
│   ├── api-specs/
│   └── deployment/
│
├── shared/
│   ├── interfaces/          # Shared interface definitions
│   │   ├── __init__.py
│   │   ├── sop_engine.py
│   │   ├── query_generator.py
│   │   ├── cache_service.py
│   │   ├── data_connector.py
│   │   └── protocol_handler.py
│   │
│   ├── models/              # Shared data models
│   │   ├── __init__.py
│   │   ├── query_models.py
│   │   ├── sop_models.py
│   │   └── response_models.py
│   │
│   └── utils/               # Shared utilities
│       ├── __init__.py
│       ├── logging_config.py
│       └── exceptions.py
│
├── services/
│   ├── core-engine/
│   │   ├── src/
│   │   │   ├── __init__.py
│   │   │   ├── orchestrator.py
│   │   │   ├── business_logic.py
│   │   │   └── config.py
│   │   ├── tests/
│   │   ├── requirements.txt
│   │   ├── Dockerfile
│   │   └── .venv/           # Virtual environment (gitignored)
│   │
│   ├── protocol-interface/
│   │   ├── src/
│   │   │   ├── __init__.py
│   │   │   ├── mcp_server.py
│   │   │   ├── a2a_handler.py
│   │   │   ├── auth/
│   │   │   └── mock/        # Mock implementation
│   │   ├── tests/
│   │   ├── requirements.txt
│   │   ├── Dockerfile
│   │   └── .venv/
│   │
│   ├── sop-engine/
│   │   ├── src/
│   │   │   ├── __init__.py
│   │   │   ├── parser.py
│   │   │   ├── vector_store.py
│   │   │   └── mock/        # Mock implementation
│   │   ├── tests/
│   │   ├── requirements.txt
│   │   ├── Dockerfile
│   │   └── .venv/
│   │
│   ├── query-generator/
│   │   ├── src/
│   │   │   ├── __init__.py
│   │   │   ├── generator.py
│   │   │   ├── platforms/
│   │   │   │   ├── kql.py
│   │   │   │   ├── spl.py
│   │   │   │   └── sql.py
│   │   │   └── mock/        # Mock implementation
│   │   ├── tests/
│   │   ├── requirements.txt
│   │   ├── Dockerfile
│   │   └── .venv/
│   │
│   ├── cache-learning/
│   │   ├── src/
│   │   │   ├── __init__.py
│   │   │   ├── cache_manager.py
│   │   │   ├── pii_detector.py
│   │   │   └── mock/        # Mock implementation
│   │   ├── tests/
│   │   ├── requirements.txt
│   │   ├── Dockerfile
│   │   └── .venv/
│   │
│   └── data-connectors/
│       ├── src/
│       │   ├── __init__.py
│       │   ├── azure_connector.py
│       │   ├── splunk_connector.py
│       │   ├── sql_connector.py
│       │   └── mock/        # Mock implementation
│       ├── tests/
│       ├── requirements.txt
│       ├── Dockerfile
│       └── .venv/
│
├── infra/
│   ├── docker/
│   │   └── docker-compose.yml
│   └── k8s/
│       ├── deployments/
│       └── services/
│
├── scripts/
│   ├── setup_envs.ps1
│   ├── build_all.ps1
│   └── run_tests.ps1
│
├── .gitignore
├── README.md
└── problem-statement.md
```

---

## 4. Development Phases

### Phase 1: Foundation & Interfaces (Week 1-2)

#### Objectives:
- Define all service interfaces
- Create shared data models
- Set up project structure
- Configure virtual environments

#### Deliverables:
1. **Interface Definitions**
   - `shared/interfaces/sop_engine.py` - SOP processing interface
   - `shared/interfaces/query_generator.py` - Query generation interface
   - `shared/interfaces/cache_service.py` - Caching interface
   - `shared/interfaces/data_connector.py` - Data connector interface
   - `shared/interfaces/protocol_handler.py` - Protocol handler interface

2. **Data Models**
   - Query request/response models
   - SOP document models
   - Cache entry models
   - Error/exception models

3. **Development Environment**
   - Virtual environment setup scripts
   - Docker base images
   - CI/CD pipeline configuration

#### Tasks:
- [ ] Create directory structure
- [ ] Define abstract base classes for all interfaces
- [ ] Create Pydantic models for data validation
- [ ] Set up virtual environments for each service
- [ ] Create .gitignore with venv exclusions
- [ ] Write setup automation scripts

---

### Phase 2: Core Engine Implementation (Week 3-5)

#### Objectives:
- Implement the core orchestration engine
- Build business logic layer
- Create integration points for all services

#### Deliverables:
1. **Core Engine Service**
   - Orchestrator that coordinates all services using CrewAI
   - Business process workflow engine with agent-based coordination
   - Error handling and retry logic
   - Configuration management

2. **Mock Service Implementations**
   - Mock SOP engine (returns pre-defined procedures)
   - Mock query generator (returns template queries)
   - Mock cache service (in-memory cache)
   - Mock data connectors (synthetic data responses)
   - Mock protocol handlers (simple request/response)

3. **Integration Tests**
   - End-to-end workflow tests with mocks
   - Interface contract tests
   - Error scenario tests

#### Tasks:
- [ ] Implement core orchestrator class
- [ ] Create workflow state machine
- [ ] Build service registry and dependency injection
- [ ] Implement all mock services
- [ ] Write integration tests
- [ ] Create Docker images for core + mocks
- [ ] Test containerized deployment

---

### Phase 3: Mock Validation & Testing (Week 6)

#### Objectives:
- Validate architecture with mocked services
- Performance baseline testing
- API contract validation

#### Deliverables:
1. **Validation Suite**
   - End-to-end scenario tests
   - Performance benchmarks
   - API documentation

2. **Docker Compose Environment**
   - Multi-container orchestration
   - Service discovery configuration
   - Network and security setup

#### Tasks:
- [ ] Run complete system tests with all mocks
- [ ] Document API contracts
- [ ] Create performance baselines
- [ ] Build docker-compose.yml for local development
- [ ] Generate API documentation
- [ ] Create developer onboarding guide

---

### Phase 4: Incremental Real Implementation (Week 7+)

#### Priority Order:
1. **Protocol Interface Service** (Critical Path)
   - Implement MCP server and client
   - Implement A2A protocol using A2A SDK for Python
   - Abstract authorization provider (Google/GCP/Azure)
   - OAuth2 integration interface

2. **SOP Engine Service**
   - Document parser implementation
   - ChromaDB vector database integration
   - GraphRAG for schema understanding
   - Semantic search implementation with Ollama embeddings

3. **Query Generator Service**
   - Ollama LLM integration for query generation
   - GraphRAG for schema-aware query construction
   - Platform-specific generators (KQL, SPL, SQL)
   - CrewAI agents for query optimization

4. **Data Connector Service**
   - Azure Log Analytics API integration
   - Splunk REST API integration
   - SQL database connectivity

5. **Cache & Learning Service**
   - PII detection and elimination
   - Pattern storage implementation
   - Learning algorithms

---

## 5. Python Virtual Environment Setup

### 5.1 Environment Isolation Strategy

Each microservice maintains its own isolated Python environment to:
- Manage dependencies independently
- Prevent version conflicts
- Enable parallel development
- Facilitate containerization

### 5.2 Virtual Environment Configuration

**Per-Service Setup:**
```powershell
# Navigate to service directory
cd services/<service-name>

# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 5.3 Dependency Management

**Common Dependencies (All Services):**
```
# Core
python>=3.11
pydantic>=2.0
fastapi>=0.100.0
uvicorn[standard]>=0.23.0

# Logging & Monitoring
structlog>=23.1.0
prometheus-client>=0.17.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
```

**Service-Specific Dependencies:**

- **Core Engine**: `pydantic`, `httpx`, `tenacity`, `crewai>=0.28.0`
- **Protocol Interface**: `mcp>=0.9.0`, `a2a-sdk-python`, `cryptography`
- **SOP Engine**: `chromadb>=0.4.0`, `ollama>=0.1.0`, `graphrag`, `langchain>=0.1.0`
- **Query Generator**: `ollama>=0.1.0`, `graphrag`, `sqlparse`, `langchain>=0.1.0`
- **Cache/Learning**: `redis`, `pandas`, `scikit-learn`
- **Data Connectors**: `azure-monitor-query`, `splunk-sdk`, `sqlalchemy`, `httpx`

---

## 6. Containerization Strategy

### 6.1 Base Image Standards

**Python Base Image:**
```dockerfile
FROM python:3.11-slim

# Common setup for all services
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
```

### 6.2 Multi-Stage Builds

Each service uses multi-stage builds:
1. **Builder stage**: Install dependencies
2. **Runtime stage**: Copy artifacts, minimal footprint

### 6.3 Container Registry Organization

```
registry.company.com/lexiquery/
├── core-engine:latest
├── protocol-interface:latest
├── sop-engine:latest
├── query-generator:latest
├── cache-learning:latest
└── data-connectors:latest
```

---

## 7. Interface Design Principles

### 7.1 Abstract Base Classes

All interfaces defined as Python ABCs:
```python
from abc import ABC, abstractmethod
from typing import Protocol

class SOPEngineInterface(ABC):
    @abstractmethod
    async def process_sop(self, document: SOPDocument) -> ProcessedSOP:
        pass
    
    @abstractmethod
    async def search_procedures(self, query: str) -> List[Procedure]:
        pass
```

### 7.2 Data Models

Use Pydantic for validation:
```python
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    natural_language: str = Field(..., description="User's query")
    context: Optional[Dict] = None
    platform: str = Field(..., description="Target platform: kql|spl|sql")
```

### 7.3 Error Handling

Standardized exception hierarchy:
```python
class LexiQueryException(Exception):
    pass

class ServiceUnavailableException(LexiQueryException):
    pass

class QueryGenerationException(LexiQueryException):
    pass
```

---

## 8. Testing Strategy

### 8.1 Test Levels

1. **Unit Tests**: Each service independently (95% coverage target)
2. **Interface Tests**: Contract validation between services
3. **Integration Tests**: Multi-service workflows with mocks
4. **E2E Tests**: Full system tests (post-real implementation)

### 8.2 Mock Testing Approach

During Phase 2, all services except core engine use mocks:
```python
class MockSOPEngine(SOPEngineInterface):
    async def search_procedures(self, query: str) -> List[Procedure]:
        return [
            Procedure(id="1", title="Standard Incident Response", 
                     steps=["Step 1", "Step 2"])
        ]
```

---

## 9. Development Workflow

### 9.1 Setup New Service

```powershell
# Run setup script
.\scripts\setup_envs.ps1 -ServiceName "new-service"

# Manual steps:
# 1. Create service directory under services/
# 2. Create virtual environment
# 3. Create requirements.txt
# 4. Implement interface from shared/interfaces/
# 5. Create Dockerfile
# 6. Add to docker-compose.yml
```

### 9.2 Development Cycle

1. Define interface in `shared/interfaces/`
2. Create mock implementation
3. Build core engine integration
4. Write integration tests
5. Validate with docker-compose
6. Implement real service logic
7. Replace mock with real implementation
8. Re-run all tests

---

## 10. Configuration Management

### 10.1 Environment Variables

Each service uses `.env` files:
```
# Core Engine
CORE_ENGINE_PORT=8000
LOG_LEVEL=INFO
ENABLE_METRICS=true

# Service Discovery
SOP_ENGINE_URL=http://sop-engine:8001
QUERY_GENERATOR_URL=http://query-generator:8002
```

### 10.2 Secrets Management

- Development: `.env` files (gitignored)
- Production: Azure Key Vault / HashiCorp Vault
- Docker: Environment variables + secrets

---

## 11. Monitoring & Observability

### 11.1 Logging Standards

- Structured JSON logging (structlog)
- Correlation IDs across services
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

### 11.2 Metrics

- Prometheus metrics exposed on `/metrics`
- Key metrics: request rate, latency, error rate
- Service health checks on `/health`

---

## 12. Next Steps

### Immediate Actions (Week 1):

1. **Day 1-2: Project Setup**
   - [ ] Create directory structure
   - [ ] Initialize git repository
   - [ ] Set up .gitignore

2. **Day 3-5: Interface Definition**
   - [ ] Define all service interfaces
   - [ ] Create shared data models
   - [ ] Write interface documentation

3. **Day 6-7: Environment Setup**
   - [ ] Create virtual environments for all services
   - [ ] Write automation scripts
   - [ ] Set up Docker base images

4. **Day 8-10: Mock Services**
   - [ ] Implement all mock services
   - [ ] Create docker-compose.yml
   - [ ] Initial integration tests

### Success Criteria:

- ✅ All interfaces defined and documented
- ✅ Core engine can orchestrate all mocked services
- ✅ Docker containers for all services build successfully
- ✅ docker-compose brings up entire system
- ✅ Basic end-to-end workflow completes with mocks
- ✅ Integration tests passing at >80% coverage

---

## 13. Risk Management

### Technical Risks:

1. **Ollama Model Performance**: Mitigated by mock-first approach and model selection testing
2. **GraphRAG Schema Extraction**: Isolated service boundary, fallback to direct parsing
3. **CrewAI Agent Coordination**: Benchmark with mock data first, incremental complexity
4. **ChromaDB Scalability**: Test with production-scale data early
5. **Abstract Auth Provider**: Design flexible interface, prototype early in Phase 4
6. **A2A Protocol Integration**: Validate with SDK examples during Phase 2

### Mitigation Strategies:

- Interface-first design allows parallel development
- Mock services de-risk integration complexity
- Containerization enables consistent environments
- Incremental real implementation reduces big-bang risk

---

## 14. Appendix

### A. Technology Decisions

| Component | Technology Choice | Rationale |
|-----------|------------------|-----------|
| Language | Python 3.11+ | Rich ML/AI ecosystem, async support |
| API Framework | FastAPI | High performance, automatic docs, async |
| Vector DB | ChromaDB | Open-source, local-first, privacy-focused |
| LLM | Ollama | Private deployment, no external API calls |
| Agentic Framework | CrewAI | Multi-agent orchestration, role-based |
| Query Intelligence | GraphRAG | Schema-aware query generation |
| Protocol Layer | MCP + A2A SDK | Standard AI integration protocols |
| Auth Provider | Abstract Interface | Pluggable: Google/GCP/Azure |
| Cache | Redis | Fast, distributed, proven |
| Container Runtime | Docker | Industry standard |
| Orchestration | Kubernetes | Scalability, self-healing |

### B. Resource Requirements

**Development Environment:**
- Python 3.11+
- Docker Desktop
- 16GB RAM minimum
- Visual Studio Code / PyCharm

**Production Environment:**
- Kubernetes cluster
- Ollama deployment (local or self-hosted)
- ChromaDB cluster
- Azure/GCP/Google Cloud subscription (for data sources, depending on auth provider)
- Splunk Enterprise instance
- Redis cluster

### C. Reference Documentation

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [A2A SDK Documentation](https://github.com/a2a-protocol/a2a-sdk-python)
- [CrewAI Documentation](https://docs.crewai.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [GraphRAG Documentation](https://microsoft.github.io/graphrag/)
- [Ollama Documentation](https://ollama.ai/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-04 | GitHub Copilot | Initial project setup plan |

