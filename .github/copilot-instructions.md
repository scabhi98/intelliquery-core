# GitHub Copilot Instructions for LexiQuery Project

## Project Overview

**LexiQuery** is a privacy-first, multi-agent query intelligence platform that orchestrates specialized A2A agents to generate optimized queries across Azure Log Analytics (KQL), Splunk (SPL), and SQL platforms.

**Core Architecture**: Thin orchestrator coordinates autonomous agents via A2A protocol - Planner decides workflow, Knowledge agents provide SOP context, Data agents generate platform-specific queries.

**Key Principle**: Fully local deployment using Ollama for LLMs - no external APIs, complete privacy.

---

## Current Implementation Status (February 2026)

### ✅ Completed Services

1. **Core Engine** (Port 8000)
   - A2A orchestrator with journey lifecycle management
   - Discovery → Planning → Knowledge → Data → Validation workflow
   - Platform-agnostic: User provides query, planner selects appropriate data agents
   - Files: `services/core_engine/src/orchestrator.py`, `main.py`

2. **Protocol Interface** (Port 8001)
   - MCP tool registry and handlers
   - A2A task invocation and discovery
   - Persistent JSON agent registry (`./data/agent_registry.json`)
   - Auth provider framework (abstract, Google, Azure, GCP placeholders)
   - Files: `services/protocol_interface/src/main.py`, `a2a/registry.py`, `a2a/handler.py`

3. **SOP Engine** (Port 8002)
   - Document parsing (Markdown, TXT, DOCX, PDF)
   - ChromaDB + Ollama embeddings (`nomic-embed-text`)
   - Semantic search and schema extraction
   - File upload endpoint: `/api/v1/procedures/upload`
   - Files: `services/sop_engine/src/service.py`, `parser/`, `vector_store/`

4. **Mock A2A Agents** (Ports 9000-9022)
   - **Planner** (9000): Creates workflow plans based on available agent capabilities
   - **Knowledge** (9010-9011): SOP procedures and known errors
   - **Data** (9020-9022): KQL, SPL, SQL query generators with templates
   - Files: `services/mock_agents/{planner,knowledge,data}/src/main.py`

### 📦 Shared Models & Interfaces

Located in `shared/`:
- **Models**: `query_models.py`, `a2a_models.py`, `journey_models.py`, `workflow_models.py`, `cost_models.py`
- **Interfaces**: Abstract base classes for all services (not yet fully implemented)
- **Utils**: `exceptions.py`, `logging_config.py`

---

## Architecture Guidelines

### A2A Agent Communication Pattern

```python
# Discovery → Select → Invoke → Parse Response
agents = await discover_agents_via_protocol()  # Protocol Interface /a2a/agents/discover
planner = select_agent_by_capability(agents, "create_plan")

task = A2ATaskRequest(
    journey_id=journey_id,
    agent_id=planner.agent_id,
    action="create_plan",
    parameters={"natural_language": query, "execution_context": {...}}
)

response = await invoke_agent_task(planner, task)  # POST /a2a/task/invoke
plan = WorkflowPlan(**response.result["plan"])
```

### Planner Context (NEW)

When invoking planner, provide:
- **Available agents** with capabilities (planner/knowledge/data)
- **Execution context**: journey state, schema hints, user context
- **NO platform** from user - planner decides which data agents to use

```python
execution_context = {
    "available_agents": {
        "knowledge": [{"agent_id": "...", "capabilities": [...], "knowledge_domain": "sop"}],
        "data": [{"agent_id": "...", "capabilities": [...], "platform": "kql"}]
    },
    "current_state": "planning",
    "user_context": {...}
}
```

### Service Structure

```
services/<service-name>/
├── src/
│   ├── main.py          # FastAPI entry point
│   ├── config.py        # Pydantic Settings
│   ├── service.py       # Core logic
│   ├── mock/            # Mock implementations (deprecated after Phase 4)
│   └── ...
├── tests/
│   ├── __init__.py
│   └── test_*.py
├── requirements.txt
├── Dockerfile
└── .venv/              # Virtual environment (gitignored)
```

---

## Technology Stack

### Core
- **Python**: 3.11+
- **Framework**: FastAPI + Uvicorn
- **Async**: httpx, asyncio
- **Validation**: Pydantic v2
- **Logging**: structlog with JSON output

### AI/ML
- **LLM**: Ollama (`llama3`, local on port 11434)
- **Embeddings**: Ollama (`nomic-embed-text`)
- **Vector DB**: ChromaDB (persistent, local)
- **Graph**: GraphRAG (lightweight schema extraction, not yet full implementation)

### Protocol
- **A2A**: Agent-to-Agent HTTP-based protocol
- **MCP**: Model Context Protocol for tool invocation
- **Registry**: JSON file-based (`protocol_interface/data/agent_registry.json`)

### Dependencies (Core)
```
fastapi>=0.100.0
pydantic>=2.0
pydantic-settings>=2.0.0
httpx>=0.24.0
structlog>=23.1.0
ollama>=0.1.0
chromadb>=0.4.0
python-multipart>=0.0.6
```

---

## Coding Conventions

### Python Style
- **PEP 8** and **PEP 257** (docstrings)
- **Type hints** everywhere: `async def func(x: int) -> str:`
- **async/await** for all I/O operations
- **Pydantic models** for all data validation
- **Interface-first**: Define ABC before implementation

### File Naming
- `snake_case.py` for modules
- `PascalCase` for classes
- `UPPER_CASE` for constants

### Import Organization
```python
# Standard library
import asyncio
from typing import List, Optional
from uuid import UUID

# Third-party
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import structlog

# Shared
from shared.models.query_models import QueryRequest
from shared.models.a2a_models import A2ATaskRequest

# Local
from .config import settings
from .service import MyService
```

---

## Essential Patterns

### FastAPI Service Template

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import structlog

from .config import Settings
from .service import MyService

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    app.state.settings = Settings()
    app.state.service = MyService(app.state.settings)
    await app.state.service.initialize()
    logger.info("Service started", service=app.state.settings.service_name)
    yield
    await app.state.service.cleanup()

app = FastAPI(title="My Service", version="1.0.0", lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "healthy", "service": app.state.settings.service_name}

@app.post("/api/v1/process")
async def process(request: MyRequest) -> MyResponse:
    try:
        result = await app.state.service.process(request)
        return result
    except Exception as e:
        logger.error("Processing failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
```

### Pydantic Settings

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        case_sensitive=False
    )
    
    service_name: str = "my-service"
    service_port: int = 8000
    log_level: str = "INFO"
    
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    
    protocol_interface_url: str = "http://localhost:8001"
```

### Ollama Client

```python
import ollama

class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3"):
        self.model = model
        ollama.set_host(host)
    
    async def generate(self, prompt: str, system: Optional[str] = None) -> str:
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            system=system
        )
        return response['response']
    
    async def embed(self, text: str) -> List[float]:
        response = ollama.embeddings(
            model="nomic-embed-text",
            prompt=text
        )
        return response['embedding']
```

### ChromaDB Vector Store

```python
import chromadb
from chromadb.utils import embedding_functions

class VectorStore:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=chromadb.Settings(anonymized_telemetry=False)
        )
        
        self.embedding_fn = embedding_functions.OllamaEmbeddingFunction(
            model_name="nomic-embed-text",
            url="http://localhost:11434/api/embeddings"
        )
        
        self.collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_fn
        )
    
    async def add(self, ids: List[str], documents: List[str], metadatas: List[dict]):
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
    
    async def search(self, query: str, n_results: int = 5) -> List[dict]:
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        return results
```

### Error Handling

```python
from shared.utils.exceptions import LexiQueryException

class ServiceUnavailableException(LexiQueryException):
    """Raised when a service is unavailable."""
    pass

# Usage
try:
    response = await client.get(url)
    response.raise_for_status()
except httpx.HTTPError as e:
    raise ServiceUnavailableException(
        f"Service call failed: {url}",
        details={"error": str(e)}
    )
```

### Structured Logging

```python
import structlog

logger = structlog.get_logger()

# Log with context
logger.info(
    "query_processed",
    journey_id=str(journey_id),
    platform="kql",
    confidence=0.87,
    cost_usd=0.005
)

# Log errors with exc_info
try:
    result = await process()
except Exception as e:
    logger.error("Processing failed", error=str(e), exc_info=True)
    raise
```

---

## Testing Guidelines

### Unit Test Pattern

```python
import pytest
from unittest.mock import AsyncMock

@pytest.fixture
async def service():
    """Create service instance for testing."""
    svc = MyService(Settings())
    await svc.initialize()
    yield svc
    await svc.cleanup()

@pytest.mark.asyncio
async def test_process(service):
    """Test processing logic."""
    request = MyRequest(data="test")
    response = await service.process(request)
    
    assert response.status == "success"
    assert response.result is not None
```

### Integration Test Pattern

```python
from httpx import AsyncClient
from .main import app

@pytest.mark.asyncio
async def test_api_endpoint():
    """Test API endpoint end-to-end."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/process",
            json={"data": "test"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
```

---

## Anti-Patterns to AVOID

### ❌ External LLM APIs
```python
# BAD
import openai
client = openai.OpenAI(api_key="...")

# GOOD
import ollama
response = ollama.generate(model="llama3", prompt="...")
```

### ❌ Hardcoded Config
```python
# BAD
DATABASE_URL = "postgresql://localhost:5432/db"

# GOOD
from .config import settings
database_url = settings.database_url
```

### ❌ Missing Type Hints
```python
# BAD
async def process(request):
    return generate(request)

# GOOD
async def process(request: QueryRequest) -> QueryResponse:
    return await generate(request)
```

### ❌ Blocking I/O
```python
# BAD
import requests
response = requests.get(url)

# GOOD
import httpx
async with httpx.AsyncClient() as client:
    response = await client.get(url)
```

---

## Quick Reference

### Current Services & Ports
- **Core Engine**: 8000 (A2A orchestrator)
- **Protocol Interface**: 8001 (MCP + A2A + registry)
- **SOP Engine**: 8002 (document parsing + ChromaDB)
- **Mock Planner**: 9000 (workflow planning)
- **Mock Knowledge**: 9010-9011 (SOP + errors)
- **Mock Data**: 9020-9022 (KQL, SPL, SQL)

### Key Endpoints
```
POST /api/v1/query                     # Core Engine: Process query (no platform needed)
GET  /a2a/agents/discover              # Protocol Interface: Discover agents
POST /a2a/agents/register              # Protocol Interface: Register agent by URI
POST /a2a/task/invoke                  # Protocol Interface: Invoke agent task
POST /api/v1/procedures/upload         # SOP Engine: Upload document
POST /api/v1/procedures/search         # SOP Engine: Semantic search
```

### Important Files
- **Orchestrator**: `services/core_engine/src/orchestrator.py`
- **Agent Registry**: `services/protocol_interface/src/a2a/registry.py`
- **SOP Service**: `services/sop_engine/src/service.py`
- **Shared Models**: `shared/models/*.py`
- **Config**: Each service has `src/config.py`

### Environment Variables
```bash
# Core Engine
CORE_ENGINE_PROTOCOL_INTERFACE_URL=http://localhost:8001
CORE_ENGINE_MAX_CONCURRENT_AGENTS=5

# Protocol Interface
A2A_REGISTRY_FILE=./data/agent_registry.json
A2A_STATIC_AGENTS_JSON='[]'

# SOP Engine
OLLAMA_HOST=http://localhost:11434
CHROMA_PERSIST_DIR=./chroma_db
```

---

## Key Principles for Code Generation

1. ✅ **Interface-first**: Define ABC before implementation
2. ✅ **Pydantic validation**: All data models inherit BaseModel
3. ✅ **Type hints**: Every function parameter and return
4. ✅ **Async I/O**: Use httpx, not requests; await all I/O
5. ✅ **Local Ollama**: Never use external LLM APIs
6. ✅ **Structured logs**: Use structlog with context fields
7. ✅ **Error handling**: Try/except with custom exceptions
8. ✅ **Config via env**: Pydantic Settings with .env support
9. ✅ **Agent registry**: Persistent JSON file storage
10. ✅ **Platform-agnostic**: User doesn't specify platform, planner decides

**Privacy First**: All AI operations use local Ollama - no external APIs, no data leakage.

**Agent-Driven**: Thin orchestrator, intelligent agents make decisions based on capabilities and context.

---

## Architecture Guidelines

### Microservices Pattern
- Each service is **independently deployable** with its own Python virtual environment
- Services communicate through **well-defined interfaces** (abstract base classes)
- Follow **interface-first design**: define interfaces before implementation
- Build **mock implementations first**, then replace with real logic incrementally

### Service Structure
```
services/<service-name>/
├── src/               # Source code
│   ├── __init__.py
│   ├── main.py       # Service entry point
│   ├── mock/         # Mock implementations
│   └── ...
├── tests/            # Unit and integration tests
├── requirements.txt  # Service-specific dependencies
├── Dockerfile        # Container image definition
└── .venv/           # Virtual environment (gitignored)
```

### Core Services
1. **core-engine**: Orchestration using CrewAI agents with cost tracking
2. **protocol-interface**: MCP server/client + A2A SDK integration
3. **sop-engine**: Document parsing, ChromaDB, GraphRAG
4. **query-generator**: Ollama + GraphRAG for query generation
5. **cache-learning**: Pattern storage with PII elimination
6. **data-connectors**: Platform-specific API integrations
7. **cost-tracker**: LLM and embedding cost tracing per journey

---

## Technology Stack

### Core Technologies
- **Language**: Python 3.11+
- **Framework**: FastAPI for all service APIs
- **Agent Framework**: CrewAI for orchestration
- **Vector DB**: ChromaDB (local, privacy-focused)
- **LLM**: Ollama (self-hosted, no external APIs)
- **Query Intelligence**: GraphRAG for schema-aware generation
- **Protocol Layer**: MCP + A2A SDK for Python
- **Cache**: Redis
- **Auth**: Abstract provider (pluggable Google/GCP/Azure)

### Key Libraries
```python
# Always use these versions or higher
fastapi>=0.100.0
pydantic>=2.0
crewai>=0.28.0
chromadb>=0.4.0
ollama>=0.1.0
graphrag
mcp>=0.9.0
a2a-sdk-python
langchain>=0.1.0
structlog>=23.1.0  # Structured logging with cost tracking
tiktoken>=0.5.0    # Token counting for cost estimation
```

---

## Coding Conventions

### Python Style
- Follow **PEP 8** and **PEP 257** (docstrings)
- Use **type hints** everywhere
- Prefer **async/await** for I/O operations
- Use **Pydantic models** for all data validation
- Follow **SOLID principles**

### File Naming
- Snake_case for modules: `query_generator.py`
- PascalCase for classes: `QueryGenerator`
- Constants in UPPER_CASE: `MAX_RETRY_ATTEMPTS`

### Import Organization
```python
# Standard library
import os
from typing import List, Optional

# Third-party
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Local - shared interfaces
from shared.interfaces.query_generator import QueryGeneratorInterface
from shared.models.query_models import QueryRequest, QueryResponse

# Local - service modules
from .config import settings
```

---

## Interface Design Patterns

### Always Use Abstract Base Classes
```python
from abc import ABC, abstractmethod
from typing import Protocol

class SOPEngineInterface(ABC):
    """Interface for SOP processing and semantic search."""
    
    @abstractmethod
    async def process_sop(self, document: SOPDocument) -> ProcessedSOP:
        """Parse and store SOP document in vector database.
        
        Args:
            document: SOP document with content and metadata
            
        Returns:
            ProcessedSOP with embedding IDs and status
        """
        pass
    
    @abstractmethod
    async def search_procedures(
        self, 
        query: str, 
        top_k: int = 5
    ) -> List[Procedure]:
        """Semantic search for relevant procedures.
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            
        Returns:
            List of relevant procedures with similarity scores
        """
        pass
```

### Pydantic Models for Data Validation
```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Literal

class QueryRequest(BaseModel):
    """Request model for query generation."""
    
    natural_language: str = Field(
        ..., 
        description="User's natural language query",
        min_length=1
    )
    platform: Literal["kql", "spl", "sql"] = Field(
        ..., 
        description="Target query platform"
    )
    context: Optional[dict] = Field(
        None, 
        description="Additional context from SOP or previous queries"
    )
    schema_hints: Optional[dict] = Field(
        None,
        description="GraphRAG schema information"
    )
    
    @validator('natural_language')
    def validate_query(cls, v):
        if len(v.strip()) == 0:
            raise ValueError("Query cannot be empty")
        return v.strip()
```

---

## Service Implementation Patterns

### FastAPI Service Template
```python
from fastapi import FastAPI, HTTPException, Depends
from contextlib import asynccontextmanager

from shared.interfaces.query_generator import QueryGeneratorInterface
from shared.models.query_models import QueryRequest, QueryResponse
from .config import settings
from .generator import QueryGenerator

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage service lifecycle."""
    # Startup
    app.state.generator = QueryGenerator(settings)
    await app.state.generator.initialize()
    yield
    # Shutdown
    await app.state.generator.cleanup()

app = FastAPI(
    title="Query Generator Service",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/generate", response_model=QueryResponse)
async def generate_query(
    request: QueryRequest,
    generator: QueryGeneratorInterface = Depends(lambda: app.state.generator)
):
    """Generate platform-specific query from natural language."""
    try:
        result = await generator.generate(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "query-generator"}
```

### Mock Implementation Pattern
```python
from shared.interfaces.sop_engine import SOPEngineInterface
from shared.models.sop_models import SOPDocument, ProcessedSOP, Procedure

class MockSOPEngine(SOPEngineInterface):
    """Mock implementation for testing and development."""
    
    def __init__(self):
        self._procedures = self._load_mock_data()
    
    async def process_sop(self, document: SOPDocument) -> ProcessedSOP:
        """Return mock processed SOP."""
        return ProcessedSOP(
            document_id=document.id,
            status="processed",
            embedding_ids=["mock_emb_1", "mock_emb_2"],
            chunks_count=5
        )
    
    async def search_procedures(self, query: str, top_k: int = 5) -> List[Procedure]:
        """Return mock procedures."""
        # Simulate semantic search with keyword matching
        results = [p for p in self._procedures if query.lower() in p.title.lower()]
        return results[:top_k]
    
    def _load_mock_data(self) -> List[Procedure]:
        """Load predefined mock procedures."""
        return [
            Procedure(
                id="sop_001",
                title="Incident Response: Service Outage",
                steps=["Identify affected service", "Check logs", "Escalate if needed"],
                tags=["incident", "outage"]
            ),
            # ... more mock data
        ]
```

---

## CrewAI Integration Patterns

### Define Agents for Orchestration
```python
from crewai import Agent, Task, Crew
from langchain_ollama import OllamaLLM

class QueryOrchestrator:
    """Orchestrate query generation using CrewAI agents."""
    
    def __init__(self, ollama_model: str = "llama3"):
        self.llm = OllamaLLM(model=ollama_model)
        self._setup_agents()
    
    def _setup_agents(self):
        """Create specialized agents."""
        self.sop_analyst = Agent(
            role="SOP Analyst",
            goal="Understand business context from SOPs",
            backstory="Expert at analyzing procedures and workflows",
            llm=self.llm,
            verbose=True
        )
        
        self.query_engineer = Agent(
            role="Query Engineer",
            goal="Generate optimized platform-specific queries",
            backstory="Expert in KQL, SPL, and SQL with deep platform knowledge",
            llm=self.llm,
            verbose=True
        )
        
        self.validator = Agent(
            role="Query Validator",
            goal="Validate and optimize generated queries",
            backstory="Ensures queries are correct, efficient, and safe",
            llm=self.llm,
            verbose=True
        )
    
    async def orchestrate_query_generation(self, request: QueryRequest) -> QueryResponse:
        """Orchestrate multi-agent query generation."""
        # Define tasks
        analyze_task = Task(
            description=f"Analyze the request: {request.natural_language}",
            agent=self.sop_analyst,
            expected_output="Business context and relevant procedures"
        )
        
        generate_task = Task(
            description=f"Generate {request.platform} query based on analysis",
            agent=self.query_engineer,
            expected_output=f"Valid {request.platform} query"
        )
        
        validate_task = Task(
            description="Validate and optimize the generated query",
            agent=self.validator,
            expected_output="Validated query with optimization notes"
        )
        
        # Create and execute crew
        crew = Crew(
            agents=[self.sop_analyst, self.query_engineer, self.validator],
            tasks=[analyze_task, generate_task, validate_task],
            verbose=True
        )
        
        result = crew.kickoff()
        return QueryResponse(query=result, platform=request.platform)
```

---

## GraphRAG Integration

### Use GraphRAG for Schema Understanding
```python
from graphrag import GraphRAG
from typing import Dict, List

class SchemaAnalyzer:
    """Analyze data schemas using GraphRAG."""
    
    def __init__(self, chromadb_client):
        self.graphrag = GraphRAG()
        self.db = chromadb_client
    
    async def extract_schema(self, platform: str) -> Dict:
        """Extract and understand data schema for platform."""
        # Build knowledge graph from schema documents
        schema_docs = await self._fetch_schema_docs(platform)
        
        graph = await self.graphrag.build_graph(
            documents=schema_docs,
            entity_types=["table", "column", "relationship", "index"]
        )
        
        # Query graph for relationships
        schema_context = await self.graphrag.query(
            graph=graph,
            query="Describe all table relationships and key columns"
        )
        
        return {
            "entities": graph.entities,
            "relationships": graph.relationships,
            "context": schema_context
        }
    
    async def enhance_query_with_schema(
        self, 
        query_request: QueryRequest,
        schema: Dict
    ) -> QueryRequest:
        """Add schema hints to query request."""
        query_request.schema_hints = {
            "tables": schema["entities"],
            "joins": schema["relationships"],
            "context": schema["context"]
        }
        return query_request
```

---

## ChromaDB Patterns

### Vector Store Setup
```python
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

class VectorStore:
    """Manage ChromaDB for SOP storage."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Use Ollama for embeddings
        self.embedding_fn = embedding_functions.OllamaEmbeddingFunction(
            model_name="nomic-embed-text",
            url="http://localhost:11434/api/embeddings"
        )
        
        self.collection = self.client.get_or_create_collection(
            name="sop_procedures",
            embedding_function=self.embedding_fn,
            metadata={"description": "SOP procedures and workflows"}
        )
    
    async def add_documents(self, documents: List[SOPDocument]):
        """Add documents to vector store."""
        self.collection.add(
            documents=[doc.content for doc in documents],
            metadatas=[doc.metadata for doc in documents],
            ids=[doc.id for doc in documents]
        )
    
    async def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Semantic search in vector store."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        return results
```

---

## Ollama Integration

### Use Ollama for LLM Operations
```python
import ollama
from typing import AsyncGenerator

class OllamaClient:
    """Client for interacting with local Ollama instance."""
    
    def __init__(self, model: str = "llama3", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        ollama.set_host(host)
    
    async def generate(
        self, 
        prompt: str, 
        system: Optional[str] = None,
        stream: bool = False
    ) -> str:
        """Generate completion from Ollama."""
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            system=system,
            stream=stream
        )
        
        if stream:
            return self._stream_response(response)
        return response['response']
    
    async def chat(self, messages: List[Dict[str, str]]) -> str:
        """Chat completion with conversation history."""
        response = ollama.chat(
            model=self.model,
            messages=messages
        )
        return response['message']['content']
    
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings for text."""
        response = ollama.embeddings(
            model="nomic-embed-text",
            prompt=text
        )
        return response['embedding']
```

---

## Error Handling Patterns

### Standardized Exception Hierarchy
```python
class LexiQueryException(Exception):
    """Base exception for all LexiQuery errors."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class ServiceUnavailableException(LexiQueryException):
    """Raised when a required service is unavailable."""
    pass

class QueryGenerationException(LexiQueryException):
    """Raised when query generation fails."""
    pass

class AuthorizationException(LexiQueryException):
    """Raised when authentication/authorization fails."""
    pass

class VectorStoreException(LexiQueryException):
    """Raised when vector database operations fail."""
    pass
```

### Retry Logic with Tenacity
```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class ResilientService:
    """Service with built-in retry logic."""
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(ServiceUnavailableException)
    )
    async def call_external_service(self, endpoint: str) -> Dict:
        """Call external service with retries."""
        try:
            response = await self.client.get(endpoint)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise ServiceUnavailableException(
                f"Service call failed: {endpoint}",
                details={"error": str(e)}
            )
```

---

## Testing Guidelines

### Unit Test Pattern
```python
import pytest
from unittest.mock import AsyncMock, patch

from shared.models.query_models import QueryRequest, QueryResponse
from services.query_generator.src.generator import QueryGenerator

@pytest.fixture
async def generator():
    """Create generator instance for testing."""
    gen = QueryGenerator(model="llama3")
    await gen.initialize()
    yield gen
    await gen.cleanup()

@pytest.mark.asyncio
async def test_generate_kql_query(generator):
    """Test KQL query generation."""
    request = QueryRequest(
        natural_language="Show all failed login attempts in the last hour",
        platform="kql",
        context={"table": "SigninLogs"}
    )
    
    response = await generator.generate(request)
    
    assert response.platform == "kql"
    assert "SigninLogs" in response.query
    assert "where" in response.query.lower()
    assert response.confidence > 0.5

@pytest.mark.asyncio
async def test_generate_query_with_schema_hints(generator):
    """Test query generation with GraphRAG schema hints."""
    request = QueryRequest(
        natural_language="Count errors by service",
        platform="sql",
        schema_hints={
            "tables": ["logs", "services"],
            "joins": ["logs.service_id = services.id"]
        }
    )
    
    response = await generator.generate(request)
    
    assert "JOIN" in response.query
    assert "GROUP BY" in response.query
```

### Integration Test Pattern
```python
import pytest
from httpx import AsyncClient

from services.core_engine.src.main import app

@pytest.mark.asyncio
async def test_end_to_end_query_workflow():
    """Test complete query workflow with all services."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Step 1: Submit query request
        response = await client.post(
            "/api/v1/query",
            json={
                "natural_language": "Show database connection errors",
                "platform": "kql"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Step 2: Verify query was generated
        assert "query" in data
        assert data["platform"] == "kql"
        
        # Step 3: Check if cached
        cache_response = await client.get(f"/api/v1/cache/{data['query_id']}")
        assert cache_response.status_code == 200
```

---

## Configuration Management

### Environment-Based Configuration
```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """Service configuration from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False
    )
    
    # Service
    service_name: str = "core-engine"
    service_port: int = 8000
    log_level: str = "INFO"
    
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    ollama_embedding_model: str = "nomic-embed-text"
    
    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    
    # Service URLs
    sop_engine_url: str = "http://sop-engine:8001"
    query_generator_url: str = "http://query-generator:8002"
    cache_service_url: str = "http://cache-learning:8003"
    
    # Auth
    auth_provider: str = "abstract"  # google, gcp, azure
    oauth_client_id: Optional[str] = None
    oauth_client_secret: Optional[str] = None

# Global settings instance
settings = Settings()
```

---

## Docker Best Practices

### Multi-Stage Dockerfile
```dockerfile
# Builder stage
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/ ./src/
COPY shared/ ./shared/

# Add local bin to PATH
ENV PATH=/root/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Common Patterns to AVOID

### ❌ Don't Use External LLM APIs
```python
# BAD - Uses external API
import openai
client = openai.OpenAI(api_key="...")

# GOOD - Uses local Ollama
import ollama
response = ollama.generate(model="llama3", prompt="...")
```

### ❌ Don't Hardcode Configuration
```python
# BAD
DATABASE_URL = "postgresql://localhost:5432/db"

# GOOD
from .config import settings
DATABASE_URL = settings.database_url
```

### ❌ Don't Skip Type Hints
```python
# BAD
async def process_query(request):
    return generate(request)

# GOOD
async def process_query(request: QueryRequest) -> QueryResponse:
    return await generate(request)
```

### ❌ Don't Ignore Error Handling
```python
# BAD
result = await external_service.call()

# GOOD
try:
    result = await external_service.call()
except ServiceUnavailableException as e:
    logger.error(f"Service unavailable: {e.details}")
    raise HTTPException(status_code=503, detail=str(e))
```

---

## Logging Standards

### Structured Logging with structlog
```python
import structlog
from structlog.stdlib import LoggerFactory

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    logger_factory=LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Usage
logger.info(
    "query_generated",
    platform="kql",
    query_length=150,
    confidence=0.87,
    correlation_id=request_id
)
```

---

## Performance Considerations

### Use Async I/O Everywhere
```python
# All I/O operations should be async
async def fetch_data():
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

# Use asyncio.gather for parallel operations
results = await asyncio.gather(
    fetch_sop_context(query),
    fetch_schema_info(platform),
    check_cache(query_hash)
)
```

### Cache Expensive Operations
```python
from functools import lru_cache
import hashlib

class QueryCache:
    """Cache query results to improve performance."""
    
    def _cache_key(self, request: QueryRequest) -> str:
        """Generate cache key from request."""
        content = f"{request.natural_language}:{request.platform}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def get_or_generate(self, request: QueryRequest) -> QueryResponse:
        """Get from cache or generate new query."""
        cache_key = self._cache_key(request)
        
        # Try cache first
        cached = await self.redis.get(cache_key)
        if cached:
            logger.info("cache_hit", key=cache_key)
            return QueryResponse.parse_raw(cached)
        
        # Generate and cache
        response = await self.generator.generate(request)
        await self.redis.setex(
            cache_key,
            3600,  # 1 hour TTL
            response.json()
        )
        
        return response
```

---

## LLM Cost Tracing Mechanism

### Overview
Implement comprehensive cost tracking for all LLM and embedding operations throughout a user journey. Track token usage, compute time, and associated costs for local Ollama models.

### Cost Tracking Data Model
```python
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime
from uuid import UUID, uuid4

class ModelUsage(BaseModel):
    """Track individual model invocation usage."""
    
    usage_id: UUID = Field(default_factory=uuid4)
    journey_id: UUID = Field(..., description="User journey/session ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Model information
    model_type: Literal["llm", "embedding"] = Field(...)
    model_name: str = Field(..., description="e.g., llama3, nomic-embed-text")
    
    # Usage metrics
    prompt_tokens: int = Field(0)
    completion_tokens: int = Field(0)
    total_tokens: int = Field(0)
    
    # Compute metrics
    latency_ms: float = Field(..., description="Response time in milliseconds")
    compute_units: float = Field(..., description="Normalized compute cost")
    
    # Cost calculation
    estimated_cost_usd: float = Field(..., description="Estimated cost in USD")
    
    # Context
    service_name: str = Field(..., description="Service that made the call")
    operation: str = Field(..., description="e.g., query_generation, semantic_search")
    metadata: Optional[dict] = Field(None)


class JourneyCostSummary(BaseModel):
    """Aggregate cost summary for a complete journey."""
    
    journey_id: UUID
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Aggregate metrics
    total_llm_calls: int = 0
    total_embedding_calls: int = 0
    total_tokens: int = 0
    total_compute_units: float = 0.0
    total_cost_usd: float = 0.0
    
    # Per-service breakdown
    cost_by_service: dict[str, float] = Field(default_factory=dict)
    
    # Per-model breakdown
    cost_by_model: dict[str, float] = Field(default_factory=dict)
    
    # Detailed usage records
    usage_records: List[ModelUsage] = Field(default_factory=list)
```

### Cost Tracker Service Interface
```python
from abc import ABC, abstractmethod
from typing import List
from uuid import UUID

class CostTrackerInterface(ABC):
    """Interface for tracking LLM and embedding costs."""
    
    @abstractmethod
    async def track_usage(
        self, 
        usage: ModelUsage
    ) -> None:
        """Record a model usage event.
        
        Args:
            usage: Model usage metrics
        """
        pass
    
    @abstractmethod
    async def get_journey_cost(
        self, 
        journey_id: UUID
    ) -> JourneyCostSummary:
        """Get cost summary for a journey.
        
        Args:
            journey_id: Journey identifier
            
        Returns:
            Aggregate cost summary
        """
        pass
    
    @abstractmethod
    async def get_recent_journeys(
        self,
        limit: int = 10
    ) -> List[JourneyCostSummary]:
        """Get recent journey cost summaries.
        
        Args:
            limit: Maximum number of journeys to return
            
        Returns:
            List of journey summaries
        """
        pass
```

### Ollama Client with Cost Tracking
```python
import ollama
import tiktoken
import time
from typing import Optional
from uuid import UUID
from contextlib import asynccontextmanager

class TrackedOllamaClient:
    """Ollama client with built-in cost tracking."""
    
    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://localhost:11434",
        cost_tracker: CostTrackerInterface = None,
        service_name: str = "unknown"
    ):
        self.model = model
        self.host = host
        self.cost_tracker = cost_tracker
        self.service_name = service_name
        ollama.set_host(host)
        
        # Token encoder for cost estimation
        try:
            self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Approximation
        except:
            self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoder.encode(text))
    
    def _calculate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float
    ) -> float:
        """Calculate estimated cost for local model.
        
        For local models, cost is based on:
        - Compute time (latency)
        - Token count (as proxy for compute intensity)
        
        Adjust rates based on your infrastructure costs.
        """
        # Example rates (adjust based on your infrastructure)
        COST_PER_1K_TOKENS = 0.001  # $0.001 per 1K tokens
        COST_PER_SECOND = 0.01      # $0.01 per second of compute
        
        token_cost = (prompt_tokens + completion_tokens) / 1000 * COST_PER_1K_TOKENS
        compute_cost = (latency_ms / 1000) * COST_PER_SECOND
        
        return token_cost + compute_cost
    
    async def generate(
        self,
        prompt: str,
        journey_id: UUID,
        operation: str = "generate",
        system: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> str:
        """Generate completion with cost tracking."""
        # Count prompt tokens
        prompt_tokens = self._count_tokens(prompt)
        if system:
            prompt_tokens += self._count_tokens(system)
        
        # Track start time
        start_time = time.time()
        
        # Generate response
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            system=system,
            stream=stream,
            **kwargs
        )
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract response text
        response_text = response['response'] if not stream else ""
        
        # Count completion tokens
        completion_tokens = self._count_tokens(response_text)
        total_tokens = prompt_tokens + completion_tokens
        
        # Calculate cost
        estimated_cost = self._calculate_cost(
            prompt_tokens,
            completion_tokens,
            latency_ms
        )
        
        # Track usage
        if self.cost_tracker:
            usage = ModelUsage(
                journey_id=journey_id,
                model_type="llm",
                model_name=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                compute_units=latency_ms / 1000,  # seconds as compute units
                estimated_cost_usd=estimated_cost,
                service_name=self.service_name,
                operation=operation
            )
            await self.cost_tracker.track_usage(usage)
        
        return response_text
    
    async def embed(
        self,
        text: str,
        journey_id: UUID,
        operation: str = "embed",
        model: str = "nomic-embed-text"
    ) -> List[float]:
        """Generate embeddings with cost tracking."""
        # Count tokens
        tokens = self._count_tokens(text)
        
        # Track start time
        start_time = time.time()
        
        # Generate embedding
        response = ollama.embeddings(
            model=model,
            prompt=text
        )
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Calculate cost (embeddings typically cheaper than generation)
        EMBEDDING_COST_PER_1K_TOKENS = 0.0001
        estimated_cost = (tokens / 1000) * EMBEDDING_COST_PER_1K_TOKENS
        
        # Track usage
        if self.cost_tracker:
            usage = ModelUsage(
                journey_id=journey_id,
                model_type="embedding",
                model_name=model,
                prompt_tokens=tokens,
                completion_tokens=0,
                total_tokens=tokens,
                latency_ms=latency_ms,
                compute_units=latency_ms / 1000,
                estimated_cost_usd=estimated_cost,
                service_name=self.service_name,
                operation=operation
            )
            await self.cost_tracker.track_usage(usage)
        
        return response['embedding']
```

### Cost Tracker Implementation
```python
import asyncio
from typing import Dict, List
from uuid import UUID
from collections import defaultdict
import structlog

logger = structlog.get_logger()

class RedisCostTracker(CostTrackerInterface):
    """Redis-backed cost tracker with in-memory aggregation."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self._journey_cache: Dict[UUID, JourneyCostSummary] = {}
    
    async def track_usage(self, usage: ModelUsage) -> None:
        """Record usage and update journey aggregates."""
        # Store individual usage record
        usage_key = f"cost:usage:{usage.usage_id}"
        await self.redis.setex(
            usage_key,
            86400 * 30,  # 30 days retention
            usage.json()
        )
        
        # Add to journey's usage list
        journey_key = f"cost:journey:{usage.journey_id}:usages"
        await self.redis.lpush(journey_key, str(usage.usage_id))
        await self.redis.expire(journey_key, 86400 * 30)
        
        # Update journey summary
        await self._update_journey_summary(usage)
        
        # Log for monitoring
        logger.info(
            "cost_tracked",
            journey_id=str(usage.journey_id),
            model_type=usage.model_type,
            model_name=usage.model_name,
            tokens=usage.total_tokens,
            cost_usd=usage.estimated_cost_usd,
            service=usage.service_name
        )
    
    async def _update_journey_summary(self, usage: ModelUsage) -> None:
        """Update aggregate journey summary."""
        summary_key = f"cost:journey:{usage.journey_id}:summary"
        
        # Get existing summary
        existing = await self.redis.get(summary_key)
        if existing:
            summary = JourneyCostSummary.parse_raw(existing)
        else:
            summary = JourneyCostSummary(
                journey_id=usage.journey_id,
                start_time=usage.timestamp
            )
        
        # Update aggregates
        if usage.model_type == "llm":
            summary.total_llm_calls += 1
        else:
            summary.total_embedding_calls += 1
        
        summary.total_tokens += usage.total_tokens
        summary.total_compute_units += usage.compute_units
        summary.total_cost_usd += usage.estimated_cost_usd
        summary.end_time = usage.timestamp
        
        # Update breakdowns
        summary.cost_by_service[usage.service_name] = \
            summary.cost_by_service.get(usage.service_name, 0.0) + usage.estimated_cost_usd
        
        summary.cost_by_model[usage.model_name] = \
            summary.cost_by_model.get(usage.model_name, 0.0) + usage.estimated_cost_usd
        
        # Store updated summary
        await self.redis.setex(
            summary_key,
            86400 * 30,
            summary.json()
        )
    
    async def get_journey_cost(self, journey_id: UUID) -> JourneyCostSummary:
        """Get complete cost summary for journey."""
        summary_key = f"cost:journey:{journey_id}:summary"
        summary_data = await self.redis.get(summary_key)
        
        if not summary_data:
            raise ValueError(f"Journey {journey_id} not found")
        
        summary = JourneyCostSummary.parse_raw(summary_data)
        
        # Optionally load detailed usage records
        journey_key = f"cost:journey:{journey_id}:usages"
        usage_ids = await self.redis.lrange(journey_key, 0, -1)
        
        for usage_id in usage_ids[:100]:  # Limit to last 100
            usage_key = f"cost:usage:{usage_id}"
            usage_data = await self.redis.get(usage_key)
            if usage_data:
                summary.usage_records.append(ModelUsage.parse_raw(usage_data))
        
        return summary
    
    async def get_recent_journeys(self, limit: int = 10) -> List[JourneyCostSummary]:
        """Get recent journey summaries."""
        # This requires maintaining a sorted set of journey IDs by timestamp
        # Implementation depends on your indexing strategy
        keys = await self.redis.keys("cost:journey:*:summary")
        summaries = []
        
        for key in keys[:limit]:
            data = await self.redis.get(key)
            if data:
                summaries.append(JourneyCostSummary.parse_raw(data))
        
        # Sort by start time
        summaries.sort(key=lambda s: s.start_time, reverse=True)
        return summaries[:limit]
```

### Integration with FastAPI Service
```python
from fastapi import FastAPI, Depends, Request
from uuid import UUID, uuid4
from contextvars import ContextVar

# Context variable for journey tracking
journey_id_ctx: ContextVar[UUID] = ContextVar('journey_id', default=None)

app = FastAPI()

@app.middleware("http")
async def journey_tracking_middleware(request: Request, call_next):
    """Middleware to track journey ID across requests."""
    # Get or create journey ID from header or session
    journey_id = request.headers.get("X-Journey-ID")
    if not journey_id:
        journey_id = str(uuid4())
    
    journey_id_ctx.set(UUID(journey_id))
    
    response = await call_next(request)
    response.headers["X-Journey-ID"] = journey_id
    
    return response

def get_journey_id() -> UUID:
    """Dependency to get current journey ID."""
    return journey_id_ctx.get()

@app.get("/api/v1/cost/journey/{journey_id}")
async def get_journey_cost(
    journey_id: UUID,
    cost_tracker: CostTrackerInterface = Depends(lambda: app.state.cost_tracker)
):
    """Get cost summary for a journey."""
    summary = await cost_tracker.get_journey_cost(journey_id)
    return summary

@app.get("/api/v1/cost/recent")
async def get_recent_costs(
    limit: int = 10,
    cost_tracker: CostTrackerInterface = Depends(lambda: app.state.cost_tracker)
):
    """Get recent journey costs."""
    summaries = await cost_tracker.get_recent_journeys(limit)
    return summaries
```

### CrewAI Integration with Cost Tracking
```python
from crewai import Agent, Task, Crew
from uuid import UUID

class CostTrackedOrchestrator:
    """Orchestrator with cost tracking for all agent operations."""
    
    def __init__(
        self,
        cost_tracker: CostTrackerInterface,
        service_name: str = "core-engine"
    ):
        self.cost_tracker = cost_tracker
        self.service_name = service_name
        
        # Create tracked Ollama client
        self.llm_client = TrackedOllamaClient(
            model="llama3",
            cost_tracker=cost_tracker,
            service_name=service_name
        )
    
    async def execute_with_tracking(
        self,
        request: QueryRequest,
        journey_id: UUID
    ) -> QueryResponse:
        """Execute query with comprehensive cost tracking."""
        # Create agents with tracked LLM
        agents = self._create_agents()
        tasks = self._create_tasks(request, agents)
        
        crew = Crew(
            agents=agents,
            tasks=tasks,
            verbose=True
        )
        
        # Execute (CrewAI will use our tracked LLM client)
        result = crew.kickoff()
        
        # Get final cost summary
        cost_summary = await self.cost_tracker.get_journey_cost(journey_id)
        
        logger.info(
            "journey_completed",
            journey_id=str(journey_id),
            total_cost=cost_summary.total_cost_usd,
            total_tokens=cost_summary.total_tokens,
            llm_calls=cost_summary.total_llm_calls
        )
        
        return QueryResponse(
            query=result,
            platform=request.platform,
            journey_id=journey_id,
            cost_summary=cost_summary
        )
```

### Cost Tracking Best Practices

1. **Always Track at Operation Level**: Track every LLM and embedding call
2. **Use Journey IDs**: Maintain journey context across all service calls
3. **Include Metadata**: Add operation context for better analysis
4. **Set Retention Policies**: Define data retention based on compliance needs
5. **Monitor Costs**: Set up alerts for unusual cost patterns
6. **Optimize Based on Data**: Use cost data to identify optimization opportunities
7. **Report Aggregates**: Provide cost summaries at journey, service, and model levels
8. **Export for Analytics**: Allow cost data export for detailed analysis

### Cost Optimization Patterns

```python
class CostOptimizer:
    """Analyze and optimize LLM costs."""
    
    async def analyze_journey_costs(
        self,
        journey_id: UUID,
        cost_tracker: CostTrackerInterface
    ) -> dict:
        """Analyze cost breakdown and suggest optimizations."""
        summary = await cost_tracker.get_journey_cost(journey_id)
        
        recommendations = []
        
        # Check for excessive calls
        if summary.total_llm_calls > 10:
            recommendations.append(
                "Consider caching results to reduce LLM calls"
            )
        
        # Check for large token usage
        avg_tokens_per_call = summary.total_tokens / max(summary.total_llm_calls, 1)
        if avg_tokens_per_call > 2000:
            recommendations.append(
                "Consider prompt optimization to reduce token usage"
            )
        
        # Check for slow operations
        slow_operations = [
            u for u in summary.usage_records 
            if u.latency_ms > 5000
        ]
        if slow_operations:
            recommendations.append(
                f"Found {len(slow_operations)} slow operations (>5s)"
            )
        
        return {
            "summary": summary,
            "recommendations": recommendations,
            "cost_breakdown": {
                "by_service": summary.cost_by_service,
                "by_model": summary.cost_by_model
            }
        }
```

---

## Summary

When generating code for LexiQuery:

1. ✅ **Always define interfaces first** (ABC with @abstractmethod)
2. ✅ **Use Pydantic models** for all data validation
3. ✅ **Implement mocks** before real logic
4. ✅ **Use Ollama** for all LLM operations (never external APIs)
5. ✅ **Leverage ChromaDB** for vector storage
6. ✅ **Apply GraphRAG** for schema understanding
7. ✅ **Orchestrate with CrewAI** agents
8. ✅ **Track LLM costs** for every operation and journey
9. ✅ **Type hint everything**
10. ✅ **Write async code** for I/O operations
11. ✅ **Include comprehensive error handling**
12. ✅ **Add structured logging** with cost metrics
13. ✅ **Write tests** alongside implementation
14. ✅ **Follow the microservices pattern**
15. ✅ **Keep services independent** and containerizable

**Privacy First**: All AI operations use local Ollama models - no external API calls, no data leakage.

**Cost Transparency**: Every LLM and embedding operation is tracked with journey-level cost aggregation for complete visibility and optimization.
