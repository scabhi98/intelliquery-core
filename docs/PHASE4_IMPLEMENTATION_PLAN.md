# Phase 4 Implementation Plan

**Version**: 1.0  
**Date**: February 5, 2026  
**Status**: Planning & Preparation  
**Phase 4 Focus**: Protocol Interface Service & SOP Engine (Real Implementations)

---

## Executive Summary

Phase 4 moves from mock implementations to **production-ready real modules**. This phase focuses on:

1. **Protocol Interface Service**: MCP server/client + A2A SDK integration for inter-service communication
2. **SOP Engine Service**: Document parsing, ChromaDB vector storage, and GraphRAG-powered semantic search

These foundational services enable the full AI-powered query intelligence pipeline by providing:
- Standardized inter-service communication (MCP + A2A)
- Semantic document retrieval (SOP procedures)
- Schema understanding (GraphRAG)
- Privacy-first embeddings (Ollama)

---

## Phase 4 Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                  Core Engine (Phase 2)                       │
│              (A2A Orchestrator - No Changes)                 │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
    ┌────▼──────────────┐   ┌───▼──────────────────┐
    │ Protocol Interface│   │  SOP Engine (Phase 4)│
    │ Service (Phase 4) │   │  (Real Implementation)
    │                  │   │                      │
    │  ┌────────────┐  │   │  ┌────────────────┐  │
    │  │MCP Server  │  │   │  │ Document Parser│  │
    │  ├────────────┤  │   │  ├────────────────┤  │
    │  │A2A Handler │  │   │  │ ChromaDB Client│  │
    │  ├────────────┤  │   │  ├────────────────┤  │
    │  │Auth Provider   │   │  │ Ollama Embedder│  │
    │  ├────────────┤  │   │  ├────────────────┤  │
    │  │MCP Tools   │  │   │  │ GraphRAG Schema│  │
    │  │Registry    │  │   │  │ Analyzer       │  │
    │  └────────────┘  │   │  └────────────────┘  │
    └────────────────────┘   └────────────────────┘
         │                            │
         │ (Tools)                    │ (Embeddings)
         │                            ▼
         │                    ┌──────────────────┐
         │                    │ Ollama Instance  │
         │                    │  - nomic-embed   │
         │                    │  - llama3        │
         │                    └──────────────────┘
         │
         │ (Data Flow)
         │
         ▼
    ┌──────────────────────┐
    │   ChromaDB Instance  │
    │   (Vector DB)        │
    └──────────────────────┘
```

---

## Phase 4 Breakdown

### Part A: Protocol Interface Service (Weeks 1-3)

#### A.1 MCP Server Implementation

**Purpose**: Standardized tool invocation across services

**Components**:

1. **MCP Server Framework**
   - Tool registration and discovery
   - Request/response validation
   - Error handling and logging
   - Health check endpoints

2. **Tool Definitions**
   - Query generator tools
   - SOP retrieval tools
   - Data connector tools
   - Cache management tools

3. **Tool Handlers**
   - Async tool execution
   - Context preservation
   - Cost tracking integration

**Dependencies**:
```
mcp>=0.9.0
pydantic>=2.0
fastapi>=0.100.0
httpx>=0.24.0
structlog>=23.1.0
```

**Key Files**:
```
services/protocol-interface/src/
├── mcp/
│   ├── __init__.py
│   ├── server.py                 # MCP server core
│   ├── tool_registry.py          # Tool registration
│   ├── tool_handlers.py          # Tool execution handlers
│   └── schemas.py                # Tool schemas
├── a2a/
│   ├── __init__.py
│   ├── handler.py                # A2A protocol handler
│   └── client.py                 # A2A service client
├── auth/
│   ├── __init__.py
│   ├── auth_provider.py          # Abstract auth interface
│   ├── oauth2_handler.py         # OAuth2 OBO flow
│   └── providers/
│       ├── google_provider.py
│       ├── azure_provider.py
│       └── gcp_provider.py
└── main.py                       # FastAPI entry point
```

#### A.2 A2A Protocol Handler

**Purpose**: Agent-to-Agent communication with A2A SDK

**Key Features**:
- Agent discovery via A2A SDK
- Task request/response handling
- Context propagation
- Error handling with retry logic

**A2A Integration Points**:
```python
# Core A2A Operations
- register_agent(descriptor: A2AAgentDescriptor) → bool
- discover_agents(filters: AgentFilter) → List[A2AAgentDescriptor]
- invoke_task(request: A2ATaskRequest) → A2ATaskResponse
- handle_incoming_task(request: A2ATaskRequest) → A2ATaskResponse
```

**Implementation Details**:
- Async/await for all I/O operations
- Automatic retry on transient failures
- Cost tracking for each agent invocation
- Structured logging with correlation IDs

#### A.3 Authentication Provider Interface

**Purpose**: Pluggable OAuth2 integration

**Supported Providers**:
1. **Google OAuth2**
   - OAuth2 authorization code flow
   - Service account integration
   - OBO (On-Behalf-Of) token exchange

2. **Azure AD**
   - OAuth2 authorization code flow
   - Service principal authentication
   - OBO flow for delegated access

3. **GCP Service Accounts**
   - Service account key authentication
   - OAuth2 scope management
   - Token refresh handling

**Key Interface**:
```python
class AuthProvider(ABC):
    @abstractmethod
    async def authenticate() -> str:
        """Get authenticated access token"""
        pass
    
    @abstractmethod
    async def exchange_token_obo(
        user_token: str,
        target_service: str
    ) -> str:
        """Exchange token for OBO flow"""
        pass
    
    @abstractmethod
    async def validate_token(token: str) -> TokenInfo:
        """Validate and get token info"""
        pass
```

**Files**:
```
services/protocol-interface/src/auth/
├── auth_provider.py              # Abstract interface
├── oauth2_handler.py             # OAuth2 core logic
└── providers/
    ├── google_provider.py        # Google OAuth2
    ├── azure_provider.py         # Azure AD
    └── gcp_provider.py           # GCP Service Account
```

---

### Part B: SOP Engine Service (Weeks 4-6)

#### B.1 Document Parser

**Purpose**: Extract procedures from various document formats

**Supported Formats**:
1. **Markdown** (.md)
   - Section-based parsing
   - Code block extraction
   - Link preservation

2. **Word Documents** (.docx)
   - Paragraph structure
   - Table extraction
   - Heading hierarchy

3. **PDF** (.pdf)
   - Text extraction
   - Layout preservation
   - Image/diagram detection

4. **Plain Text** (.txt)
   - Line-by-line parsing
   - Delimiter detection
   - Structure inference

**Key Classes**:

```python
class DocumentParser(ABC):
    @abstractmethod
    async def parse(
        document: SOPDocument
    ) -> List[ExtractedProcedure]:
        """Parse document into procedures"""
        pass

class ProcedureExtractor:
    """Extract procedures from parsed content"""
    - Identify procedure boundaries
    - Extract steps and substeps
    - Tag keywords and entities
    - Preserve cross-references

class MetadataExtractor:
    """Extract procedure metadata"""
    - Author, created date, last modified
    - Version information
    - Related procedures
    - Referenced tables/columns
```

**Implementation Approach**:
- Use `python-docx` for Word documents
- Use `PyPDF2` or `pdfplumber` for PDFs
- Use `markdown-it-py` for Markdown
- Custom regex for Plain text

**Key Files**:
```
services/sop-engine/src/parser/
├── __init__.py
├── document_parser.py            # Abstract interface
├── markdown_parser.py            # Markdown implementation
├── docx_parser.py                # Word document parser
├── pdf_parser.py                 # PDF parser
├── text_parser.py                # Plain text parser
├── procedure_extractor.py        # Procedure extraction logic
├── metadata_extractor.py         # Metadata extraction
└── models.py                     # Data models
```

**Dependencies**:
```
python-docx>=0.8.11
PyPDF2>=3.0.0
pdfplumber>=0.9.0
markdown-it-py>=3.0.0
langchain>=0.1.0  # For text splitting
tiktoken>=0.5.0   # For token counting
```

#### B.2 ChromaDB Vector Store Integration

**Purpose**: Store and retrieve embeddings for semantic search

**Key Components**:

1. **Vector Store Manager**
   ```python
   class VectorStoreManager:
       - initialize() → Initialize collection
       - add_documents() → Store embeddings
       - search() → Semantic search
       - delete_collection() → Cleanup
       - get_stats() → Collection stats
   ```

2. **Ollama Embedding Function**
   ```python
   class OllamaEmbedder:
       - embed_text(text: str) → List[float]
       - embed_batch(texts: List[str]) → List[List[float]]
       - get_embedding_dim() → int
   ```

3. **Collection Management**
   - Separate collections per SOP source
   - Metadata preservation
   - Version tracking
   - Deletion and cleanup

**ChromaDB Configuration**:
```python
# Persistent local storage
settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db",
    anonymized_telemetry=False,
    is_persistent=True
)

# Collections
- "sop_procedures": Main SOP procedure collection
- "sop_metadata": SOP metadata and references
- "error_patterns": Known error patterns
- "schema_info": Data schema information
```

**Key Files**:
```
services/sop-engine/src/vector_store/
├── __init__.py
├── vector_store.py               # Main vector store class
├── embedder.py                   # Ollama embedder
├── collection_manager.py         # Collection operations
└── models.py                     # Vector store models
```

**Dependencies**:
```
chromadb>=0.4.0
ollama>=0.1.0
```

#### B.3 GraphRAG Schema Analyzer

**Purpose**: Extract and understand data schemas for query generation

**Key Features**:

1. **Schema Extraction**
   - Identify data tables/sources
   - Extract column information
   - Detect relationships and joins
   - Identify key fields

2. **Relationship Mapping**
   - Table-to-table relationships
   - Common join keys
   - Dependency graphs
   - Access patterns

3. **Context Generation**
   - Query templates
   - Common filters
   - Aggregation patterns
   - Performance hints

**Implementation Approach**:

```python
class GraphRAGAnalyzer:
    """Use GraphRAG for schema understanding"""
    
    async def extract_schema(
        documents: List[Document]
    ) -> SchemaGraph:
        """Build knowledge graph from schema docs"""
        - Parse schema documentation
        - Create entity relationships
        - Extract constraints and rules
        - Build query patterns
    
    async def enrich_procedure(
        procedure: Procedure,
        schema: SchemaGraph
    ) -> EnrichedProcedure:
        """Add schema context to procedure"""
        - Identify relevant tables
        - Extract column references
        - Add schema hints
        - Link to examples
```

**Schema Models**:
```python
class SchemaEntity:
    - name: str (table/field name)
    - type: str (table, column, index, etc)
    - properties: dict
    - relationships: List[Relationship]

class Relationship:
    - source_entity: str
    - target_entity: str
    - relationship_type: str (join, reference, etc)
    - cardinality: str (1-to-1, 1-to-many, etc)
    - condition: str (join condition)

class SchemaGraph:
    - entities: Dict[str, SchemaEntity]
    - relationships: List[Relationship]
    - context: Dict[str, Any]
```

**Key Files**:
```
services/sop-engine/src/schema/
├── __init__.py
├── graph_analyzer.py             # GraphRAG analyzer
├── schema_extractor.py           # Schema extraction
├── relationship_mapper.py        # Relationship discovery
├── query_pattern_finder.py       # Query pattern extraction
└── models.py                     # Schema models
```

**Dependencies**:
```
graphrag>=0.1.0
networkx>=3.0  # For graph operations
```

#### B.4 Semantic Search Engine

**Purpose**: Find relevant procedures for user queries

**Search Capabilities**:

1. **Vector Search**
   - Embedding-based similarity
   - Top-K retrieval
   - Similarity scoring
   - Reranking support

2. **Hybrid Search**
   - Combine vector + keyword search
   - BM25 scoring
   - Weighted combination
   - Result aggregation

3. **Metadata Filtering**
   - Filter by source/domain
   - Filter by procedure type
   - Filter by keywords/tags
   - Filter by date range

**Implementation**:
```python
class SemanticSearchEngine:
    async def search(
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
        threshold: float = 0.5
    ) -> List[SearchResult]:
        """
        1. Embed query using Ollama
        2. Vector search in ChromaDB
        3. Apply metadata filters
        4. Score and rank results
        5. Return with confidence
        """
    
    async def hybrid_search(
        query: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        1. Vector search
        2. Keyword search (BM25)
        3. Rerank using language model
        4. Return combined results
        """
```

**Key Files**:
```
services/sop-engine/src/search/
├── __init__.py
├── search_engine.py              # Main search implementation
├── bm25_searcher.py              # Keyword search
├── vector_searcher.py            # Vector search
├── hybrid_searcher.py            # Hybrid search
└── reranker.py                   # Result reranking
```

**Dependencies**:
```
rank-bm25>=0.2.2
```

#### B.5 SOP Engine Service API

**Purpose**: REST API for SOP operations

**Endpoints**:
```
POST /api/v1/procedures/upload
  - Upload and parse SOP document
  - Extract procedures
  - Store embeddings

GET /api/v1/procedures/search
  - Semantic search for procedures
  - Query parameter: q (search query)
  - Optional: top_k, filters

GET /api/v1/procedures/{procedure_id}
  - Get specific procedure details
  - Include schema context
  - Include related procedures

GET /api/v1/schema
  - Get schema graph
  - Optional: source filter

POST /api/v1/schema/refresh
  - Refresh schema from documents
  - Rebuild relationships

GET /health
  - Health check
  - Dependencies status
```

**Key Files**:
```
services/sop-engine/src/
├── main.py                       # FastAPI app
├── config.py                     # Configuration
├── routes/
│   ├── procedures.py             # Procedure endpoints
│   ├── schema.py                 # Schema endpoints
│   └── health.py                 # Health check
└── dependencies.py               # FastAPI dependencies
```

---

## Implementation Timeline

### Week 1-2: Protocol Interface Service - Part 1 (MCP)

**Tasks**:
1. [ ] Set up Protocol Interface service structure
2. [ ] Implement MCP server framework
3. [ ] Create tool registry and registration system
4. [ ] Implement tool handler execution
5. [ ] Write unit tests for MCP components
6. [ ] Create Docker image

**Deliverables**:
- Working MCP server accepting tool calls
- Tool registry with sample tools
- 100% test coverage

**Key Milestones**:
- Day 3: MCP server responds to tool list requests
- Day 5: Tool execution working with cost tracking
- Day 7: Docker image building and running

---

### Week 3: Protocol Interface Service - Part 2 (A2A + Auth)

**Tasks**:
1. [ ] Implement A2A protocol handler
2. [ ] Integrate A2A SDK
3. [ ] Create abstract auth provider interface
4. [ ] Implement Google OAuth2 provider
5. [ ] Implement Azure AD provider
6. [ ] Implement OAuth2 OBO token exchange
7. [ ] Write integration tests
8. [ ] Update docker-compose.yml

**Deliverables**:
- Complete Protocol Interface Service (real implementation)
- OAuth2 authentication working
- A2A communication established

**Key Milestones**:
- Day 1: A2A handler integrated
- Day 3: OAuth2 login flow working
- Day 5: OBO token exchange functional
- Day 7: Integration tests passing

---

### Week 4-5: SOP Engine Service - Part 1 (Parser & ChromaDB)

**Tasks**:
1. [ ] Set up SOP Engine service structure
2. [ ] Implement document parser framework
3. [ ] Build Markdown parser
4. [ ] Build Word document parser
5. [ ] Build PDF parser
6. [ ] Build plain text parser
7. [ ] Implement procedure extractor
8. [ ] Implement metadata extractor
9. [ ] Integrate ChromaDB
10. [ ] Implement Ollama embedder
11. [ ] Write unit tests

**Deliverables**:
- Working document parser for all formats
- ChromaDB integration with embeddings
- Vector store operational

**Key Milestones**:
- Day 2: All parsers implemented
- Day 4: ChromaDB working with Ollama embeddings
- Day 7: Document upload and embedding storage working

---

### Week 6: SOP Engine Service - Part 2 (GraphRAG + Search)

**Tasks**:
1. [ ] Implement GraphRAG schema analyzer
2. [ ] Build schema extraction logic
3. [ ] Implement relationship mapping
4. [ ] Build semantic search engine
5. [ ] Implement hybrid search (vector + keyword)
6. [ ] Create SOP Engine API endpoints
7. [ ] Build Docker image
8. [ ] Write integration tests
9. [ ] Update docker-compose.yml

**Deliverables**:
- Complete SOP Engine Service (real implementation)
- Schema understanding via GraphRAG
- Semantic search operational
- API endpoints ready for core engine

**Key Milestones**:
- Day 1: GraphRAG schema extraction working
- Day 2: Semantic search functional
- Day 3: API endpoints serving requests
- Day 5: Docker image building and tests passing

---

## Technical Implementation Details

### Protocol Interface Service

#### MCP Server Implementation

```python
# services/protocol-interface/src/mcp/server.py

from fastapi import FastAPI
from typing import List, Dict, Any
import structlog

class MCPServer:
    """Model Context Protocol Server"""
    
    def __init__(self):
        self.app = FastAPI()
        self.tool_registry = ToolRegistry()
        self.setup_routes()
    
    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable
    ) -> None:
        """Register a new tool"""
        self.tool_registry.register(
            MCPTool(
                name=name,
                description=description,
                input_schema=input_schema,
                handler=handler
            )
        )
    
    @app.get("/mcp/tools/list")
    async def list_tools(self) -> Dict[str, List[MCPTool]]:
        """List all available tools"""
        return {
            "tools": self.tool_registry.list_tools(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @app.post("/mcp/tools/call")
    async def call_tool(
        self,
        request: MCPToolCallRequest
    ) -> MCPToolCallResponse:
        """Execute a tool"""
        tool = self.tool_registry.get_tool(request.tool_name)
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        
        try:
            result = await tool.execute(request.arguments)
            return MCPToolCallResponse(
                result=result,
                success=True
            )
        except Exception as e:
            return MCPToolCallResponse(
                result={"error": str(e)},
                success=False
            )
```

#### A2A Handler Implementation

```python
# services/protocol-interface/src/a2a/handler.py

from a2a_sdk import A2AClient, AgentDescriptor

class A2AHandler:
    """Handle A2A protocol operations"""
    
    def __init__(self, config: Config):
        self.client = A2AClient(
            agent_id=config.agent_id,
            agent_name=config.agent_name,
            endpoint=config.a2a_endpoint
        )
        self.cost_tracker = app.state.cost_tracker
    
    async def register_service(
        self,
        descriptor: A2AAgentDescriptor
    ) -> bool:
        """Register this service as A2A agent"""
        return await self.client.register(descriptor)
    
    async def discover_agents(
        self,
        agent_type: str
    ) -> List[A2AAgentDescriptor]:
        """Discover available agents"""
        return await self.client.discover(
            filters={"type": agent_type}
        )
    
    async def invoke_agent(
        self,
        agent_id: str,
        task: A2ATaskRequest
    ) -> A2ATaskResponse:
        """Invoke another agent's action"""
        start_time = time.time()
        
        try:
            response = await self.client.invoke_task(
                agent_id=agent_id,
                task=task
            )
            
            # Track cost
            latency_ms = (time.time() - start_time) * 1000
            await self.cost_tracker.track_usage(
                ModelUsage(
                    journey_id=task.journey_id,
                    model_type="a2a_call",
                    model_name=agent_id,
                    latency_ms=latency_ms,
                    service_name="protocol-interface",
                    operation="agent_invocation"
                )
            )
            
            return response
        except Exception as e:
            logger.error(
                "agent_invocation_failed",
                agent_id=agent_id,
                error=str(e)
            )
            raise
```

#### Auth Provider Implementation

```python
# services/protocol-interface/src/auth/providers/google_provider.py

from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials

class GoogleAuthProvider(AuthProvider):
    """Google OAuth2 authentication"""
    
    def __init__(self, config: GoogleConfig):
        self.config = config
        self.credentials = None
    
    async def authenticate(self) -> str:
        """Get authenticated token"""
        if not self.credentials or self.credentials.expired:
            credentials = Credentials.from_service_account_file(
                self.config.service_account_key_path,
                scopes=self.config.scopes
            )
            self.credentials = credentials
        
        return self.credentials.token
    
    async def exchange_token_obo(
        self,
        user_token: str,
        target_service: str
    ) -> str:
        """Exchange token for OBO flow"""
        # Implement JWT assertion flow
        jwt_grant_assertion = self._create_jwt_assertion(
            user_token,
            target_service
        )
        
        response = await self._exchange_assertion(jwt_grant_assertion)
        return response['access_token']
```

---

### SOP Engine Service

#### Document Parser Implementation

```python
# services/sop-engine/src/parser/document_parser.py

from abc import ABC, abstractmethod
from typing import List

class DocumentParser(ABC):
    """Base class for document parsers"""
    
    @abstractmethod
    async def parse(
        self,
        document: SOPDocument
    ) -> List[ExtractedProcedure]:
        """Parse document into procedures"""
        pass

# services/sop-engine/src/parser/markdown_parser.py

class MarkdownParser(DocumentParser):
    """Parse Markdown documents"""
    
    async def parse(
        self,
        document: SOPDocument
    ) -> List[ExtractedProcedure]:
        """Parse markdown into procedures"""
        md = markdown.Markdown()
        tree = md.parser.parseDocument(
            document.content.split('\n')
        ).getroot()
        
        procedures = []
        current_procedure = None
        
        for element in tree:
            if element.tag == 'h2':  # Procedure heading
                if current_procedure:
                    procedures.append(current_procedure)
                
                current_procedure = ExtractedProcedure(
                    title=element.text,
                    source=document.id
                )
            elif element.tag == 'li' and current_procedure:
                # Add step
                current_procedure.steps.append(element.text)
            elif element.tag == 'code' and current_procedure:
                # Add code block
                current_procedure.code_blocks.append(element.text)
        
        if current_procedure:
            procedures.append(current_procedure)
        
        return procedures
```

#### Vector Store Implementation

```python
# services/sop-engine/src/vector_store/vector_store.py

import chromadb
from chromadb.config import Settings

class VectorStoreManager:
    """Manage ChromaDB collections and embeddings"""
    
    def __init__(self, config: Config):
        self.client = chromadb.PersistentClient(
            path=config.chroma_db_path,
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        self.embedder = OllamaEmbedder(
            model=config.embedding_model,
            host=config.ollama_host
        )
    
    async def add_procedure(
        self,
        procedure: ExtractedProcedure,
        journey_id: UUID
    ) -> str:
        """Add procedure with embeddings"""
        # Create collection if needed
        collection = self.client.get_or_create_collection(
            name="sop_procedures",
            embedding_function=self.embedder
        )
        
        # Embed procedure content
        text = f"{procedure.title} {' '.join(procedure.steps)}"
        
        # Add to collection
        collection.add(
            ids=[procedure.id],
            documents=[text],
            metadatas=[{
                "title": procedure.title,
                "source": procedure.source,
                "created": datetime.utcnow().isoformat()
            }]
        )
        
        # Track cost
        await self.cost_tracker.track_usage(
            ModelUsage(
                journey_id=journey_id,
                model_type="embedding",
                model_name=config.embedding_model,
                prompt_tokens=len(text.split()),
                latency_ms=latency,
                estimated_cost_usd=cost
            )
        )
        
        return procedure.id
    
    async def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """Search for procedures"""
        collection = self.client.get_collection("sop_procedures")
        
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        return [
            SearchResult(
                procedure_id=results['ids'][0][i],
                title=results['metadatas'][0][i]['title'],
                content=results['documents'][0][i],
                similarity_score=1 - results['distances'][0][i]
            )
            for i in range(len(results['ids'][0]))
        ]
```

#### GraphRAG Schema Analyzer Implementation

```python
# services/sop-engine/src/schema/graph_analyzer.py

from graphrag import GraphRAG
import networkx as nx

class GraphRAGAnalyzer:
    """Analyze schemas using GraphRAG"""
    
    def __init__(self, config: Config):
        self.graphrag = GraphRAG()
        self.graph = nx.DiGraph()
    
    async def extract_schema(
        self,
        documents: List[Document]
    ) -> SchemaGraph:
        """Extract schema from documentation"""
        # Build knowledge graph
        graph = await self.graphrag.build_graph(
            documents=[doc.content for doc in documents],
            entity_types=[
                "table", "column", "join", "index",
                "relationship", "constraint"
            ]
        )
        
        # Extract entities
        entities = {}
        for node, attrs in graph.nodes(data=True):
            if attrs.get('type') == 'table':
                entities[node] = SchemaEntity(
                    name=node,
                    type="table",
                    properties=attrs
                )
        
        # Extract relationships
        relationships = []
        for source, target, attrs in graph.edges(data=True):
            relationships.append(
                Relationship(
                    source_entity=source,
                    target_entity=target,
                    relationship_type=attrs.get('type'),
                    condition=attrs.get('condition')
                )
            )
        
        return SchemaGraph(
            entities=entities,
            relationships=relationships
        )
```

---

## Integration Points

### Core Engine Integration

```python
# In core_engine/src/orchestrator.py

async def _invoke_sop_engine(
    self,
    query: str,
    journey_id: UUID
) -> Optional[List[Procedure]]:
    """Call SOP engine for procedure retrieval"""
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{settings.sop_engine_url}/api/v1/procedures/search",
            params={"q": query, "top_k": 5},
            headers={"X-Journey-ID": str(journey_id)}
        )
    
    if response.status_code == 200:
        procedures = response.json()
        logger.info(
            "sop_retrieval_success",
            count=len(procedures),
            journey_id=str(journey_id)
        )
        return procedures
    else:
        logger.error("sop_retrieval_failed")
        return None

# MCP tool invocation from core engine
async def _invoke_mcp_tool(
    self,
    tool_name: str,
    arguments: Dict[str, Any],
    journey_id: UUID
) -> Any:
    """Invoke MCP tool through protocol interface"""
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.protocol_interface_url}/mcp/tools/call",
            json={
                "tool_name": tool_name,
                "arguments": arguments
            },
            headers={"X-Journey-ID": str(journey_id)}
        )
    
    return response.json()["result"]
```

---

## Data Models

### Protocol Interface Models

```python
# shared/models/protocol_models.py (Updates)

class MCPTool(BaseModel):
    """MCP Tool Definition"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]] = None
    tags: List[str] = []

class MCPToolCallRequest(BaseModel):
    """MCP Tool Call Request"""
    tool_name: str
    arguments: Dict[str, Any]
    correlation_id: Optional[str] = None

class MCPToolCallResponse(BaseModel):
    """MCP Tool Call Response"""
    result: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None

class A2AAgentDescriptor(BaseModel):
    """A2A Agent Registration"""
    agent_id: str
    agent_name: str
    agent_type: str  # planner, knowledge, data
    capabilities: List[str]
    endpoint: str
    version: str = "1.0"

class AuthToken(BaseModel):
    """Authentication Token"""
    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    refresh_token: Optional[str] = None
```

### SOP Engine Models

```python
# shared/models/sop_models.py (Updates)

class ExtractedProcedure(BaseModel):
    """Extracted procedure from document"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    source: str  # Document source ID
    steps: List[str]
    code_blocks: List[str] = []
    tables_referenced: List[str] = []
    columns_referenced: List[str] = []
    keywords: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)

class SearchResult(BaseModel):
    """Search result from SOP engine"""
    procedure_id: str
    title: str
    content: str
    similarity_score: float
    metadata: Dict[str, Any] = {}

class SchemaEntity(BaseModel):
    """Schema entity (table, column, etc)"""
    name: str
    type: str  # table, column, index, etc
    properties: Dict[str, Any] = {}
    relationships: List['Relationship'] = []

class Relationship(BaseModel):
    """Schema relationship"""
    source_entity: str
    target_entity: str
    relationship_type: str  # join, reference, etc
    cardinality: str = "1-to-many"
    condition: Optional[str] = None

class SchemaGraph(BaseModel):
    """Complete schema graph"""
    entities: Dict[str, SchemaEntity]
    relationships: List[Relationship]
    context: Dict[str, Any] = {}
```

---

## Docker Configuration Updates

### Protocol Interface Dockerfile

```dockerfile
# services/protocol-interface/Dockerfile

FROM python:3.11-slim as builder

WORKDIR /build
RUN apt-get update && apt-get install -y gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY src/ ./src/
COPY shared/ ./shared/

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8001/health')" || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

### SOP Engine Dockerfile

```dockerfile
# services/sop-engine/Dockerfile

FROM python:3.11-slim as builder

WORKDIR /build
RUN apt-get update && apt-get install -y \
    gcc g++ \
    libpoppler-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y libpoppler-cpp0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
COPY src/ ./src/
COPY shared/ ./shared/

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Persistent volume for ChromaDB
VOLUME ["/app/chroma_db"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8002/health')" || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8002"]
```

### Docker Compose Updates

```yaml
# docker-compose.yml (Updates)

services:
  # ... existing services ...

  protocol-interface:
    build: ./services/protocol-interface
    container_name: protocol-interface
    ports:
      - "8001:8001"
    environment:
      FASTAPI_ENV: development
      LOG_LEVEL: INFO
      AUTH_PROVIDER: ${AUTH_PROVIDER:-google}
      GOOGLE_CLIENT_ID: ${GOOGLE_CLIENT_ID}
      AZURE_CLIENT_ID: ${AZURE_CLIENT_ID}
    depends_on:
      core-engine:
        condition: service_healthy
    networks:
      - lexiquery
    restart: on-failure

  sop-engine:
    build: ./services/sop-engine
    container_name: sop-engine
    ports:
      - "8002:8002"
    environment:
      FASTAPI_ENV: development
      LOG_LEVEL: INFO
      OLLAMA_HOST: http://ollama:11434
      CHROMA_DB_PATH: /app/chroma_db
      GRAPHRAG_MODEL: llama3
    depends_on:
      core-engine:
        condition: service_healthy
      ollama:
        condition: service_healthy
    volumes:
      - sop_engine_chroma:/app/chroma_db
    networks:
      - lexiquery
    restart: on-failure

  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"
    environment:
      OLLAMA_HOST: 0.0.0.0:11434
    volumes:
      - ollama_data:/root/.ollama
    networks:
      - lexiquery
    restart: on-failure

volumes:
  sop_engine_chroma:
  ollama_data:

networks:
  lexiquery:
    driver: bridge
```

---

## Requirements Files

### Protocol Interface Requirements

```txt
# services/protocol-interface/requirements.txt

# Core
fastapi>=0.100.0
pydantic>=2.0
pydantic-settings>=2.0
uvicorn[standard]>=0.23.0

# MCP & A2A
mcp>=0.9.0
a2a-sdk-python>=1.0.0

# Authentication
PyJWT>=2.8.0
cryptography>=41.0.0
google-auth>=2.26.0
azure-identity>=1.15.0
google-cloud-iam>=2.14.0

# HTTP & Async
httpx>=0.24.0
tenacity>=8.2.0

# Logging
structlog>=23.1.0
python-json-logger>=2.0.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
httpx-mock>=0.21.0
```

### SOP Engine Requirements

```txt
# services/sop-engine/requirements.txt

# Core
fastapi>=0.100.0
pydantic>=2.0
pydantic-settings>=2.0
uvicorn[standard]>=0.23.0

# Vector DB & Embeddings
chromadb>=0.4.0
ollama>=0.1.0

# Schema & Query Intelligence
graphrag>=0.1.0
langchain>=0.1.0
networkx>=3.0

# Document Parsing
python-docx>=0.8.11
PyPDF2>=3.0.0
pdfplumber>=0.9.0
markdown-it-py>=3.0.0

# Search & Ranking
rank-bm25>=0.2.2

# Utilities
tiktoken>=0.5.0
httpx>=0.24.0
tenacity>=8.2.0

# Logging
structlog>=23.1.0
python-json-logger>=2.0.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
```

---

## Testing Strategy

### Protocol Interface Testing

```python
# services/protocol-interface/tests/test_mcp_server.py

@pytest.mark.asyncio
async def test_mcp_tool_registration():
    """Test MCP tool registration"""
    server = MCPServer()
    
    server.register_tool(
        name="test_tool",
        description="Test tool",
        input_schema={"query": "string"},
        handler=mock_handler
    )
    
    tools = await server.list_tools()
    assert len(tools["tools"]) > 0
    assert tools["tools"][0].name == "test_tool"

@pytest.mark.asyncio
async def test_a2a_agent_discovery():
    """Test A2A agent discovery"""
    handler = A2AHandler(config)
    agents = await handler.discover_agents("planner")
    
    assert len(agents) > 0
    assert agents[0].agent_type == "planner"

@pytest.mark.asyncio
async def test_oauth2_token_exchange():
    """Test OAuth2 OBO token exchange"""
    auth = GoogleAuthProvider(config)
    token = await auth.exchange_token_obo(
        user_token="user_token",
        target_service="sop-engine"
    )
    
    assert token is not None
    assert isinstance(token, str)
```

### SOP Engine Testing

```python
# services/sop-engine/tests/test_document_parser.py

@pytest.mark.asyncio
async def test_markdown_parser():
    """Test Markdown document parsing"""
    parser = MarkdownParser()
    
    doc = SOPDocument(
        id="test_doc",
        content="## Procedure 1\n- Step 1\n- Step 2"
    )
    
    procedures = await parser.parse(doc)
    assert len(procedures) > 0
    assert procedures[0].title == "Procedure 1"
    assert len(procedures[0].steps) == 2

@pytest.mark.asyncio
async def test_vector_store_search():
    """Test semantic search in ChromaDB"""
    store = VectorStoreManager(config)
    
    # Add procedure
    procedure = ExtractedProcedure(
        title="Database Connection Error",
        steps=["Check connection string", "Verify credentials"]
    )
    await store.add_procedure(procedure, uuid4())
    
    # Search
    results = await store.search("database error")
    assert len(results) > 0
    assert results[0].similarity_score > 0.5

@pytest.mark.asyncio
async def test_graphrag_schema_extraction():
    """Test GraphRAG schema extraction"""
    analyzer = GraphRAGAnalyzer(config)
    
    docs = [
        Document(
            content="Table Users (id INT, name VARCHAR)"
        ),
        Document(
            content="Table Orders (id INT, user_id INT)"
        )
    ]
    
    schema = await analyzer.extract_schema(docs)
    assert len(schema.entities) >= 2
    assert len(schema.relationships) > 0
```

---

## Success Criteria

### Phase 4 Completion Criteria

#### Protocol Interface Service
- [ ] ✅ MCP server accepting and executing tools
- [ ] ✅ A2A agent discovery working
- [ ] ✅ OAuth2 authentication (Google, Azure, GCP)
- [ ] ✅ Token OBO exchange functional
- [ ] ✅ Cost tracking for all operations
- [ ] ✅ Docker image builds and runs
- [ ] ✅ All tests passing (>90% coverage)
- [ ] ✅ Integration with core engine verified

#### SOP Engine Service
- [ ] ✅ Document parsing for all formats (MD, DOCX, PDF, TXT)
- [ ] ✅ ChromaDB with Ollama embeddings working
- [ ] ✅ Semantic search operational
- [ ] ✅ GraphRAG schema extraction functional
- [ ] ✅ Hybrid search (vector + keyword) implemented
- [ ] ✅ REST API endpoints working
- [ ] ✅ Docker image builds and runs
- [ ] ✅ All tests passing (>90% coverage)
- [ ] ✅ Integration with core engine verified

### Quality Metrics
- Code coverage: > 90%
- Type hints: 100%
- Docstring coverage: 100%
- Linting: All checks pass
- Error handling: Comprehensive
- Logging: Structured throughout

---

## Risk Mitigation

### Key Risks

1. **Ollama Model Performance**
   - Mitigated by: Model selection testing, fallback to simpler models
   - Contingency: Use CPU-optimized models, implement caching

2. **ChromaDB Scalability**
   - Mitigated by: Early load testing, schema optimization
   - Contingency: Implement sharding strategy

3. **GraphRAG Accuracy**
   - Mitigated by: Schema validation, manual review process
   - Contingency: Fallback to rule-based extraction

4. **OAuth2 Token Expiry**
   - Mitigated by: Automatic refresh token handling
   - Contingency: Implement token caching with early refresh

5. **A2A SDK Compatibility**
   - Mitigated by: Version pinning, early integration testing
   - Contingency: Implement HTTP fallback

---

## Next Steps

### Immediate (This Week)

1. [ ] Create Protocol Interface service directory structure
2. [ ] Set up MCP server framework
3. [ ] Initialize A2A SDK integration
4. [ ] Create SOP Engine service directory structure
5. [ ] Set up document parser framework

### Week 1-2

1. [ ] Complete MCP server implementation
2. [ ] Implement A2A protocol handler
3. [ ] Build OAuth2 authentication providers
4. [ ] Write comprehensive tests

### Week 3-4

1. [ ] Implement all document parsers
2. [ ] Integrate ChromaDB
3. [ ] Connect Ollama for embeddings
4. [ ] Implement semantic search

### Week 5-6

1. [ ] Implement GraphRAG schema analyzer
2. [ ] Build SOP Engine API
3. [ ] Complete testing and documentation
4. [ ] Docker integration and deployment

---

## Conclusion

Phase 4 represents a significant step forward, implementing core infrastructure services that enable the full AI-powered query intelligence pipeline. With the Protocol Interface Service handling communication and the SOP Engine providing semantic document retrieval and schema understanding, the system will be ready for the final phases of implementation.

**Estimated Effort**: 6 weeks  
**Team Size**: 2-3 developers  
**Code Volume Expected**: ~3,500+ lines of real implementation code

This plan provides a clear roadmap for implementing enterprise-grade services while maintaining the privacy-first, local-deployment principles established in earlier phases.
