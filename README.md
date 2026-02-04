# LexiQuery

**Business-Process-Aware Query Intelligence Engine**

LexiQuery is a privacy-first, microservices-based query intelligence platform that leverages RAG-enhanced SOP processing with GraphRAG to generate optimized queries across multiple data platforms while maintaining conversational context and learning from successful interactions.

## 🎯 Key Features

- **Privacy-First**: Fully local deployment using Ollama for LLMs
- **Multi-Platform**: Generate queries for KQL (Azure Log Analytics), SPL (Splunk), and SQL
- **Business-Aware**: SOP-driven context understanding with ChromaDB vector storage
- **Intelligent**: GraphRAG for schema-aware query generation
- **Agentic**: CrewAI-powered orchestration
- **Cost-Transparent**: Comprehensive LLM cost tracking per journey
- **Scalable**: Microservices architecture with independent deployment

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    API Gateway Layer                         │
│              (MCP & A2A Protocol Handlers)                   │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────────────┐
│                    Core Engine                               │
│         (CrewAI Orchestration & Business Logic)              │
└──────┬────────────┬────────────┬────────────┬───────────────┘
       │            │            │            │
   ┌───▼───┐   ┌───▼───┐   ┌───▼───┐   ┌───▼────┐
   │  SOP  │   │ Query │   │Cache/ │   │  Data  │
   │Engine │   │  Gen  │   │Learn  │   │Connec- │
   │       │   │       │   │       │   │  tors  │
   └───────┘   └───────┘   └───────┘   └────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker Desktop
- Ollama (for local LLM deployment)
- 16GB RAM minimum

### Setup

1. **Clone the repository**
   ```powershell
   git clone <repository-url>
   cd engine-core
   ```

2. **Create virtual environments for all services**
   ```powershell
   .\scripts\setup_envs.ps1
   ```

3. **Or setup a specific service**
   ```powershell
   .\scripts\setup_envs.ps1 -ServiceName core-engine
   ```

4. **Install Ollama models**
   ```powershell
   ollama pull llama3
   ollama pull nomic-embed-text
   ```

### Development

**Activate a service environment:**
```powershell
cd services\<service-name>
.\.venv\Scripts\Activate.ps1
```

**Run tests:**
```powershell
.\scripts\run_tests.ps1                    # All services
.\scripts\run_tests.ps1 -ServiceName core-engine  # Specific service
.\scripts\run_tests.ps1 -Coverage          # With coverage report
```

**Build Docker images:**
```powershell
.\scripts\build_all.ps1                    # All services
.\scripts\build_all.ps1 -ServiceName sop-engine  # Specific service
```

**Create a new service:**
```powershell
.\scripts\init_service.ps1 -ServiceName my-service -WithMock
```

## 📦 Services

| Service | Port | Description |
|---------|------|-------------|
| **core-engine** | 8000 | CrewAI orchestration and business logic |
| **protocol-interface** | 8001 | MCP server/client + A2A SDK integration |
| **sop-engine** | 8002 | Document parsing, ChromaDB, GraphRAG |
| **query-generator** | 8003 | Ollama + GraphRAG query generation |
| **cache-learning** | 8004 | Pattern caching with PII elimination |
| **data-connectors** | 8005 | Azure, Splunk, SQL integrations |
| **cost-tracker** | 8006 | LLM cost tracking and analytics |

## 🔧 Technology Stack

- **Language**: Python 3.11+
- **Framework**: FastAPI
- **Vector DB**: ChromaDB
- **LLM**: Ollama (llama3, nomic-embed-text)
- **Agentic Framework**: CrewAI
- **Query Intelligence**: GraphRAG
- **Protocol Layer**: MCP + A2A SDK for Python
- **Cache**: Redis
- **Auth**: Abstract provider (Google/GCP/Azure)

## 📖 Documentation

- [Project Setup Plan](PROJECT_SETUP_PLAN.md) - Detailed development roadmap
- [Problem Statement](problem-statement.md) - System overview and requirements
- [GitHub Copilot Instructions](.github/copilot-instructions.md) - Coding guidelines

### Architecture Docs
- [docs/architecture/](docs/architecture/) - System architecture details
- [docs/api-specs/](docs/api-specs/) - API specifications
- [docs/deployment/](docs/deployment/) - Deployment guides

## 🧪 Testing

Run the full test suite:
```powershell
.\scripts\run_tests.ps1 -Coverage -Verbose
```

View coverage reports:
```
services\<service-name>\htmlcov\index.html
```

## 🐳 Docker Deployment

**Build all images:**
```powershell
.\scripts\build_all.ps1 -Registry lexiquery -Tag v1.0.0
```

**Run with docker-compose:**
```powershell
cd infra\docker
docker-compose up -d
```

## 🔐 Security

- **Privacy-First**: All LLM operations use local Ollama (no external API calls)
- **PII Elimination**: Automatic PII detection and removal in cached patterns
- **Abstract Auth**: Pluggable authentication (Google/GCP/Azure)
- **OAuth2 OBO**: On-Behalf-Of flow for multi-tenant access

## 💰 Cost Tracking

LexiQuery includes comprehensive cost tracking for all LLM and embedding operations:

- Per-operation token counting
- Journey-level cost aggregation
- Service and model breakdowns
- Cost optimization recommendations

**Query journey costs:**
```
GET /api/v1/cost/journey/{journey_id}
```

## 🤝 Contributing

1. Create a new branch for your feature
2. Implement changes following [Copilot Instructions](.github/copilot-instructions.md)
3. Write tests (95% coverage target)
4. Run `.\scripts\run_tests.ps1 -Coverage`
5. Submit pull request

### Coding Standards

- Follow PEP 8 and PEP 257
- Use type hints everywhere
- Prefer async/await for I/O
- Use Pydantic for data validation
- Implement interface-first design

## 📝 Development Phases

- [x] **Phase 1**: Foundation & Interfaces (Week 1-2)
- [ ] **Phase 2**: Core Engine + Mocks (Week 3-5)
- [ ] **Phase 3**: Mock Validation (Week 6)
- [ ] **Phase 4**: Real Implementation (Week 7+)

## 📄 License

[Add your license here]

## 👥 Authors

[Add author information]

## 🔗 Links

- [Documentation](docs/)
- [API Specifications](docs/api-specs/)
- [Architecture Diagrams](docs/architecture/)

---

**Built with ❤️ for privacy-first query intelligence**
