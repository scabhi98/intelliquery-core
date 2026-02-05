# LexiQuery Development Guide

## Development Environment Strategy

LexiQuery Phase 2+ uses a **single root-level virtual environment** approach for optimal development experience.

### Why Single Root .venv?

**Problems with per-service venvs**:
- ❌ VS Code workspace conflicts
- ❌ Multiple Python interpreters to manage
- ❌ Duplicate dependencies across services
- ❌ Difficult to run multiple services simultaneously

**Benefits of root .venv**:
- ✅ Single Python interpreter for entire workspace
- ✅ No VS Code conflicts
- ✅ Shared dependencies (faster, less disk space)
- ✅ Easy to run all services from one terminal
- ✅ Simpler debugging across services

### Docker Strategy

**Development**: Single root .venv with all dependencies
**Production/Docker**: Lightweight images with only required dependencies

**requirements.txt files** are kept in each service directory for:
- Docker image builds (minimal per-service deps)
- Documentation of service dependencies
- Future microservice deployment

---

## Quick Start

### 1. Initial Setup

```powershell
# Clone and navigate to project
cd j:\projects\engine-core

# Run setup script (creates .venv and installs all deps)
.\scripts\setup_envs.ps1

# Clean setup (removes existing .venv first)
.\scripts\setup_envs.ps1 -Clean
```

**What the setup does**:
1. Creates `.venv/` at project root
2. Installs shared dependencies from `shared/requirements.txt`
3. Loops through all services and installs their dependencies
4. Optionally builds wheel distributions for deployment

### 2. Running Services

**Option A: All Services at Once (Recommended for Testing)**

```powershell
# Start all 7 services in separate windows
.\scripts\run_services.ps1

# Services started:
# - Core Engine (8000)
# - Planner Agent (9000)
# - SOP Knowledge (9010)
# - Error Knowledge (9011)
# - KQL Data Agent (9020)
# - SPL Data Agent (9021)
# - SQL Data Agent (9022)
```

**Option B: Individual Services (Recommended for Development)**

```powershell
# Activate venv
.\.venv\Scripts\Activate.ps1

# Start specific service
python -m uvicorn services.core_engine.src.main:app --port 8000 --reload

# Or with hot reload for development
python -m uvicorn services.mock_agents.planner.src.main:app --port 9000 --reload
```

**Option C: VS Code Tasks (Coming Soon)**

Create `.vscode/tasks.json` for one-click service launching.

### 3. Testing

**Quick API Test**:
```powershell
.\scripts\test_phase2.ps1
```

**Integration Tests**:
```powershell
.\.venv\Scripts\Activate.ps1
cd services/core-engine
pytest tests/test_integration.py -v
```

**Manual API Test**:
```powershell
curl -X POST http://localhost:8000/api/v1/query `
  -H "Content-Type: application/json" `
  -d '{
    "natural_language": "Show failed login attempts",
    "platform": "kql",
    "user_id": "test_user"
  }'
```

### 4. Checking Service Status

```powershell
# Check which services are running
.\scripts\run_services.ps1 -Status

# Output:
# ✓ Core Engine (Port 8000) - RUNNING
# ✓ Planner Agent (Port 9000) - RUNNING
# ...
```

### 5. Stopping Services

```powershell
# Stop all services started with run_services.ps1
.\scripts\run_services.ps1 -Stop

# Or manually Ctrl+C in each terminal
```

---

## VS Code Configuration

### 1. Select Python Interpreter

1. Press `Ctrl+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose `.\.venv\Scripts\python.exe`

### 2. Workspace Settings

Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe",
  "python.terminal.activateEnvironment": true,
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["services/core-engine/tests"],
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.tabSize": 4
  }
}
```

### 3. Launch Configurations

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Core Engine",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": [
        "services.core_engine.src.main:app",
        "--port", "8000",
        "--reload"
      ],
      "jinja": true,
      "justMyCode": false
    },
    {
      "name": "Planner Agent",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": [
        "services.mock_agents.planner.src.main:app",
        "--port", "9000",
        "--reload"
      ]
    }
  ]
}
```

---

## Development Workflow

### Typical Development Session

```powershell
# 1. Activate venv
.\.venv\Scripts\Activate.ps1

# 2. Start services you're working on
python -m uvicorn services.core_engine.src.main:app --port 8000 --reload

# 3. Make code changes (auto-reloads with --reload flag)

# 4. Test changes
.\scripts\test_phase2.ps1

# 5. Run unit tests
pytest services/core-engine/tests/ -v

# 6. Commit changes
git add .
git commit -m "feat: implemented feature X"
```

### Adding New Service

1. Create service directory structure:
```
services/new-service/
├── src/
│   ├── __init__.py
│   └── main.py
├── tests/
│   └── __init__.py
├── requirements.txt
├── Dockerfile
└── README.md
```

2. Add dependencies to `requirements.txt`

3. Update `scripts/setup_envs.ps1` to include new service

4. Update `docker-compose.yml` with new service

5. Run setup:
```powershell
.\scripts\setup_envs.ps1
```

---

## Debugging

### Debugging in VS Code

1. Set breakpoints in code (F9)
2. Select launch configuration from dropdown
3. Press F5 to start debugging
4. Use Debug Console to inspect variables

### Logging

All services use structured logging (structlog):

```python
import structlog

logger = structlog.get_logger()

logger.info(
    "processing_query",
    journey_id=str(journey_id),
    platform=platform,
    query_length=len(query)
)
```

Logs are in JSON format for easy parsing:
```json
{
  "event": "processing_query",
  "journey_id": "uuid",
  "platform": "kql",
  "query_length": 150,
  "timestamp": "2026-02-05T10:30:00Z",
  "level": "info"
}
```

### Common Issues

**Issue**: ImportError: No module named 'shared'
```powershell
# Solution: Ensure PYTHONPATH includes project root
$env:PYTHONPATH = "j:\projects\engine-core"
```

**Issue**: Port already in use
```powershell
# Solution: Find and kill process on port
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Issue**: VS Code not finding interpreter
```powershell
# Solution: Reload window
# Ctrl+Shift+P -> "Developer: Reload Window"
```

---

## Docker Development

### Building Images

```powershell
# Build all services
docker-compose build

# Build specific service
docker-compose build core-engine

# No cache build
docker-compose build --no-cache
```

### Running with Docker

```powershell
# Start all
docker-compose up -d

# View logs
docker-compose logs -f core-engine

# Execute commands in container
docker-compose exec core-engine python --version

# Stop all
docker-compose down
```

### Docker + Local Hybrid

Run some services in Docker, some locally:

```powershell
# Start only dependencies in Docker
docker-compose up -d mock-planner mock-sop-knowledge

# Run core engine locally for debugging
python -m uvicorn services.core_engine.src.main:app --port 8000 --reload
```

---

## Best Practices

### Code Style

- ✅ Use type hints everywhere
- ✅ Pydantic models for validation
- ✅ Async/await for I/O operations
- ✅ Comprehensive docstrings
- ✅ Structured logging

### Testing

- ✅ Write tests alongside code
- ✅ Test at integration level
- ✅ Mock external dependencies
- ✅ Use pytest fixtures

### Git Workflow

```powershell
# Create feature branch
git checkout -b feature/new-agent

# Make changes and commit frequently
git add .
git commit -m "feat: add new agent capability"

# Push to remote
git push origin feature/new-agent

# Create pull request
```

### Performance

- ✅ Use `--reload` only in development
- ✅ Profile with `py-spy` or `cProfile`
- ✅ Monitor memory with logs
- ✅ Test with realistic data volumes

---

## Deployment Preparation

### Building for Production

```powershell
# Build wheel distributions
python setup.py bdist_wheel

# Install from wheel in Docker
pip install dist/lexi_shared-1.0.0-py3-none-any.whl
```

### Environment Variables

Create `.env` file:
```env
CORE_ENGINE_SERVICE_PORT=8000
CORE_ENGINE_LOG_LEVEL=INFO
CORE_ENGINE_MAX_RETRIES=3
CORE_ENGINE_PLANNER_AGENT_URL=http://mock-planner:9000
```

Load in Docker:
```yaml
services:
  core-engine:
    env_file:
      - .env
```

---

## Troubleshooting

### Setup Issues

**Problem**: `setup_envs.ps1` fails
```powershell
# Check Python version
python --version  # Should be 3.11+

# Run with clean flag
.\scripts\setup_envs.ps1 -Clean
```

**Problem**: Dependencies fail to install
```powershell
# Upgrade pip
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# Install dependencies manually
.\.venv\Scripts\pip.exe install -r shared/requirements.txt
```

### Runtime Issues

**Problem**: Service won't start
```powershell
# Check for port conflicts
netstat -ano | findstr :8000

# Check logs for errors
# Look in PowerShell window output
```

**Problem**: Import errors
```powershell
# Verify PYTHONPATH
echo $env:PYTHONPATH  # Should be project root

# Set manually if needed
$env:PYTHONPATH = "j:\projects\engine-core"
```

---

## Resources

### Documentation
- [PHASE2_README.md](../PHASE2_README.md) - Complete Phase 2 guide
- [PHASE2_COMPLETION_SUMMARY.md](../PHASE2_COMPLETION_SUMMARY.md) - Implementation details
- [.github/copilot-instructions.md](../.github/copilot-instructions.md) - Development guidelines

### Scripts
- `scripts/setup_envs.ps1` - Environment setup
- `scripts/run_services.ps1` - Start all services
- `scripts/test_phase2.ps1` - Quick API tests
- `scripts/build_all.ps1` - Build Docker images

### Key Files
- `docker-compose.yml` - Service orchestration
- `shared/requirements.txt` - Shared dependencies
- `services/*/requirements.txt` - Service-specific deps

---

## Getting Help

1. Check existing documentation
2. Review error messages and logs
3. Search codebase for similar patterns
4. Consult `.github/copilot-instructions.md`
5. Ask team members

Happy coding! 🚀
