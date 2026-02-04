#!/bin/bash
# Initialize a new LexiQuery service with boilerplate code (Linux/Mac)

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Check if service name is provided
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: Service name is required${NC}"
    echo "Usage: $0 <service-name>"
    exit 1
fi

SERVICE_NAME=$1
SERVICE_DIR="$ROOT_DIR/services/$SERVICE_NAME"

# Check if service already exists
if [ -d "$SERVICE_DIR" ]; then
    echo -e "${RED}Error: Service '$SERVICE_NAME' already exists${NC}"
    exit 1
fi

echo -e "${YELLOW}Initializing new service: $SERVICE_NAME${NC}"
echo ""

# Create directory structure
echo "Creating directory structure..."
mkdir -p "$SERVICE_DIR/src"
mkdir -p "$SERVICE_DIR/tests"

# Create __init__.py files
touch "$SERVICE_DIR/src/__init__.py"
touch "$SERVICE_DIR/tests/__init__.py"

# Create main.py
cat > "$SERVICE_DIR/src/main.py" << 'EOF'
"""
Main entry point for the service.
"""
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import structlog

from shared.utils.logging_config import setup_logging
from .config import settings

# Setup logging
setup_logging(service_name=settings.service_name, log_level=settings.log_level)
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage service lifecycle."""
    # Startup
    logger.info("service_starting", service=settings.service_name)
    # Initialize service components here
    yield
    # Shutdown
    logger.info("service_stopping", service=settings.service_name)


app = FastAPI(
    title=f"{settings.service_name.title()} Service",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.service_name
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": settings.service_name,
        "version": "1.0.0",
        "status": "running"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.service_port,
        reload=True
    )
EOF

# Create config.py
cat > "$SERVICE_DIR/src/config.py" << 'EOF'
"""
Service configuration.
"""
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
    service_name: str = "SERVICE_NAME_PLACEHOLDER"
    service_port: int = 8000
    log_level: str = "INFO"
    
    # Add service-specific settings here


# Global settings instance
settings = Settings()
EOF

# Replace placeholder in config.py
sed -i.bak "s/SERVICE_NAME_PLACEHOLDER/$SERVICE_NAME/g" "$SERVICE_DIR/src/config.py"
rm "$SERVICE_DIR/src/config.py.bak"

# Create requirements.txt
cat > "$SERVICE_DIR/requirements.txt" << 'EOF'
# Include shared dependencies
-r ../../shared/requirements.txt

# Service-specific dependencies
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
pydantic-settings>=2.0.0
EOF

# Create Dockerfile
cat > "$SERVICE_DIR/Dockerfile" << 'EOF'
# Multi-stage Dockerfile for service

# Builder stage
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy shared dependencies
COPY shared/ ./shared/

# Copy service requirements
ARG SERVICE_NAME
COPY services/${SERVICE_NAME}/requirements.txt ./requirements.txt

# Install dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy shared code
COPY shared/ ./shared/

# Copy service code
ARG SERVICE_NAME
COPY services/${SERVICE_NAME}/src/ ./src/

# Add local bin to PATH
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# Create test file
cat > "$SERVICE_DIR/tests/test_main.py" << 'EOF'
"""
Tests for main service endpoints.
"""
import pytest
from httpx import AsyncClient
from src.main import app


@pytest.mark.asyncio
async def test_health_check():
    """Test health check endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_root_endpoint():
    """Test root endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
EOF

# Create .env template
cat > "$SERVICE_DIR/.env.example" << 'EOF'
# Service Configuration
SERVICE_NAME=SERVICE_NAME_PLACEHOLDER
SERVICE_PORT=8000
LOG_LEVEL=INFO

# Add service-specific environment variables here
EOF

sed -i.bak "s/SERVICE_NAME_PLACEHOLDER/$SERVICE_NAME/g" "$SERVICE_DIR/.env.example"
rm "$SERVICE_DIR/.env.example.bak"

# Create README
cat > "$SERVICE_DIR/README.md" << 'EOF'
# SERVICE_NAME_PLACEHOLDER Service

## Description

[Add service description here]

## Setup

1. Create virtual environment:
```bash
cd services/SERVICE_NAME_PLACEHOLDER
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

## Running

### Development
```bash
python -m src.main
```

### Production
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## Testing

```bash
pytest tests/
```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Endpoints

- `GET /health` - Health check
- `GET /` - Service information

[Add service-specific endpoints here]
EOF

sed -i.bak "s/SERVICE_NAME_PLACEHOLDER/$SERVICE_NAME/g" "$SERVICE_DIR/README.md"
rm "$SERVICE_DIR/README.md.bak"

echo -e "${GREEN}✓ Service structure created${NC}"
echo ""
echo "Created files:"
echo "  - src/main.py"
echo "  - src/config.py"
echo "  - tests/test_main.py"
echo "  - requirements.txt"
echo "  - Dockerfile"
echo "  - .env.example"
echo "  - README.md"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Implement service logic in src/"
echo "2. Add tests in tests/"
echo "3. Update requirements.txt with dependencies"
echo "4. Configure .env.example with required variables"
echo "5. Run: cd services/$SERVICE_NAME && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
echo ""
