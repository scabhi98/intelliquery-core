# LexiQuery - Initialize New Service Script
# This script creates the directory structure for a new microservice

param(
    [Parameter(Mandatory=$true)]
    [string]$ServiceName,
    
    [switch]$WithMock
)

$ErrorActionPreference = "Stop"

# Color output functions
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Info { Write-Host $args -ForegroundColor Cyan }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }

# Get project root
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Info "==================================="
Write-Info " Create New LexiQuery Service"
Write-Info "==================================="
Write-Info ""
Write-Info "Service name: $ServiceName"
Write-Info ""

# Validate service name
if ($ServiceName -notmatch '^[a-z][a-z0-9-]*$') {
    Write-Error "Invalid service name. Use lowercase letters, numbers, and hyphens only."
    Write-Info "Example: my-service"
    exit 1
}

$servicePath = Join-Path $ProjectRoot "services\$ServiceName"

# Check if service already exists
if (Test-Path $servicePath) {
    Write-Error "Service '$ServiceName' already exists!"
    exit 1
}

# Create directory structure
Write-Info "Creating directory structure..."

$directories = @(
    "$servicePath\src",
    "$servicePath\tests"
)

if ($WithMock) {
    $directories += "$servicePath\src\mock"
}

foreach ($dir in $directories) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
    Write-Success "  ✓ Created: $dir"
}

# Create __init__.py files
Write-Info "Creating Python package files..."

$initFiles = @(
    "$servicePath\src\__init__.py",
    "$servicePath\tests\__init__.py"
)

if ($WithMock) {
    $initFiles += "$servicePath\src\mock\__init__.py"
}

foreach ($file in $initFiles) {
    '"""' + "`n" + "$ServiceName package.`n" + '"""' | Out-File -FilePath $file -Encoding UTF8
    Write-Success "  ✓ Created: $file"
}

# Create main.py
$mainContent = @"
"""$ServiceName service - FastAPI application."""

from fastapi import FastAPI
from contextlib import asynccontextmanager

# TODO: Import service interface and implementation
# from shared.interfaces.your_interface import YourInterface
# from .service import YourService

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage service lifecycle."""
    # Startup
    # app.state.service = YourService()
    # await app.state.service.initialize()
    yield
    # Shutdown
    # await app.state.service.cleanup()

app = FastAPI(
    title="$ServiceName Service",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "$ServiceName"
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "$ServiceName",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"@

$mainPath = Join-Path $servicePath "src\main.py"
$mainContent | Out-File -FilePath $mainPath -Encoding UTF8
Write-Success "  ✓ Created: $mainPath"

# Create config.py
$configContent = @"
"""Configuration management for $ServiceName service."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Service configuration from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        env_prefix='${ServiceName.ToUpper().Replace('-','_')}_'
    )
    
    # Service configuration
    service_name: str = "$ServiceName"
    service_port: int = 8000
    log_level: str = "INFO"
    
    # Add service-specific configuration here
    
    # Ollama (if needed)
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    
    # Redis (if needed)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None


# Global settings instance
settings = Settings()
"@

$configPath = Join-Path $servicePath "src\config.py"
$configContent | Out-File -FilePath $configPath -Encoding UTF8
Write-Success "  ✓ Created: $configPath"

# Create requirements.txt
$requirementsContent = @"
# Core dependencies
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
pydantic>=2.0
pydantic-settings>=2.0.0

# Async HTTP client
httpx>=0.24.0

# Logging
structlog>=23.1.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0

# Add service-specific dependencies below
"@

$requirementsPath = Join-Path $servicePath "requirements.txt"
$requirementsContent | Out-File -FilePath $requirementsPath -Encoding UTF8
Write-Success "  ✓ Created: $requirementsPath"

# Create Dockerfile
$dockerfileContent = @"
# Multi-stage Dockerfile for $ServiceName

# Builder stage
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY services/$ServiceName/requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy shared code
COPY shared/ ./shared/

# Copy service code
COPY services/$ServiceName/src/ ./src/

# Add local bin to PATH
ENV PATH=/root/.local/bin:`$PATH
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()" || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
"@

$dockerfilePath = Join-Path $servicePath "Dockerfile"
$dockerfileContent | Out-File -FilePath $dockerfilePath -Encoding UTF8
Write-Success "  ✓ Created: $dockerfilePath"

# Create .env.example
$envExampleContent = @"
# $ServiceName Service Configuration

# Service settings
${ServiceName.ToUpper().Replace('-','_')}_SERVICE_PORT=8000
${ServiceName.ToUpper().Replace('-','_')}_LOG_LEVEL=INFO

# Ollama settings (if needed)
${ServiceName.ToUpper().Replace('-','_')}_OLLAMA_HOST=http://localhost:11434
${ServiceName.ToUpper().Replace('-','_')}_OLLAMA_MODEL=llama3

# Redis settings (if needed)
${ServiceName.ToUpper().Replace('-','_')}_REDIS_HOST=localhost
${ServiceName.ToUpper().Replace('-','_')}_REDIS_PORT=6379

# Add service-specific environment variables below
"@

$envExamplePath = Join-Path $servicePath ".env.example"
$envExampleContent | Out-File -FilePath $envExamplePath -Encoding UTF8
Write-Success "  ✓ Created: $envExamplePath"

# Create sample test
$testContent = @"
"""Tests for $ServiceName service."""

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["service"] == "$ServiceName"


def test_root(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "$ServiceName"
"@

$testPath = Join-Path $servicePath "tests\test_main.py"
$testContent | Out-File -FilePath $testPath -Encoding UTF8
Write-Success "  ✓ Created: $testPath"

Write-Info ""
Write-Success "==================================="
Write-Success "Service '$ServiceName' created!"
Write-Success "==================================="
Write-Info ""
Write-Info "Next steps:"
Write-Info "  1. cd services\$ServiceName"
Write-Info "  2. Create virtual environment:"
Write-Info "     ..\..\scripts\setup_envs.ps1 -ServiceName $ServiceName"
Write-Info "  3. Implement your service interface in src/"
Write-Info "  4. Add tests in tests/"
Write-Info "  5. Build Docker image:"
Write-Info "     ..\..\scripts\build_all.ps1 -ServiceName $ServiceName"
Write-Info ""
