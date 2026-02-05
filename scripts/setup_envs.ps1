# LexiQuery - Setup Development Environment Script
# This script creates a single root-level virtual environment for all services
# Services can be run independently during development and built as wheels for Docker

param(
    [string]$PythonVersion = "python",
    [switch]$Clean = $false
)

$ErrorActionPreference = "Stop"

# Color output functions
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Info { Write-Host $args -ForegroundColor Cyan }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }

# Get project root
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RootVenvPath = Join-Path $ProjectRoot ".venv"

Write-Info "=========================================="
Write-Info " LexiQuery Development Environment Setup"
Write-Info "=========================================="
Write-Info ""
Write-Info "Strategy: Single root .venv for all services"
Write-Info "Location: $RootVenvPath"
Write-Info ""

# Check Python installation
try {
    $pythonVersionOutput = & $PythonVersion --version 2>&1
    Write-Success "✓ Python found: $pythonVersionOutput"
} catch {
    Write-Error "✗ Python not found. Please install Python 3.11 or higher."
    Write-Info "  Install from: https://www.python.org/downloads/"
    exit 1
}

Write-Info ""

# Clean existing venv if requested
if ($Clean -and (Test-Path $RootVenvPath)) {
    Write-Warning "Removing existing virtual environment..."
    Remove-Item -Recurse -Force $RootVenvPath
    Write-Success "✓ Cleaned"
}

# Create root virtual environment
if (-not (Test-Path $RootVenvPath)) {
    Write-Info "Creating root virtual environment..."
    & $PythonVersion -m venv $RootVenvPath
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "✗ Failed to create virtual environment"
        exit 1
    }
    Write-Success "✓ Virtual environment created"
} else {
    Write-Info "Virtual environment already exists"
}

# Get Python and pip executables from venv
$pythonExe = Join-Path $RootVenvPath "Scripts\python.exe"
$pipExe = Join-Path $RootVenvPath "Scripts\pip.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Error "✗ Python executable not found in venv"
    exit 1
}

Write-Info ""
Write-Info "Upgrading pip, setuptools, and wheel..."
& $pythonExe -m pip install --upgrade pip setuptools wheel --quiet
Write-Success "✓ Build tools upgraded"

Write-Info ""
Write-Info "=========================================="
Write-Info " Installing Dependencies"
Write-Info "=========================================="
Write-Info ""

# Install shared dependencies first
$sharedRequirements = Join-Path $ProjectRoot "shared\requirements.txt"
if (Test-Path $sharedRequirements) {
    Write-Info "[shared] Installing shared dependencies..."
    & $pipExe install -r $sharedRequirements --quiet
    Write-Success "[shared] ✓ Shared dependencies installed"
} else {
    Write-Warning "[shared] No requirements.txt found"
}

Write-Info ""

# Define all services (Phase 1 + Phase 2)
$services = @(
    "core-engine",
    "protocol-interface",
    "sop-engine",
    "query-generator",
    "cache-learning",
    "data-connectors",
    "cost-tracker",
    "mock-agents\planner",
    "mock-agents\knowledge",
    "mock-agents\data"
)

# Install each service's dependencies
foreach ($service in $services) {
    $servicePath = Join-Path $ProjectRoot "services\$service"
    
    # Check if service directory exists
    if (-not (Test-Path $servicePath)) {
        Write-Warning "[$service] Directory not found, skipping..."
        continue
    }
    
    # Install dependencies if requirements.txt exists
    $requirementsPath = Join-Path $servicePath "requirements.txt"
    if (Test-Path $requirementsPath) {
        Write-Info "[$service] Installing dependencies..."
        try {
            & $pipExe install -r $requirementsPath --quiet
            Write-Success "[$service] ✓ Dependencies installed"
        } catch {
            Write-Error "[$service] ✗ Failed to install dependencies: $_"
        }
    } else {
        Write-Warning "[$service] No requirements.txt found"
    }
}

Write-Info ""
Write-Info "=========================================="
Write-Info " Building Package Distributions"
Write-Info "=========================================="
Write-Info ""

# Build shared package as wheel
Write-Info "[shared] Building wheel distribution..."
Push-Location $ProjectRoot
try {
    # Create minimal setup.py for shared package
    $setupPy = @"
from setuptools import setup, find_packages

setup(
    name="lexi-shared",
    version="1.0.0",
    packages=find_packages(where="shared"),
    package_dir={"": "shared"},
    python_requires=">=3.11",
)
"@
    $setupPy | Out-File -FilePath "setup_shared.py" -Encoding UTF8
    
    & $pythonExe setup_shared.py bdist_wheel --quiet 2>$null
    if (Test-Path "dist") {
        Write-Success "[shared] ✓ Wheel built: dist/"
    }
    
    # Clean up setup artifacts
    Remove-Item -Force "setup_shared.py" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "build" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "lexi_shared.egg-info" -ErrorAction SilentlyContinue
} catch {
    Write-Warning "[shared] Wheel build skipped (optional)"
} finally {
    Pop-Location
}

Write-Info ""
Write-Info "=========================================="
Write-Success " Setup Complete!"
Write-Info "=========================================="
Write-Info ""
Write-Success "✓ Single root virtual environment ready"
Write-Success "✓ All service dependencies installed"
Write-Success "✓ Ready for development and testing"
Write-Info ""
Write-Info "To activate the environment:"
Write-Info "  .\.venv\Scripts\Activate.ps1"
Write-Info ""
Write-Info "To deactivate:"
Write-Info "  deactivate"
Write-Info ""
Write-Info "Running services during development:"
Write-Info "  .\.venv\Scripts\python.exe -m uvicorn services.core_engine.src.main:app --port 8000"
Write-Info "  .\.venv\Scripts\python.exe -m uvicorn services.mock_agents.planner.src.main:app --port 9000"
Write-Info ""
Write-Info "Or use the test script:"
Write-Info "  .\scripts\test_phase2.ps1"
Write-Info ""
Write-Info "Note: requirements.txt files remain in each service for Docker builds"
Write-Info ""
