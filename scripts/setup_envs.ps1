# LexiQuery - Setup Virtual Environments Script
# This script creates isolated Python virtual environments for all services

param(
    [string]$PythonVersion = "py",
    [string]$ServiceName = $null
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
Write-Info " LexiQuery Environment Setup"
Write-Info "==================================="
Write-Info ""

# Check Python installation
# try {
#     $pythonVersionOutput = & $PythonVersion --version 2>&1
#     Write-Success "✓ Python found: $pythonVersionOutput"
# } catch {
#     Write-Error "✗ Python not found. Please install Python 3.11 or higher."
#     exit 1
# }

# Define services
$services = @(
    "core-engine",
    "protocol-interface",
    "sop-engine",
    "query-generator",
    "cache-learning",
    "data-connectors",
    "cost-tracker"
)

# If specific service provided, only setup that one
if ($ServiceName) {
    if ($services -contains $ServiceName) {
        $services = @($ServiceName)
        Write-Info "Setting up virtual environment for: $ServiceName"
    } else {
        Write-Error "Unknown service: $ServiceName"
        Write-Info "Available services: $($services -join ', ')"
        exit 1
    }
} else {
    Write-Info "Setting up virtual environments for all services..."
}

Write-Info ""

# Setup each service
foreach ($service in $services) {
    $servicePath = Join-Path $ProjectRoot "services\$service"
    $venvPath = Join-Path $servicePath ".venv"
    
    Write-Info "[$service] Setting up environment..."
    
    # Check if service directory exists
    if (-not (Test-Path $servicePath)) {
        Write-Warning "[$service] Directory not found: $servicePath"
        continue
    }
    
    # Remove existing venv if present
    if (Test-Path $venvPath) {
        Write-Warning "[$service] Removing existing virtual environment..."
        Remove-Item -Recurse -Force $venvPath
    }
    
    # Create virtual environment
    try {
        Write-Info "[$service] Creating virtual environment..."
        & $PythonVersion -m venv $venvPath
        
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create virtual environment"
        }
        
        # Use direct python.exe path (no activation needed in scripts)
        $pythonExe = Join-Path $venvPath "Scripts\python.exe"
        $pipExe = Join-Path $venvPath "Scripts\pip.exe"
        
        if (Test-Path $pythonExe) {
            Write-Info "[$service] Upgrading pip..."
            & $pythonExe -m pip install --upgrade pip setuptools wheel --quiet
            
            # Install dependencies if requirements.txt exists
            $requirementsPath = Join-Path $servicePath "requirements.txt"
            if (Test-Path $requirementsPath) {
                Write-Info "[$service] Installing dependencies..."
                & $pipExe install -r $requirementsPath --quiet
                Write-Success "[$service] ✓ Dependencies installed"
            } else {
                Write-Warning "[$service] No requirements.txt found"
            }
            
            Write-Success "[$service] ✓ Virtual environment ready"
        } else {
            throw "Python executable not found in venv"
        }
    } catch {
        Write-Error "[$service] ✗ Failed to setup: $_"
        continue
    }
    
    Write-Info ""
}

Write-Info "==================================="
Write-Success "Setup Complete!"
Write-Info "==================================="
Write-Info ""
Write-Info "To activate a service environment:"
Write-Info "  cd services\<service-name>"
Write-Info "  .\.venv\Scripts\Activate.ps1"
Write-Info ""
Write-Info "To deactivate:"
Write-Info "  deactivate"
Write-Info ""
