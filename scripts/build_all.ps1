# LexiQuery - Build All Docker Images Script
# This script builds Docker images for all services

param(
    [string]$ServiceName = $null,
    [string]$Tag = "latest",
    [string]$Registry = "lexiquery"
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
Write-Info " LexiQuery Docker Build"
Write-Info "==================================="
Write-Info ""

# Check Docker installation
try {
    $dockerVersion = docker --version
    Write-Success "✓ Docker found: $dockerVersion"
} catch {
    Write-Error "✗ Docker not found. Please install Docker Desktop."
    exit 1
}

Write-Info ""

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

# If specific service provided, only build that one
if ($ServiceName) {
    if ($services -contains $ServiceName) {
        $services = @($ServiceName)
        Write-Info "Building Docker image for: $ServiceName"
    } else {
        Write-Error "Unknown service: $ServiceName"
        Write-Info "Available services: $($services -join ', ')"
        exit 1
    }
} else {
    Write-Info "Building Docker images for all services..."
}

Write-Info ""

# Build each service
$successCount = 0
$failCount = 0

foreach ($service in $services) {
    $servicePath = Join-Path $ProjectRoot "services\$service"
    $dockerfilePath = Join-Path $servicePath "Dockerfile"
    $imageName = "$Registry/$service`:$Tag"
    
    Write-Info "[$service] Building Docker image..."
    Write-Info "  Image: $imageName"
    
    # Check if Dockerfile exists
    if (-not (Test-Path $dockerfilePath)) {
        Write-Warning "[$service] Dockerfile not found, skipping..."
        $failCount++
        continue
    }
    
    try {
        # Build Docker image
        $buildArgs = @(
            "build",
            "-t", $imageName,
            "-f", $dockerfilePath,
            $ProjectRoot  # Build context is project root to access shared code
        )
        
        & docker $buildArgs
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "[$service] ✓ Image built successfully"
            $successCount++
        } else {
            throw "Docker build failed with exit code: $LASTEXITCODE"
        }
    } catch {
        Write-Error "[$service] ✗ Build failed: $_"
        $failCount++
    }
    
    Write-Info ""
}

Write-Info "==================================="
Write-Info "Build Summary"
Write-Info "==================================="
Write-Success "Successfully built: $successCount"
if ($failCount -gt 0) {
    Write-Error "Failed: $failCount"
}
Write-Info ""
Write-Info "To run a service container:"
Write-Info "  docker run -p 8000:8000 $Registry/<service-name>:$Tag"
Write-Info ""
Write-Info "To view images:"
Write-Info "  docker images | findstr $Registry"
Write-Info ""
