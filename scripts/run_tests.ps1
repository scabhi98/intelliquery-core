# LexiQuery - Run Tests Script
# This script runs tests for all services or a specific service

param(
    [string]$ServiceName = $null,
    [switch]$Coverage,
    [switch]$Verbose
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
Write-Info " LexiQuery Test Suite"
Write-Info "==================================="
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

# If specific service provided, only test that one
if ($ServiceName) {
    if ($services -contains $ServiceName) {
        $services = @($ServiceName)
        Write-Info "Running tests for: $ServiceName"
    } else {
        Write-Error "Unknown service: $ServiceName"
        Write-Info "Available services: $($services -join ', ')"
        exit 1
    }
} else {
    Write-Info "Running tests for all services..."
}

Write-Info ""

# Run tests for each service
$totalTests = 0
$totalPassed = 0
$totalFailed = 0

foreach ($service in $services) {
    $servicePath = Join-Path $ProjectRoot "services\$service"
    $venvPath = Join-Path $servicePath ".venv"
    $testsPath = Join-Path $servicePath "tests"
    
    Write-Info "[$service] Running tests..."
    
    # Check if tests directory exists
    if (-not (Test-Path $testsPath)) {
        Write-Warning "[$service] No tests directory found, skipping..."
        continue
    }
    
    # Check if venv exists
    if (-not (Test-Path $venvPath)) {
        Write-Warning "[$service] Virtual environment not found. Run setup_envs.ps1 first."
        continue
    }
    
    # Build pytest command
    $pytestPath = Join-Path $venvPath "Scripts\pytest.exe"
    
    if (-not (Test-Path $pytestPath)) {
        Write-Warning "[$service] pytest not installed in virtual environment"
        continue
    }
    
    $pytestArgs = @($testsPath)
    
    if ($Coverage) {
        $pytestArgs += "--cov=$servicePath\src"
        $pytestArgs += "--cov-report=term-missing"
        $pytestArgs += "--cov-report=html:$servicePath\htmlcov"
    }
    
    if ($Verbose) {
        $pytestArgs += "-v"
    }
    
    try {
        # Run pytest
        & $pytestPath $pytestArgs
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "[$service] ✓ All tests passed"
            $totalPassed++
        } else {
            Write-Error "[$service] ✗ Some tests failed"
            $totalFailed++
        }
    } catch {
        Write-Error "[$service] ✗ Test execution failed: $_"
        $totalFailed++
    }
    
    Write-Info ""
}

Write-Info "==================================="
Write-Info "Test Summary"
Write-Info "==================================="
Write-Success "Services passed: $totalPassed"
if ($totalFailed -gt 0) {
    Write-Error "Services failed: $totalFailed"
}

if ($Coverage) {
    Write-Info ""
    Write-Info "Coverage reports generated in:"
    Write-Info "  services\<service-name>\htmlcov\index.html"
}

Write-Info ""

# Exit with error if any tests failed
if ($totalFailed -gt 0) {
    exit 1
}
