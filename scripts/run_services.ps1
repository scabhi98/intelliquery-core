#!/usr/bin/env pwsh
# Run all Phase 2 services locally for development/testing

param(
    [switch]$Stop = $false,
    [switch]$Status = $false
)

$ErrorActionPreference = "Stop"

# Get project root
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

# Check if venv exists
if (-not (Test-Path $PythonExe)) {
    Write-Host "Virtual environment not found. Run setup first:" -ForegroundColor Red
    Write-Host "  .\scripts\setup_envs.ps1" -ForegroundColor Yellow
    exit 1
}

# Service definitions
$services = @(
    @{Name="Core Engine"; Module="services.core_engine.src.main:app"; Port=8000; Color="Cyan"}
    @{Name="Planner Agent"; Module="services.mock_agents.planner.src.main:app"; Port=9000; Color="Green"}
    @{Name="SOP Knowledge"; Module="services.mock_agents.knowledge.src.main:app_sop"; Port=9010; Color="Yellow"}
    @{Name="Error Knowledge"; Module="services.mock_agents.knowledge.src.main:app_error"; Port=9011; Color="Magenta"}
    @{Name="KQL Data Agent"; Module="services.mock_agents.data.src.main:app_kql"; Port=9020; Color="Blue"}
    @{Name="SPL Data Agent"; Module="services.mock_agents.data.src.main:app_spl"; Port=9021; Color="DarkCyan"}
    @{Name="SQL Data Agent"; Module="services.mock_agents.data.src.main:app_sql"; Port=9022; Color="DarkBlue"}
)

# Status check
if ($Status) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host " Service Status Check" -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan
    
    foreach ($svc in $services) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:$($svc.Port)/health" -Method GET -TimeoutSec 2 -ErrorAction Stop
            Write-Host "✓ $($svc.Name) (Port $($svc.Port))" -ForegroundColor Green -NoNewline
            Write-Host " - RUNNING" -ForegroundColor White
        } catch {
            Write-Host "✗ $($svc.Name) (Port $($svc.Port))" -ForegroundColor Red -NoNewline
            Write-Host " - STOPPED" -ForegroundColor Gray
        }
    }
    Write-Host ""
    exit 0
}

# Stop services
if ($Stop) {
    Write-Host "`nStopping all services..." -ForegroundColor Yellow
    
    foreach ($svc in $services) {
        $processes = Get-Process | Where-Object { $_.MainWindowTitle -like "*$($svc.Name)*" -or $_.CommandLine -like "*$($svc.Port)*" }
        foreach ($proc in $processes) {
            try {
                Stop-Process -Id $proc.Id -Force
                Write-Host "✓ Stopped: $($svc.Name)" -ForegroundColor Green
            } catch {
                Write-Host "✗ Failed to stop: $($svc.Name)" -ForegroundColor Red
            }
        }
    }
    
    Write-Host "`nAll services stopped`n" -ForegroundColor Green
    exit 0
}

# Start services
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " Starting LexiQuery Services" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Create logs directory
$logsDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
    Write-Host "Created logs directory: $logsDir`n" -ForegroundColor Gray
}

# Set PYTHONPATH to project root
$env:PYTHONPATH = $ProjectRoot

foreach ($svc in $services) {
    Write-Host "Starting: $($svc.Name) on port $($svc.Port)..." -ForegroundColor $svc.Color
    
    # Create log file path
    $serviceName = $svc.Name -replace ' ', '-' | ForEach-Object { $_.ToLower() }
    $logFile = Join-Path $logsDir "$serviceName.log"
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    # Write log header
    "=== $($svc.Name) Started at $timestamp ===" | Out-File -FilePath $logFile -Encoding UTF8
    
    # Start service in new window with log redirection
    $startCmd = "cd '$ProjectRoot'; `$env:PYTHONPATH='$ProjectRoot'; Write-Host 'Starting $($svc.Name) - Logs: $logFile' -ForegroundColor $($svc.Color); uvicorn $($svc.Module) --host 0.0.0.0 --port $($svc.Port) --reload --log-level info 2>&1 | Tee-Object -FilePath '$logFile' -Append"
    
    Start-Process pwsh -ArgumentList "-NoExit", "-Command", $startCmd `
        -WindowStyle Normal
    
    Write-Host "  Log file: $logFile" -ForegroundColor Gray
    Start-Sleep -Milliseconds 500
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " All Services Started!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Services running:" -ForegroundColor White
foreach ($svc in $services) {
    Write-Host "  • $($svc.Name): http://localhost:$($svc.Port)" -ForegroundColor $svc.Color
}

Write-Host "`nHealth check URLs:" -ForegroundColor White
foreach ($svc in $services) {
    Write-Host "  http://localhost:$($svc.Port)/health" -ForegroundColor Gray
}

Write-Host "`nTest the system:" -ForegroundColor Yellow
Write-Host "  .\scripts\test_phase2.ps1" -ForegroundColor White

Write-Host "`nCheck service status:" -ForegroundColor Yellow
Write-Host "  .\scripts\run_services.ps1 -Status" -ForegroundColor White

Write-Host "`nStop all services:" -ForegroundColor Yellow
Write-Host "  .\scripts\run_services.ps1 -Stop" -ForegroundColor White

Write-Host "`nView logs:" -ForegroundColor Yellow
Write-Host "  Log files: $logsDir" -ForegroundColor White
Write-Host "  Tail logs: Get-Content $logsDir\core-engine.log -Wait -Tail 50" -ForegroundColor Gray
Write-Host "  View all: .\scripts\view_logs.ps1" -ForegroundColor Gray

Write-Host ""
