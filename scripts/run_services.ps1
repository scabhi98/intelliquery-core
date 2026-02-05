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
    @{Name="Planner Agent"; Module="services.planner.src.main:app"; Port=9000; Color="Green"}
    @{Name="SOP Knowledge"; Module="services.mock_agents.knowledge.src.main:app_sop"; Port=9010; Color="Yellow"}
    @{Name="Error Knowledge"; Module="services.mock_agents.knowledge.src.main:app_error"; Port=9011; Color="Magenta"}
    @{Name="KQL Data Agent"; Module="services.mock_agents.data.src.main:app_kql"; Port=9020; Color="Blue"}
    # @{Name="SPL Data Agent"; Module="services.mock_agents.data.src.main:app_spl"; Port=9021; Color="DarkCyan"}
    @{Name="SQL Data Agent"; Module="services.mock_agents.data.src.main:app_sql"; Port=9022; Color="DarkBlue"}
    @{Name="Protocol Interface"; Module="services.protocol_interface.src.main:app"; Port=8001; Color="DarkGreen"}
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

# Set PYTHONPATH to project root
$env:PYTHONPATH = $ProjectRoot

foreach ($svc in $services) {
    Write-Host "Starting: $($svc.Name) on port $($svc.Port)..." -ForegroundColor $svc.Color
    
    # Start service in new window
    $startCmd = "& uvicorn $($svc.Module) --port $($svc.Port) --reload"
    
    Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd '$ProjectRoot'; `$env:PYTHONPATH='$ProjectRoot'; $startCmd" `
        -WindowStyle Normal
    
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
Write-Host "  Check individual PowerShell windows for each service" -ForegroundColor White

Write-Host ""
