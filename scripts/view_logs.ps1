#!/usr/bin/env pwsh
# View and tail log files from all services

param(
    [string]$Service = $null,
    [switch]$Tail = $false,
    [int]$Lines = 50,
    [switch]$Clear = $false
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$logsDir = Join-Path $ProjectRoot "logs"

# Clear logs
if ($Clear) {
    if (Test-Path $logsDir) {
        Write-Host "Clearing all log files..." -ForegroundColor Yellow
        Remove-Item "$logsDir\*.log" -Force
        Write-Host "✓ Logs cleared" -ForegroundColor Green
    }
    exit 0
}

# Check if logs directory exists
if (-not (Test-Path $logsDir)) {
    Write-Host "No logs directory found. Start services first:" -ForegroundColor Red
    Write-Host "  .\scripts\run_services.ps1" -ForegroundColor Yellow
    exit 1
}

# Get all log files
$logFiles = Get-ChildItem -Path $logsDir -Filter "*.log"

if ($logFiles.Count -eq 0) {
    Write-Host "No log files found. Start services first:" -ForegroundColor Red
    Write-Host "  .\scripts\run_services.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " LexiQuery Service Logs" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# List available services
if (-not $Service) {
    Write-Host "Available log files:" -ForegroundColor Yellow
    foreach ($file in $logFiles) {
        $size = (Get-Item $file.FullName).Length / 1KB
        Write-Host "  • $($file.BaseName) ($([math]::Round($size, 2)) KB)" -ForegroundColor White
    }
    
    Write-Host "`nUsage:" -ForegroundColor Yellow
    Write-Host "  .\scripts\view_logs.ps1 -Service core-engine" -ForegroundColor Gray
    Write-Host "  .\scripts\view_logs.ps1 -Service core-engine -Tail" -ForegroundColor Gray
    Write-Host "  .\scripts\view_logs.ps1 -Service core-engine -Lines 100" -ForegroundColor Gray
    Write-Host "  .\scripts\view_logs.ps1 -Clear" -ForegroundColor Gray
    Write-Host ""
    exit 0
}

# Find specific service log
$logFile = Join-Path $logsDir "$Service.log"

if (-not (Test-Path $logFile)) {
    Write-Host "Log file not found: $logFile" -ForegroundColor Red
    Write-Host "`nAvailable services:" -ForegroundColor Yellow
    foreach ($file in $logFiles) {
        Write-Host "  • $($file.BaseName)" -ForegroundColor White
    }
    exit 1
}

# Tail logs (follow)
if ($Tail) {
    Write-Host "Tailing logs: $Service (Ctrl+C to stop)`n" -ForegroundColor Cyan
    Get-Content -Path $logFile -Wait -Tail $Lines
}
# Show last N lines
else {
    Write-Host "Last $Lines lines from: $Service`n" -ForegroundColor Cyan
    Get-Content -Path $logFile -Tail $Lines
    Write-Host "`n" -ForegroundColor Gray
    Write-Host "Tail logs: .\scripts\view_logs.ps1 -Service $Service -Tail" -ForegroundColor Gray
}
