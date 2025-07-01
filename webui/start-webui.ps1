# WebUI Startup Script for Windows
# Use local project dependencies to avoid version conflicts

Write-Host "Starting KrunchWrapper WebUI..." -ForegroundColor Green

# Check if node_modules exists
if (-not (Test-Path "node_modules")) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
}

# Use local vite to avoid version conflicts
Write-Host "Starting Vite development server..." -ForegroundColor Cyan
Write-Host "WebUI will be available at: http://localhost:5173" -ForegroundColor Yellow

# Run with local vite from node_modules
& ".\node_modules\.bin\vite.cmd" --host 0.0.0.0 --port 5173 