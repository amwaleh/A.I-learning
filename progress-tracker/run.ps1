<#
.SYNOPSIS
    Starts the AI Learning Progress Tracker (backend + frontend).

.DESCRIPTION
    Sets up Python venv, installs dependencies, seeds the database,
    and launches both the FastAPI backend (port 8000) and React frontend (port 5173).

.PARAMETER Seed
    Re-seed the database (drops existing data). Use on first run or to reset.

.PARAMETER BackendOnly
    Start only the backend server.

.PARAMETER FrontendOnly
    Start only the frontend server.

.EXAMPLE
    .\run.ps1              # Start both servers
    .\run.ps1 -Seed        # Reset DB and start both servers
    .\run.ps1 -BackendOnly # Start only the backend
#>

param(
    [switch]$Seed,
    [switch]$BackendOnly,
    [switch]$FrontendOnly
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

Write-Host "`n🚀 AI Learning Progress Tracker" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# --- Backend Setup ---
if (-not $FrontendOnly) {
    $backendDir = Join-Path $Root "backend"
    $venvDir = Join-Path $backendDir ".venv"
    $venvPython = Join-Path $venvDir "Scripts\python.exe"
    $requirementsFile = Join-Path $backendDir "requirements.txt"

    Write-Host "`n📦 Setting up backend..." -ForegroundColor Yellow

    # Create venv if missing
    if (-not (Test-Path $venvPython)) {
        Write-Host "  Creating Python virtual environment..."
        python -m venv $venvDir
        if ($LASTEXITCODE -ne 0) { Write-Error "Failed to create venv. Is Python installed?" }
    }

    # Install dependencies
    Write-Host "  Installing Python dependencies..."
    & $venvPython -m pip install -r $requirementsFile --quiet
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to install Python dependencies." }

    # Seed database
    $dbFile = Join-Path $backendDir "learning_progress.db"
    if ($Seed -or -not (Test-Path $dbFile)) {
        Write-Host "  🌱 Seeding database..." -ForegroundColor Green
        if ($Seed -and (Test-Path $dbFile)) { Remove-Item $dbFile -Force }
        Push-Location $backendDir
        & $venvPython -m app.seed
        if ($LASTEXITCODE -ne 0) { Pop-Location; Write-Error "Database seeding failed." }
        Pop-Location
    }
}

# --- Frontend Setup ---
if (-not $BackendOnly) {
    $frontendDir = Join-Path $Root "frontend"
    $nodeModules = Join-Path $frontendDir "node_modules"

    Write-Host "`n📦 Setting up frontend..." -ForegroundColor Yellow

    if (-not (Test-Path $nodeModules)) {
        Write-Host "  Installing npm dependencies..."
        Push-Location $frontendDir
        npm install --silent
        if ($LASTEXITCODE -ne 0) { Pop-Location; Write-Error "npm install failed." }
        Pop-Location
    }
}

# --- Start Servers ---
Write-Host "`n🟢 Starting servers..." -ForegroundColor Green

$jobs = @()

if (-not $FrontendOnly) {
    $backendDir = Join-Path $Root "backend"
    $venvPython = Join-Path $backendDir ".venv\Scripts\python.exe"

    $jobs += Start-Job -Name "Backend" -ScriptBlock {
        param($py, $dir)
        Set-Location $dir
        & $py -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload 2>&1
    } -ArgumentList $venvPython, $backendDir

    Write-Host "  ✅ Backend starting on http://localhost:8000" -ForegroundColor Green
}

if (-not $BackendOnly) {
    $frontendDir = Join-Path $Root "frontend"

    $jobs += Start-Job -Name "Frontend" -ScriptBlock {
        param($dir)
        Set-Location $dir
        npm run dev 2>&1
    } -ArgumentList $frontendDir

    Write-Host "  ✅ Frontend starting on http://localhost:5173" -ForegroundColor Green
}

Write-Host "`n📋 Press Ctrl+C to stop all servers`n" -ForegroundColor Gray

# Stream output from both jobs until user cancels
try {
    while ($true) {
        foreach ($job in $jobs) {
            Receive-Job $job -ErrorAction SilentlyContinue
        }
        Start-Sleep -Milliseconds 500

        # Check if any job failed
        foreach ($job in $jobs) {
            if ($job.State -eq "Failed") {
                Write-Host "`n❌ $($job.Name) server failed!" -ForegroundColor Red
                Receive-Job $job
                throw "$($job.Name) crashed"
            }
        }
    }
}
finally {
    Write-Host "`n🛑 Shutting down servers..." -ForegroundColor Yellow
    $jobs | Stop-Job -PassThru | Remove-Job -Force
    Write-Host "Done." -ForegroundColor Green
}
