# PowerShell version of start.sh

# Define colors for console output
function Write-ColorText {
    param(
        [string]$Text,
        [string]$Color = "White"
    )
    Write-Host $Text -ForegroundColor $Color
}

Write-ColorText "Starting KrunchWrapper" "Green"
Write-Host "==================="
Write-ColorText "Using configuration from config\server.jsonc and config\config.jsonc" "Yellow"

# Get the directory of the script
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Define the virtual environment directory relative to the script location
$VenvDir = Join-Path $ScriptDir ".venv"

# Check if virtual environment exists
if (-not (Test-Path $VenvDir)) {
    Write-ColorText "Virtual environment not found at $VenvDir" "Yellow"
    Write-Host "Please run install.ps1 first to set up the environment."
    Write-Host "`nPress any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Function to read config values
function Read-Config {
    $pythonScript = @"
import json
import os
import sys
import re

# Read server config (webui_enabled, webui_port)
server_config = {}
try:
    server_config_path = os.path.join(r'$($ScriptDir.Replace('\', '\\'))', 'config', 'server.jsonc')
    with open(server_config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Remove comments
    content = re.sub(r'//.*', '', content)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    server_config = json.loads(content)
except Exception as e:
    pass  # use defaults

# Read main config (use_cline)
main_config = {}
try:
    main_config_path = os.path.join(r'$($ScriptDir.Replace('\', '\\'))', 'config', 'config.jsonc')
    with open(main_config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Remove comments
    content = re.sub(r'//.*', '', content)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    main_config = json.loads(content)
except Exception as e:
    pass  # use defaults

# Get values with defaults
webui_enabled = server_config.get('webui_enabled', True)
webui_port = server_config.get('webui_port', 5173)
use_cline = main_config.get('system_prompt', {}).get('use_cline', False)

print('{}|{}|{}'.format(webui_enabled, webui_port, use_cline))
"@

    try {
        $configValues = python -c $pythonScript
        return $configValues -split '\|'
    } catch {
        # Return defaults if python fails
        return @("True", "5173", "False")
    }
}

# Get config values
$ConfigValues = Read-Config
$WebuiEnabled = $ConfigValues[0]
$WebuiPort = $ConfigValues[1]
$UseCline = $ConfigValues[2]

# Create a new PowerShell session with activated virtual environment
$startScript = {
    param($ScriptDir, $VenvDir, $WebuiEnabled, $WebuiPort, $UseCline)

    Set-Location $ScriptDir

    # Activate virtual environment (Windows)
    $activateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        Write-Host "Activating virtual environment..." -ForegroundColor Yellow
        try {
            & $activateScript
            if ($LASTEXITCODE -ne 0) {
                throw "Activation script returned non-zero exit code: $LASTEXITCODE"
            }
        } catch {
            Write-Host "Failed to activate using PowerShell script: $($_.Exception.Message)" -ForegroundColor Red
            Write-Host "Trying batch file activation as fallback..." -ForegroundColor Yellow

            # Try batch file activation as fallback
            $activateBat = Join-Path $VenvDir "Scripts\activate.bat"
            if (Test-Path $activateBat) {
                cmd /c "`"$activateBat`" && set" | Where-Object { $_ -match '^([^=]+)=(.*)$' } | ForEach-Object {
                    $name = $matches[1]
                    $value = $matches[2]
                    Set-Item -Path "env:$name" -Value $value
                }
            } else {
                Write-Host "Activation script not found at $activateScript" -ForegroundColor Red
                Write-Host "Please run install.ps1 first to set up the environment properly." -ForegroundColor Yellow
                Write-Host "`nPress any key to exit..."
                $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
                exit 1
            }
        }
    } else {
        # Try batch file activation as fallback
        $activateBat = Join-Path $VenvDir "Scripts\activate.bat"
        if (Test-Path $activateBat) {
            Write-Host "PowerShell activation script not found, using batch file..." -ForegroundColor Yellow
            cmd /c "`"$activateBat`" && set" | Where-Object { $_ -match '^([^=]+)=(.*)$' } | ForEach-Object {
                $name = $matches[1]
                $value = $matches[2]
                Set-Item -Path "env:$name" -Value $value
            }
        } else {
            Write-Host "Activation script not found at $activateScript" -ForegroundColor Red
            Write-Host "Please run install.ps1 first to set up the environment properly." -ForegroundColor Yellow
            Write-Host "`nPress any key to exit..."
            $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
            exit 1
        }
    }

    # Verify virtual environment activation worked
    Write-Host "Verifying virtual environment activation..." -ForegroundColor Yellow
    try {
        $pythonPath = (python -c "import sys; print(sys.executable)" 2>&1)
        if ($LASTEXITCODE -eq 0 -and $pythonPath -like "*$VenvDir*") {
            Write-Host "Virtual environment activated successfully!" -ForegroundColor Green
            Write-Host "   Python path: $pythonPath" -ForegroundColor Cyan
        } else {
            throw "Python not using virtual environment. Path: $pythonPath"
        }
    } catch {
        Write-Host "Virtual environment activation verification failed: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Please run install.ps1 first to set up the environment properly." -ForegroundColor Yellow
        Write-Host "`nPress any key to exit..."
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        exit 1
    }

    # Verify critical dependencies are available
    Write-Host "Verifying dependencies..." -ForegroundColor Yellow
    try {
        $uvicornCheck = python -c "import uvicorn; print('uvicorn OK')" 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "uvicorn import failed: $uvicornCheck"
        }
        $fastapiCheck = python -c "import fastapi; print('fastapi OK')" 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "fastapi import failed: $fastapiCheck"
        }
        $aiohttpCheck = python -c "import aiohttp; print('aiohttp OK')" 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "aiohttp import failed: $aiohttpCheck"
        }
        $tiktokenCheck = python -c "import tiktoken; print('tiktoken OK')" 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "tiktoken import failed: $tiktokenCheck"
        }
        $pydanticCheck = python -c "import pydantic; print('pydantic OK')" 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "pydantic import failed: $pydanticCheck"
        }
        Write-Host "All critical dependencies are available!" -ForegroundColor Green
    } catch {
        Write-Host "Dependency verification failed: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Installing missing dependencies..." -ForegroundColor Yellow
        try {
            pip install -r requirements.txt
            if ($LASTEXITCODE -ne 0) {
                throw "pip install failed"
            }
            Write-Host "Dependencies installed successfully!" -ForegroundColor Green
        } catch {
            Write-Host "Failed to install dependencies. Please run install.ps1 first." -ForegroundColor Red
            Write-Host "`nPress any key to exit..."
            $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
            exit 1
        }
    }

    Write-Host "KrunchWrapper virtual environment activated!" -ForegroundColor Green

    # Define the server script path
    $ServerScript = Join-Path $ScriptDir "server\run_server.py"

    # Check if server script exists
    if (-not (Test-Path $ServerScript)) {
        Write-Host "Server script not found at $ServerScript" -ForegroundColor Yellow
        Write-Host "`nPress any key to exit..."
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        exit 1
    }

    Write-Host "Starting server..." -ForegroundColor Green

    # Start the server without WindowStyle Hidden to see any errors
    $ServerProcess = Start-Process -FilePath "python" -ArgumentList "`"$ServerScript`"" -PassThru

    # Handle different modes based on configuration
    if ($WebuiEnabled -eq "True" -and $UseCline -eq "False" -and (Test-Path (Join-Path $ScriptDir "webui")) -and (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
        # WebUI mode - wait for server to be ready before starting WebUI
        Write-Host "Waiting for server to be ready..." -ForegroundColor Yellow
        Start-Sleep -Seconds 5

        # Check if server is running
        if ($ServerProcess.HasExited) {
            Write-Host "Server failed to start - Exit code: $($ServerProcess.ExitCode)" -ForegroundColor Red
            Write-Host "`nPress any key to exit..."
            $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
            exit 1
        }

        Write-Host "Starting WebUI on port $WebuiPort..." -ForegroundColor Green
        $WebuiDir = Join-Path $ScriptDir "webui"
        Set-Location $WebuiDir
        $WebuiProcess = Start-Process -FilePath "npm.cmd" -ArgumentList "run", "dev" -PassThru -WorkingDirectory $WebuiDir
        Set-Location $ScriptDir

        # Wait a moment for webui to start, then open browser
        Start-Sleep -Seconds 3
        try {
            Start-Process "http://localhost:$WebuiPort"
        } catch {
            Write-Host "Could not open browser automatically. Please visit http://localhost:$WebuiPort" -ForegroundColor Yellow
        }

        Write-Host "WebUI started! Browser should open automatically..." -ForegroundColor Green
        Write-Host "WebUI URL: http://localhost:$WebuiPort" -ForegroundColor Yellow

        # Function to cleanup on exit
        $cleanup = {
            Write-Host "Shutting down WebUI and server..." -ForegroundColor Yellow
            try {
                if ($WebuiProcess -and -not $WebuiProcess.HasExited) {
                    Stop-Process -Id $WebuiProcess.Id -Force -ErrorAction SilentlyContinue
                }
            } catch { }
            try {
                if ($ServerProcess -and -not $ServerProcess.HasExited) {
                    Stop-Process -Id $ServerProcess.Id -Force -ErrorAction SilentlyContinue
                }
            } catch { }
        }

        # Register cleanup on Ctrl+C
        try {
            [Console]::TreatControlCAsInput = $false
            Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action $cleanup
        } catch { }

        # Wait for either process to exit
        while (-not $ServerProcess.HasExited -and -not $WebuiProcess.HasExited) {
            Start-Sleep -Seconds 1
        }

    } elseif ($UseCline -eq "True") {
        Write-Host "WebUI disabled (use_cline: true in config)" -ForegroundColor Yellow
        Write-Host "Server running in Cline mode" -ForegroundColor Green

        # Function to cleanup on exit
        $cleanup = {
            Write-Host "Shutting down server..." -ForegroundColor Yellow
            try {
                if ($ServerProcess -and -not $ServerProcess.HasExited) {
                    Stop-Process -Id $ServerProcess.Id -Force -ErrorAction SilentlyContinue
                }
            } catch { }
        }

        # Register cleanup on Ctrl+C
        try {
            [Console]::TreatControlCAsInput = $false
            Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action $cleanup
        } catch { }

        # Wait for server process
        $ServerProcess.WaitForExit()

    } else {
        Write-Host "WebUI not started" -ForegroundColor Yellow

        # Function to cleanup on exit
        $cleanup = {
            Write-Host "Shutting down server..." -ForegroundColor Yellow
            try {
                if ($ServerProcess -and -not $ServerProcess.HasExited) {
                    Stop-Process -Id $ServerProcess.Id -Force -ErrorAction SilentlyContinue
                }
            } catch { }
        }

        # Register cleanup on Ctrl+C
        try {
            [Console]::TreatControlCAsInput = $false
            Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action $cleanup
        } catch { }

        # Wait for server process
        $ServerProcess.WaitForExit()
    }

    Write-Host "Press any key to close this window..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

# Start new PowerShell window with the script
$encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes("& {$startScript} -ScriptDir '$($ScriptDir.Replace("'", "''"))' -VenvDir '$($VenvDir.Replace("'", "''"))' -WebuiEnabled '$WebuiEnabled' -WebuiPort '$WebuiPort' -UseCline '$UseCline'"))
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-EncodedCommand", $encodedCommand
