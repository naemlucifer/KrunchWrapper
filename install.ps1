# KrunchWrapper Installation Script (PowerShell)
# Compatible with Windows PowerShell 5.0+ and PowerShell Core 6.0+

param(
    [switch]$Force,
    [switch]$SkipWebUI,
    [switch]$SkipML
)

# Ensure we're running on Windows
if ($PSVersionTable.PSVersion.Major -lt 5) {
    Write-Host "[ERROR] This script requires PowerShell 5.0 or later" -ForegroundColor Red
    Write-Host "[INFO] Current version: $($PSVersionTable.PSVersion)" -ForegroundColor Yellow
    Read-Host "`nPress Enter to exit"
    exit 1
}

# Set execution policy for this session (Windows-specific)
try {
    Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force -ErrorAction SilentlyContinue
} catch {
    Write-Host "[WARNING] Could not set execution policy. Script may still work." -ForegroundColor Yellow
}

Write-Host "`n[INFO] KrunchWrapper Installation Script" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan

Write-Host "`n[INFO] System Requirements Check:" -ForegroundColor Cyan

# Function to check if a command exists
function Test-CommandExists {
    param($Command)
    try {
        $cmd = Get-Command $Command -ErrorAction SilentlyContinue
        if ($cmd) {
            return $true
        }
    }
    catch {
        # Silently handle any errors
    }
    return $false
}

# Function to get version from command output with timeout
function Get-VersionFromCommand {
    param($Command, $Args = @("--version"), $TimeoutSeconds = 10)
    try {
        # Create a job to run the command with timeout
        $job = Start-Job -ScriptBlock {
            param($cmd, $arguments)
            & $cmd $arguments 2>&1
        } -ArgumentList $Command, $Args
        
        # Wait for job completion with timeout
        $completed = Wait-Job -Job $job -Timeout $TimeoutSeconds
        
        if ($completed) {
            $output = Receive-Job -Job $job
            Remove-Job -Job $job -Force
            
            if ($output) {
                # Handle both string and array outputs
                if ($output -is [array]) {
                    $versionString = $output[0].ToString().Trim()
                } else {
                    $versionString = $output.ToString().Trim()
                }
                return $versionString
            } else {
                return "Unknown"
            }
        } else {
            # Command timed out
            Stop-Job -Job $job
            Remove-Job -Job $job -Force
            return "Timeout"
        }
    }
    catch {
        return "Unknown"
    }
}

# Check Python 3.12 (Windows typically uses 'python' not 'python3.12')
Write-Host "[DEBUG] Starting Python check..." -ForegroundColor Yellow
$pythonCmd = $null

Write-Host "[DEBUG] Checking if python command exists..." -ForegroundColor Yellow
$pythonExists = Test-CommandExists "python"
Write-Host "[DEBUG] Python exists: $pythonExists" -ForegroundColor Yellow

if ($pythonExists) {
    Write-Host "[DEBUG] Getting Python version..." -ForegroundColor Yellow
    try {
        # Direct approach without function
        $pythonVersion = python --version 2>&1
        Write-Host "[DEBUG] Raw Python version output: '$pythonVersion'" -ForegroundColor Yellow
        
        if ($pythonVersion -match "Python 3\.12\.") {
            Write-Host "[SUCCESS] * Python 3.12: [OK] Found ($pythonVersion)" -ForegroundColor Green
            $pythonCmd = "python"
            
            # Check if it's the recommended version
            if ($pythonVersion -notmatch "3\.12\.3") {
                Write-Host "[INFO] * Using Python $pythonVersion (3.12.3 was originally recommended but any 3.12.x works)" -ForegroundColor Cyan
            }
        } else {
            Write-Host "[ERROR] * Python 3.12: [X] Wrong version found ($pythonVersion)" -ForegroundColor Red
            Write-Host "[ERROR] Python 3.12.x is required. Please install Python 3.12.3 or newer." -ForegroundColor Red
            Write-Host "[INFO] Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
            Read-Host "`nPress Enter to exit"
            exit 1
        }
    } catch {
        Write-Host "[ERROR] Failed to get Python version: $($_.Exception.Message)" -ForegroundColor Red
        Read-Host "`nPress Enter to exit"
        exit 1
    }
} else {
    Write-Host "[ERROR] * Python 3.12: [X] Missing" -ForegroundColor Red
    Write-Host "[ERROR] Python 3.12.x is not installed. Please install Python 3.12.3 or newer." -ForegroundColor Red
    Write-Host "[INFO] Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    Read-Host "`nPress Enter to exit"
    exit 1
}

# Check pip
$pipExists = Test-CommandExists "pip"
if ($pipExists) {
    $pipVersion = Get-VersionFromCommand "pip"
    Write-Host "[SUCCESS] * pip: [OK] Found ($pipVersion)" -ForegroundColor Green
} else {
    Write-Host "[WARNING] * pip: [--] Not found properly, will try with python -m pip" -ForegroundColor Yellow
}

# Check Git
$gitExists = Test-CommandExists "git"
if ($gitExists) {
    $gitVersion = Get-VersionFromCommand "git"
    Write-Host "[SUCCESS] * Git: [OK] Found ($gitVersion)" -ForegroundColor Green
} else {
    Write-Host "[INFO] * Git: [--] Not found (optional)" -ForegroundColor Yellow
}

# Check Node.js
$nodeExists = Test-CommandExists "node"
if ($nodeExists) {
    $nodeVersion = Get-VersionFromCommand "node"
    Write-Host "[SUCCESS] * Node.js: [OK] Found ($nodeVersion)" -ForegroundColor Green
    $hasNode = $true
} else {
    Write-Host "[WARNING] * Node.js: [X] Missing (required for webui)" -ForegroundColor Yellow
    $hasNode = $false
}

# Check npm
$npmExists = Test-CommandExists "npm"
if ($npmExists) {
    $npmVersion = Get-VersionFromCommand "npm"
    Write-Host "[SUCCESS] * npm: [OK] Found ($npmVersion)" -ForegroundColor Green
    $hasNpm = $true
} else {
    Write-Host "[WARNING] * npm: [X] Missing (required for webui)" -ForegroundColor Yellow
    $hasNpm = $false
}

# Check curl (handle Windows PowerShell curl alias)
$curlExists = $false
try {
    # Try curl.exe specifically to avoid PowerShell Invoke-WebRequest alias
    $curlTest = Get-Command "curl.exe" -ErrorAction SilentlyContinue
    if ($curlTest) {
        $curlExists = $true
        Write-Host "[SUCCESS] * curl: [OK] Found" -ForegroundColor Green
    }
} catch {
    # Silently handle any errors
}

if (-not $curlExists) {
    $wgetExists = Test-CommandExists "wget"
    if ($wgetExists) {
        Write-Host "[SUCCESS] * wget: [OK] Found" -ForegroundColor Green
    } else {
        Write-Host "[INFO] * curl/wget: [--] Not found (not critical for Windows)" -ForegroundColor Yellow
    }
}

Write-Host "`n[INFO] Checking Python venv module..." -ForegroundColor Cyan

# Check if venv module is available (using specific python version)
try {
    $venvCheck = & $pythonCmd -c "import venv" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[SUCCESS] Python venv module is available!" -ForegroundColor Green
    } else {
        Write-Host "[WARNING] Python venv module is not available. Installing virtualenv..." -ForegroundColor Yellow
        & $pythonCmd -m pip install virtualenv
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ERROR] Failed to install virtualenv. Please check your pip installation." -ForegroundColor Red
            Read-Host "`nPress Enter to exit"
            exit 1
        }
        Write-Host "[SUCCESS] virtualenv installed successfully!" -ForegroundColor Green
    }
} catch {
    Write-Host "[ERROR] Failed to check Python venv module." -ForegroundColor Red
    Read-Host "`nPress Enter to exit"
    exit 1
}

$venvDir = ".venv"

# Check if virtual environment already exists (Windows-compatible paths)
if (Test-Path $venvDir) {
    Write-Host "`n[WARNING] Virtual environment already exists at $venvDir" -ForegroundColor Yellow
    
    if (-not $Force) {
        $useExisting = Read-Host "Do you want to use the existing environment? (y/n)"
        if ($useExisting -ne "y") {
            Write-Host "[INFO] Creating a new virtual environment..." -ForegroundColor Cyan
            try {
                Remove-Item -Recurse -Force $venvDir -ErrorAction Stop
                & $pythonCmd -m venv "$venvDir"
                if ($LASTEXITCODE -ne 0) {
                    Write-Host "[ERROR] Failed to create virtual environment." -ForegroundColor Red
                    Read-Host "`nPress Enter to exit"
                    exit 1
                }
                Write-Host "[SUCCESS] New virtual environment created!" -ForegroundColor Green
            } catch {
                Write-Host "[ERROR] Failed to remove existing virtual environment: $($_.Exception.Message)" -ForegroundColor Red
                Read-Host "`nPress Enter to exit"
                exit 1
            }
        } else {
            Write-Host "[INFO] Using existing virtual environment." -ForegroundColor Cyan
        }
    } else {
        Write-Host "[INFO] Force flag specified, using existing virtual environment." -ForegroundColor Cyan
    }
} else {
    Write-Host "`n[INFO] Creating a new virtual environment..." -ForegroundColor Cyan
    & $pythonCmd -m venv "$venvDir"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create virtual environment." -ForegroundColor Red
        Read-Host "`nPress Enter to exit"
        exit 1
    }
    Write-Host "[SUCCESS] Virtual environment created successfully!" -ForegroundColor Green
}

# Activate the virtual environment (Windows-specific)
Write-Host "`n[INFO] Activating virtual environment..." -ForegroundColor Cyan

$activateScript = Join-Path $venvDir "Scripts" | Join-Path -ChildPath "Activate.ps1"
$activateBat = Join-Path $venvDir "Scripts" | Join-Path -ChildPath "activate.bat"

# Try PowerShell activation first, fall back to batch file
if (Test-Path $activateScript) {
    try {
        & $activateScript
        Write-Host "[SUCCESS] Virtual environment activated!" -ForegroundColor Green
    } catch {
        Write-Host "[WARNING] PowerShell activation failed, trying batch activation..." -ForegroundColor Yellow
        if (Test-Path $activateBat) {
            # For Windows, we'll continue without explicit activation
            # The python commands will still work within the venv context
            Write-Host "[INFO] Using virtual environment implicitly" -ForegroundColor Cyan
        } else {
            Write-Host "[ERROR] No activation script found" -ForegroundColor Red
            Read-Host "`nPress Enter to exit"
            exit 1
        }
    }
} elseif (Test-Path $activateBat) {
    Write-Host "[INFO] Using batch activation (PowerShell script not found)" -ForegroundColor Cyan
    # We'll use the venv python directly instead of trying to activate
} else {
    Write-Host "[ERROR] No activation script found at $activateScript or $activateBat" -ForegroundColor Red
    Read-Host "`nPress Enter to exit"
    exit 1
}

# Set python path to use venv python directly (Windows-compatible approach)
$venvPython = Join-Path $venvDir "Scripts" | Join-Path -ChildPath "python.exe"
$venvPip = Join-Path $venvDir "Scripts" | Join-Path -ChildPath "pip.exe"

# Validate that the executables exist
if (-not (Test-Path $venvPip)) {
    Write-Host "[ERROR] Pip executable not found in virtual environment at $venvPip" -ForegroundColor Red
    Read-Host "`nPress Enter to exit"
    exit 1
}

if (-not (Test-Path $venvPython)) {
    Write-Host "[ERROR] Python executable not found in virtual environment" -ForegroundColor Red
    Read-Host "`nPress Enter to exit"
    exit 1
}

# Install requirements using venv python directly (Windows-compatible)
if (Test-Path "requirements.txt") {
    Write-Host "`n[INFO] Installing requirements from requirements.txt..." -ForegroundColor Cyan
    Write-Host "[INFO] Upgrading pip first..." -ForegroundColor Cyan
    
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARNING] Failed to upgrade pip, but continuing with installation..." -ForegroundColor Yellow
    } else {
        Write-Host "[SUCCESS] pip upgraded successfully!" -ForegroundColor Green
    }
    
    Write-Host "[INFO] Installing Python packages..." -ForegroundColor Cyan
    & $venvPip install -r requirements.txt
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install some requirements. Please check the error messages above." -ForegroundColor Red
        Write-Host "[ERROR] Common solutions:" -ForegroundColor Red
        Write-Host "[ERROR] 1. Check your internet connection" -ForegroundColor Red
        Write-Host "[ERROR] 2. Update pip: python -m pip install --upgrade pip" -ForegroundColor Red
        Write-Host "[ERROR] 3. Try installing packages individually" -ForegroundColor Red
        Read-Host "`nPress Enter to exit"
        exit 1
    }
    Write-Host "[SUCCESS] Requirements installed successfully!" -ForegroundColor Green
    
    # Verify critical dependencies are installed and working
    Write-Host "`n[INFO] Verifying critical dependencies..." -ForegroundColor Cyan
    $verificationFailed = $false
    
    try {
        & $venvPython -c "import uvicorn; print('✅ uvicorn: OK')"
        if ($LASTEXITCODE -ne 0) { 
            Write-Host "[ERROR] uvicorn verification failed" -ForegroundColor Red
            $verificationFailed = $true 
        }
    } catch {
        Write-Host "[ERROR] uvicorn verification failed: $($_.Exception.Message)" -ForegroundColor Red
        $verificationFailed = $true
    }
    
    try {
        & $venvPython -c "import fastapi; print('✅ fastapi: OK')"
        if ($LASTEXITCODE -ne 0) { 
            Write-Host "[ERROR] fastapi verification failed" -ForegroundColor Red
            $verificationFailed = $true 
        }
    } catch {
        Write-Host "[ERROR] fastapi verification failed: $($_.Exception.Message)" -ForegroundColor Red
        $verificationFailed = $true
    }
    
    try {
        & $venvPython -c "import aiohttp; print('✅ aiohttp: OK')"
        if ($LASTEXITCODE -ne 0) { 
            Write-Host "[ERROR] aiohttp verification failed" -ForegroundColor Red
            $verificationFailed = $true 
        }
    } catch {
        Write-Host "[ERROR] aiohttp verification failed: $($_.Exception.Message)" -ForegroundColor Red
        $verificationFailed = $true
    }
    
    try {
        & $venvPython -c "import tiktoken; print('✅ tiktoken: OK')"
        if ($LASTEXITCODE -ne 0) { 
            Write-Host "[ERROR] tiktoken verification failed" -ForegroundColor Red
            $verificationFailed = $true 
        }
    } catch {
        Write-Host "[ERROR] tiktoken verification failed: $($_.Exception.Message)" -ForegroundColor Red
        $verificationFailed = $true
    }
    
    try {
        & $venvPython -c "import pydantic; print('✅ pydantic: OK')"
        if ($LASTEXITCODE -ne 0) { 
            Write-Host "[ERROR] pydantic verification failed" -ForegroundColor Red
            $verificationFailed = $true 
        }
    } catch {
        Write-Host "[ERROR] pydantic verification failed: $($_.Exception.Message)" -ForegroundColor Red
        $verificationFailed = $true
    }
    
    if ($verificationFailed) {
        Write-Host "`n[ERROR] Critical dependency verification failed!" -ForegroundColor Red
        Write-Host "[INFO] Attempting to reinstall critical packages..." -ForegroundColor Yellow
        
        & $venvPip install --force-reinstall uvicorn fastapi aiohttp tiktoken pydantic
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ERROR] Failed to reinstall critical packages." -ForegroundColor Red
            Read-Host "`nPress Enter to exit"
            exit 1
        }
        
        # Test again
        & $venvPython -c "import uvicorn, fastapi, aiohttp, tiktoken, pydantic; print('✅ Critical dependencies reinstalled successfully!')"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ERROR] Critical dependencies still not working after reinstall." -ForegroundColor Red
            Read-Host "`nPress Enter to exit"
            exit 1
        }
    }
    
    Write-Host "[SUCCESS] All critical dependencies verified!" -ForegroundColor Green
} else {
    Write-Host "[ERROR] requirements.txt not found!" -ForegroundColor Red
    Write-Host "[ERROR] Please ensure you're running this script from the KrunchWrapper directory." -ForegroundColor Red
    Read-Host "`nPress Enter to exit"
    exit 1
}

# Install webui dependencies
if (-not $SkipWebUI) {
    Write-Host "`n[INFO] Installing WebUI Dependencies" -ForegroundColor Cyan
    Write-Host "====================================" -ForegroundColor Cyan
    
    $webuiInstalled = $false

    if (-not $hasNode -or -not $hasNpm) {
        Write-Host "[WARNING] Node.js and/or npm not found." -ForegroundColor Yellow
        Write-Host "[INFO] The webui requires Node.js and npm to build and run." -ForegroundColor Yellow
        
        $installNode = Read-Host "`nWould you like to automatically install Node.js? (y/n)"
        
        if ($installNode -eq "y") {
            Write-Host "[INFO] Installing Node.js..." -ForegroundColor Cyan
            
            $nodeInstalled = $false
            
            # Try winget first (Windows Package Manager)
            if (Test-CommandExists "winget") {
                Write-Host "[INFO] Installing Node.js via winget..." -ForegroundColor Cyan
                try {
                    winget install -e --id OpenJS.NodeJS --silent --accept-source-agreements --accept-package-agreements
                    if ($LASTEXITCODE -eq 0) {
                        Write-Host "[SUCCESS] Node.js installed via winget!" -ForegroundColor Green
                        $nodeInstalled = $true
                    }
                } catch {
                    Write-Host "[WARNING] winget installation failed, trying next method..." -ForegroundColor Yellow
                }
            }
            
            # Try Chocolatey if winget failed
            if (-not $nodeInstalled -and (Test-CommandExists "choco")) {
                Write-Host "[INFO] Installing Node.js via Chocolatey..." -ForegroundColor Cyan
                try {
                    choco install nodejs -y
                    if ($LASTEXITCODE -eq 0) {
                        Write-Host "[SUCCESS] Node.js installed via Chocolatey!" -ForegroundColor Green
                        $nodeInstalled = $true
                    }
                } catch {
                    Write-Host "[WARNING] Chocolatey installation failed, trying next method..." -ForegroundColor Yellow
                }
            }
            
            # Try Scoop if other methods failed
            if (-not $nodeInstalled -and (Test-CommandExists "scoop")) {
                Write-Host "[INFO] Installing Node.js via Scoop..." -ForegroundColor Cyan
                try {
                    scoop install nodejs
                    if ($LASTEXITCODE -eq 0) {
                        Write-Host "[SUCCESS] Node.js installed via Scoop!" -ForegroundColor Green
                        $nodeInstalled = $true
                    }
                } catch {
                    Write-Host "[WARNING] Scoop installation failed." -ForegroundColor Yellow
                }
            }
            
            # If no package manager worked, suggest manual installation
            if (-not $nodeInstalled) {
                Write-Host "[WARNING] No package manager found or installation failed." -ForegroundColor Yellow
                Write-Host "[INFO] Please install Node.js manually from: https://nodejs.org/" -ForegroundColor Yellow
                Write-Host "[INFO] Recommended: Download and run the Windows Installer (.msi)" -ForegroundColor Yellow
                Write-Host "[WARNING] Skipping webui installation for now." -ForegroundColor Yellow
                $webuiInstalled = $false
            } else {
                # Refresh environment variables
                $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
                
                # Re-check for Node.js after installation
                Start-Sleep -Seconds 2
                $hasNode = Test-CommandExists "node"
                $hasNpm = Test-CommandExists "npm"
                
                if ($hasNode -and $hasNpm) {
                    Write-Host "[SUCCESS] Node.js and npm are now available!" -ForegroundColor Green
                } else {
                    Write-Host "[WARNING] Node.js may need a system restart to be fully available." -ForegroundColor Yellow
                    Write-Host "[INFO] Try restarting your terminal or computer if Node.js doesn't work immediately." -ForegroundColor Yellow
                }
            }
        } else {
            Write-Host "[INFO] Node.js installation declined." -ForegroundColor Yellow
            Write-Host "[WARNING] Skipping webui installation for now." -ForegroundColor Yellow
            Write-Host "[INFO] After installing Node.js manually, you can install webui dependencies by running:" -ForegroundColor Yellow
            Write-Host "[INFO]   Set-Location webui" -ForegroundColor Yellow
            Write-Host "[INFO]   npm install" -ForegroundColor Yellow
            $webuiInstalled = $false
                 }
    }
    
    # Install webui dependencies if Node.js is available
    if ($hasNode -and $hasNpm -and (Test-Path "webui")) {
        Write-Host "[INFO] Installing webui dependencies..." -ForegroundColor Cyan
        Push-Location "webui"
        
        npm install
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[SUCCESS] WebUI dependencies installed successfully!" -ForegroundColor Green
            $webuiInstalled = $true
        } else {
            Write-Host "[ERROR] Failed to install webui dependencies. Please check the error messages above." -ForegroundColor Red
            Write-Host "[INFO] You can try installing manually with: Set-Location webui; npm install" -ForegroundColor Yellow
            $webuiInstalled = $false
        }
        
        Pop-Location
    } elseif ($hasNode -and $hasNpm) {
        Write-Host "[WARNING] webui directory not found. Skipping webui installation." -ForegroundColor Yellow
        $webuiInstalled = $false
    } else {
        # Node.js/npm not available, webui already marked as not installed above
        $webuiInstalled = $false
    }
} else {
    Write-Host "`n[INFO] Skipping WebUI installation (--SkipWebUI flag specified)" -ForegroundColor Yellow
    $webuiInstalled = $false
}

# Optional: Install advanced pattern detection dependencies
if (-not $SkipML) {
    Write-Host "`n[INFO] Optional: Enhanced Pattern Detection" -ForegroundColor Cyan
    Write-Host "=========================================" -ForegroundColor Cyan
    Write-Host "[INFO] Advanced pattern detection libraries provide superior intelligence" -ForegroundColor Cyan
    Write-Host "[INFO] for compression analysis. They're optional but recommended for optimal performance." -ForegroundColor Cyan

    $installML = Read-Host "`nWould you like to install enhanced pattern detection libraries? (y/n)"

    if ($installML -eq "y") {
        Write-Host "[INFO] Installing enhanced pattern detection libraries..." -ForegroundColor Cyan
        
        # Install spaCy and download English model (using venv python)
        Write-Host "[INFO] Installing spaCy..." -ForegroundColor Cyan
        & $venvPip install "spacy>=3.4.0"
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[SUCCESS] spaCy installed successfully!" -ForegroundColor Green
            Write-Host "[INFO] Downloading spaCy English model..." -ForegroundColor Cyan
            & $venvPython -m spacy download en_core_web_sm
            if ($LASTEXITCODE -eq 0) {
                Write-Host "[SUCCESS] spaCy English model downloaded successfully!" -ForegroundColor Green
            } else {
                Write-Host "[WARNING] spaCy installed but English model download failed." -ForegroundColor Yellow
                Write-Host "[INFO] You can try downloading it later with: `"$venvPython`" -m spacy download en_core_web_sm" -ForegroundColor Yellow
            }
        } else {
            Write-Host "[ERROR] Failed to install spaCy." -ForegroundColor Red
        }
        
        # Install NetworkX for graph analysis (using venv pip)
        Write-Host "[INFO] Installing NetworkX for graph analysis..." -ForegroundColor Cyan
        & $venvPip install "networkx>=2.8.0"
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[SUCCESS] NetworkX installed successfully!" -ForegroundColor Green
        } else {
            Write-Host "[ERROR] Failed to install NetworkX." -ForegroundColor Red
        }
        
        # Install sentence-transformers for semantic embeddings (using venv pip)
        Write-Host "[INFO] Installing Sentence Transformers for semantic embeddings..." -ForegroundColor Cyan
        & $venvPip install "sentence-transformers>=2.2.0"
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[SUCCESS] Sentence Transformers installed successfully!" -ForegroundColor Green
        } else {
            Write-Host "[ERROR] Failed to install Sentence Transformers." -ForegroundColor Red
        }
        
        Write-Host "[SUCCESS] Enhanced pattern detection setup completed!" -ForegroundColor Green
        $mlInstalled = $true
    } else {
        Write-Host "[INFO] Skipping enhanced pattern detection libraries." -ForegroundColor Yellow
        Write-Host "[INFO] You can install them later with:" -ForegroundColor Yellow
        Write-Host "[INFO]   `"$venvPip`" install spacy networkx sentence-transformers" -ForegroundColor Yellow
        Write-Host "[INFO]   `"$venvPython`" -m spacy download en_core_web_sm" -ForegroundColor Yellow
        $mlInstalled = $false
    }
} else {
    Write-Host "`n[INFO] Skipping enhanced pattern detection (--SkipML flag specified)" -ForegroundColor Yellow
    $mlInstalled = $false
}

# Installation Summary
Write-Host "`n[SUCCESS] Installation completed successfully!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green

Write-Host "`n[INFO] Summary:" -ForegroundColor Cyan
Write-Host "[SUCCESS] * Virtual environment: [OK] Created at $venvDir" -ForegroundColor Green
Write-Host "[SUCCESS] * Python packages: [OK] Installed from requirements.txt" -ForegroundColor Green

# Show webui installation status
if ($hasNode -and $hasNpm) {
    if (Test-Path "webui") {
        if ($webuiInstalled) {
            Write-Host "[SUCCESS] * WebUI dependencies: [OK] Installed" -ForegroundColor Green
        } else {
            Write-Host "[WARNING] * WebUI dependencies: [--] Installation failed" -ForegroundColor Yellow
        }
    } else {
        Write-Host "[WARNING] * WebUI dependencies: [--] Skipped (webui directory not found)" -ForegroundColor Yellow
    }
} elseif (-not $hasNode) {
    Write-Host "[WARNING] * WebUI dependencies: [--] Skipped (Node.js not found)" -ForegroundColor Yellow
} else {
    Write-Host "[WARNING] * WebUI dependencies: [--] Skipped (npm not found)" -ForegroundColor Yellow
}

if ($mlInstalled) {
    Write-Host "[SUCCESS] * Enhanced pattern detection: [OK] Installed" -ForegroundColor Green
} else {
    Write-Host "[INFO] * Enhanced pattern detection: [--] Skipped" -ForegroundColor Yellow
}

Write-Host "[SUCCESS] * Language detection: [OK] Content-agnostic dynamic compression enabled" -ForegroundColor Green

Write-Host "`n[INFO] Next steps:" -ForegroundColor Cyan
Write-Host "===============" -ForegroundColor Cyan
Write-Host "[INFO] 1. Start the server using: .\start.ps1" -ForegroundColor Cyan
Write-Host "[INFO] 2. Or manually activate the environment:" -ForegroundColor Cyan
Write-Host "[INFO]    PowerShell: `"$activateScript`"" -ForegroundColor Cyan
Write-Host "[INFO]    Command Prompt: `"$activateBat`"" -ForegroundColor Cyan

if ($hasNode -and $hasNpm) {
    if (Test-Path "webui") {
        if ($webuiInstalled) {
            Write-Host "[INFO] 3. The WebUI will start automatically with the server" -ForegroundColor Cyan
            Write-Host "[INFO] 4. To deactivate later: deactivate" -ForegroundColor Cyan
        } else {
            Write-Host "[WARNING] 3. Fix WebUI installation: Set-Location webui; npm install" -ForegroundColor Yellow
            Write-Host "[INFO] 4. To deactivate later: deactivate" -ForegroundColor Cyan
        }
    } else {
        Write-Host "[INFO] 3. To deactivate later: deactivate" -ForegroundColor Cyan
    }
} else {
    Write-Host "[WARNING] 3. Install Node.js from https://nodejs.org/ to use the WebUI" -ForegroundColor Yellow
    Write-Host "[WARNING] 4. Then run: Set-Location webui; npm install" -ForegroundColor Yellow
    Write-Host "[INFO] 5. To deactivate later: deactivate" -ForegroundColor Cyan
}

Write-Host "`n[INFO] Note: KrunchWrapper now uses content-agnostic dynamic compression." -ForegroundColor Cyan
Write-Host "[INFO] No additional language detection setup required!" -ForegroundColor Cyan

# Show any important warnings
if (-not $hasNode) {
    Write-Host "`n[WARNING] =============================================================" -ForegroundColor Yellow
    Write-Host "[WARNING] Node.js is missing - WebUI functionality will be unavailable" -ForegroundColor Yellow
    Write-Host "[WARNING] Install from: https://nodejs.org/" -ForegroundColor Yellow
    Write-Host "[WARNING] =============================================================" -ForegroundColor Yellow
}

Write-Host "`n[INFO] Installation complete! Press Enter to exit..." -ForegroundColor Cyan
Read-Host 