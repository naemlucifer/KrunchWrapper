# PowerShell version of start-venv.sh

# Define colors for console output
function Write-ColorText {
    param(
        [string]$Text,
        [string]$Color = "White"
    )
    Write-Host $Text -ForegroundColor $Color
}

# Get the directory of the script
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Define the virtual environment directory relative to the script location
$VenvDir = Join-Path $ScriptDir ".venv"

# Check if virtual environment exists
if (-not (Test-Path $VenvDir)) {
    Write-ColorText "Virtual environment not found at $VenvDir" "Yellow"
    Write-Host "Please run .\install.bat first to set up the environment."
    Write-Host "`nPress any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

Write-ColorText "Activating KrunchWrapper virtual environment..." "Green"

# Create a script block for the new PowerShell session
$startScript = {
    param($ScriptDir, $VenvDir)
    
    Set-Location $ScriptDir
    
    # Activate virtual environment (Windows)
    $activateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        & $activateScript
    } else {
        # Try batch file activation as fallback
        $activateBat = Join-Path $VenvDir "Scripts\activate.bat"
        if (Test-Path $activateBat) {
            cmd /c "`"$activateBat`" && set" | Where-Object { $_ -match '^([^=]+)=(.*)$' } | ForEach-Object {
                $name = $matches[1]
                $value = $matches[2]
                Set-Item -Path "env:$name" -Value $value
            }
        } else {
            Write-Host "Activation script not found at $activateScript" -ForegroundColor Yellow
            exit 1
        }
    }
    
    Write-Host "KrunchWrapper virtual environment activated!" -ForegroundColor Green
    Write-Host "Type 'exit' to close this terminal." -ForegroundColor Yellow
}

# Start new PowerShell window with the activation script
$encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes("& {$startScript} -ScriptDir '$($ScriptDir.Replace("'", "''"))' -VenvDir '$($VenvDir.Replace("'", "''"))'"))
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-EncodedCommand", $encodedCommand