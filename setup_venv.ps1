param(
    [string]$PythonVersion = "3.10"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"

function Test-CommandExists {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Invoke-Native {
    param(
        [string]$File,
        [string[]]$Arguments
    )

    & $File @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $File $($Arguments -join ' ')"
    }
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating isolated virtual environment: .venv"

    $created = $false
    $candidates = @()

    if (Test-CommandExists "py") {
        $candidates += ,@("py", "-$PythonVersion", "-m", "venv", ".venv")
        if ($PythonVersion -ne "3.10") {
            $candidates += ,@("py", "-3.10", "-m", "venv", ".venv")
        }
        if ($PythonVersion -ne "3.11") {
            $candidates += ,@("py", "-3.11", "-m", "venv", ".venv")
        }
    }

    if (Test-CommandExists "python") {
        $candidates += ,@("python", "-m", "venv", ".venv")
    }

    foreach ($candidate in $candidates) {
        $file = $candidate[0]
        $args = $candidate[1..($candidate.Length - 1)]
        Write-Host "Trying: $file $($args -join ' ')"
        & $file @args
        if ($LASTEXITCODE -eq 0 -and (Test-Path $VenvPython)) {
            $created = $true
            break
        }
    }

    if (-not $created) {
        throw "Could not create .venv. Install Python 3.10 or 3.11, then run this script again."
    }
}
else {
    Write-Host "Using existing .venv"
}

Invoke-Native $VenvPython @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Native $VenvPython @("-m", "pip", "install", "-r", "requirements.txt")

Write-Host ""
Write-Host "Environment is ready."
Write-Host "Run the app with:"
Write-Host "powershell -ExecutionPolicy Bypass -File .\run_app.ps1"
