param(
    [string]$HostName = "",
    [int]$Port = 0,
    [switch]$Debug
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host ".venv is missing. Creating it first..."
    & powershell -ExecutionPolicy Bypass -File (Join-Path $Root "setup_venv.ps1")
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

if ($HostName) {
    $env:HOST = $HostName
}

if ($Port -gt 0) {
    $env:PORT = [string]$Port
}

if ($Debug) {
    $env:FLASK_DEBUG = "1"
}

$DisplayHost = if ($env:HOST) { $env:HOST } else { "127.0.0.1" }
$DisplayPort = if ($env:PORT) { $env:PORT } else { "5000" }

Write-Host "Starting Manga Translator..."
Write-Host "URL: http://${DisplayHost}:${DisplayPort}"

& $VenvPython app.py
