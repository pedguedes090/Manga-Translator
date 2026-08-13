param(
    [string]$ImageName = "manga-translator",
    [string]$PythonVersion = "3.11",
    [int]$Port = 7860,
    [switch]$Run
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

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

Invoke-Native "docker" @(
    "build",
    "--build-arg", "PYTHON_VERSION=$PythonVersion",
    "-t", $ImageName,
    "."
)

if ($Run) {
    Invoke-Native "docker" @(
        "run",
        "--rm",
        "-p", "${Port}:7860",
        "-e", "PORT=7860",
        $ImageName
    )
}
