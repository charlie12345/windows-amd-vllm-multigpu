#Requires -Version 5.1

[CmdletBinding()]
param(
    [switch]$SkipBenchmark
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$ProbePython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $ProbePython)) {
    throw 'Run scripts\bootstrap-nightly.ps1 first.'
}

$Probes = @(
    'probes\environment_probe.py',
    'probes\two_rank_gloo.py'
)
if (-not $SkipBenchmark) {
    $Probes += 'probes\benchmark_host_staged.py'
}

foreach ($Probe in $Probes) {
    Write-Host "Running $Probe" -ForegroundColor Cyan
    & $ProbePython (Join-Path $ProjectRoot $Probe)
    if ($LASTEXITCODE -ne 0) {
        throw "$Probe failed with exit code $LASTEXITCODE."
    }
}

