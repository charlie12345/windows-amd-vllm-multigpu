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

Write-Host 'Building the native HIP transport' -ForegroundColor Cyan
& (Join-Path $PSScriptRoot 'build-native.cmd')
if ($LASTEXITCODE -ne 0) {
    throw "Native build failed with exit code $LASTEXITCODE."
}

$Probes = @(
    'probes\environment_probe.py',
    'probes\hip_ipc_probe.py',
    'probes\d3d12_cross_process_probe.py',
    'probes\d3d12_all_reduce_probe.py',
    'probes\hip_shared_memory_probe.py',
    'probes\two_rank_gloo.py'
)
if (-not $SkipBenchmark) {
    $Probes += @(
        'probes\benchmark_d3d12_cross_adapter.py',
        'probes\benchmark_host_staged.py',
        'probes\benchmark_stream_all_reduce.py'
    )
}

foreach ($Probe in $Probes) {
    Write-Host "Running $Probe" -ForegroundColor Cyan
    & $ProbePython (Join-Path $ProjectRoot $Probe)
    if ($LASTEXITCODE -ne 0) {
        throw "$Probe failed with exit code $LASTEXITCODE."
    }
}
