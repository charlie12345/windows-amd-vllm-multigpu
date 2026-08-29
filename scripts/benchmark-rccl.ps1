#Requires -Version 5.1

[CmdletBinding()]
param(
    [ValidateSet('f16', 'f32', 'bf16')]
    [string]$Dtype = 'f16',
    [ValidateRange(1, 2147483647)]
    [int]$Count = 8388608,
    [ValidateRange(1, 10000)]
    [int]$Iterations = 100,
    [ValidateRange(0, 10000)]
    [int]$WarmupIterations = 10
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw 'Run scripts\bootstrap-nightly.ps1 first.'
}
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot 'build\rccl-windows\rccl.dll'))) {
    throw 'Build native Windows RCCL first.'
}

& (Join-Path $PSScriptRoot 'assert-windows-amd-gpu-health.ps1') -RequiredCount 2

$env:NCCL_DEBUG = 'WARN'
$env:NCCL_RAS_ENABLE = '0'
$env:NCCL_SHM_DISABLE = '1'
$env:NCCL_P2P_DISABLE = '1'
$env:NCCL_HOSTID = 'windows-local'
$env:NCCL_COMM_BLOCKING = '1'
$env:NCCL_ALGO = 'Ring'
$env:NCCL_PROTO = 'Simple'
$env:WAVMG_RCCL_WORLD_SIZE = '2'
$env:WAVMG_RCCL_OPERATION = 'all_reduce'
$env:WAVMG_RCCL_DTYPE = $Dtype
$env:WAVMG_RCCL_COUNT = [string]$Count
$env:WAVMG_RCCL_ITERATIONS = [string]$Iterations
$env:WAVMG_RCCL_WARMUP_ITERATIONS = [string]$WarmupIterations
$Result = Join-Path $ProjectRoot (
    'logs\rccl-benchmark-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.json'
)
$env:WAVMG_RCCL_RESULT_FILE = $Result

& $Python (Join-Path $ProjectRoot 'probes\rccl_two_rank_all_reduce.py')
if ($LASTEXITCODE -ne 0) {
    throw 'RCCL benchmark failed correctness validation.'
}
Write-Host "RCCL benchmark passed. Result: $Result" -ForegroundColor Green
