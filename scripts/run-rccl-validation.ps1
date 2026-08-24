#Requires -Version 5.1

[CmdletBinding()]
param(
    [ValidateSet('Full', 'Vllm', 'Minimal')]
    [string]$FunctionProfile = 'Vllm',
    [ValidateRange(1, 64)]
    [int]$Jobs = 8,
    [ValidateRange(1, 2147483647)]
    [int]$Count = 1048576,
    [ValidateRange(1, 10000)]
    [int]$StressIterations = 100,
    [switch]$SkipBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$Probe = Join-Path $ProjectRoot 'probes\rccl_two_rank_all_reduce.py'
if (-not (Test-Path -LiteralPath $Python)) {
    throw 'Run scripts\bootstrap-nightly.ps1 first.'
}

if (-not $SkipBuild) {
    & (Join-Path $PSScriptRoot 'configure-rccl-windows.ps1') `
        -FunctionProfile $FunctionProfile
    if ($LASTEXITCODE -ne 0) {
        throw 'Configuring RCCL failed.'
    }
    & (Join-Path $PSScriptRoot 'build-rccl-windows.ps1') -Jobs $Jobs
    if ($LASTEXITCODE -ne 0) {
        throw 'Building RCCL failed.'
    }
}

$RunRoot = Join-Path $ProjectRoot (
    'logs\rccl-validation-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
)
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$env:NCCL_DEBUG = 'WARN'
$env:NCCL_RAS_ENABLE = '0'
$env:NCCL_SHM_DISABLE = '1'
$env:NCCL_P2P_DISABLE = '1'
$env:NCCL_HOSTID = 'windows-local'
$env:NCCL_COMM_BLOCKING = '1'
$env:NCCL_ALGO = 'Ring'
$env:NCCL_PROTO = 'Simple'
$env:WAVMG_RCCL_WORLD_SIZE = '2'
$env:WAVMG_RCCL_COUNT = [string]$Count
Remove-Item Env:WAVMG_RCCL_DEVICE -ErrorAction SilentlyContinue

$Cases = @(
    @('all_reduce', 'f16'),
    @('all_reduce', 'f32'),
    @('all_reduce', 'bf16'),
    @('all_gather', 'f16'),
    @('all_gather', 'bf16'),
    @('reduce_scatter', 'f32'),
    @('reduce_scatter', 'bf16'),
    @('broadcast', 'f32')
)

foreach ($Case in $Cases) {
    $Operation = $Case[0]
    $Dtype = $Case[1]
    $Name = "$Operation-$Dtype"
    Write-Host "Validating RCCL $Name" -ForegroundColor Cyan
    $env:WAVMG_RCCL_OPERATION = $Operation
    $env:WAVMG_RCCL_DTYPE = $Dtype
    $env:WAVMG_RCCL_ITERATIONS = '1'
    $env:WAVMG_RCCL_RANK_LOG_DIR = Join-Path $RunRoot $Name
    $env:WAVMG_RCCL_RESULT_FILE = Join-Path $RunRoot "$Name.json"
    & $Python $Probe
    if ($LASTEXITCODE -ne 0) {
        throw "RCCL validation failed for $Name."
    }
}

Write-Host "Stress testing FP16 AllReduce for $StressIterations iterations" -ForegroundColor Cyan
$env:WAVMG_RCCL_OPERATION = 'all_reduce'
$env:WAVMG_RCCL_DTYPE = 'f16'
$env:WAVMG_RCCL_ITERATIONS = [string]$StressIterations
$env:WAVMG_RCCL_RANK_LOG_DIR = Join-Path $RunRoot 'stress-all-reduce-f16'
$env:WAVMG_RCCL_RESULT_FILE = Join-Path $RunRoot 'stress-all-reduce-f16.json'
& $Python $Probe
if ($LASTEXITCODE -ne 0) {
    throw 'RCCL stress validation failed.'
}

Write-Host 'Validating RCCL on PyTorch HIP tensors and a non-default stream' -ForegroundColor Cyan
& $Python (Join-Path $ProjectRoot 'probes\rccl_torch_tensor_probe.py')
if ($LASTEXITCODE -ne 0) {
    throw 'RCCL PyTorch tensor validation failed.'
}

Write-Host "RCCL validation passed. Results: $RunRoot" -ForegroundColor Green
