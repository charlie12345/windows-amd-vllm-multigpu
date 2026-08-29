#Requires -Version 5.1

[CmdletBinding()]
param(
    [ValidateRange(128, 4096)]
    [int]$MaxModelLen = 512,
    [ValidateRange(1, 128)]
    [int]$MaxTokens = 16,
    [ValidateRange(0.5, 0.99)]
    [double]$GpuMemoryUtilization = 0.92
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
& (Join-Path $PSScriptRoot 'assert-windows-amd-gpu-health.ps1') -RequiredCount 2
$Python = Join-Path $ProjectRoot '.venv-vllm\Scripts\python.exe'
$Resolver = Join-Path $ProjectRoot 'scripts\download-large-test-model.py'
if (-not (Test-Path -LiteralPath $Python)) {
    throw 'Run scripts\bootstrap-vllm.ps1 first.'
}
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot 'build\rccl-windows\rccl.dll'))) {
    throw 'Build RCCL with scripts\configure-rccl-windows.ps1 and scripts\build-rccl-windows.ps1 first.'
}
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot 'build\native\wavmg_d3d12_v1.dll'))) {
    throw 'Build the D3D12 transport with scripts\build-native.cmd first.'
}

$ResolverOutput = @(& $Python $Resolver --local-files-only)
if ($LASTEXITCODE -ne 0) {
    throw 'The pinned large model is not complete. Run scripts\download-large-test-model.py first.'
}
$ModelLine = $ResolverOutput | Where-Object { $_ -like 'WAVMG_MODEL_PATH=*' } |
    Select-Object -Last 1
if (-not $ModelLine) {
    throw 'Could not resolve the pinned large-model snapshot.'
}
$ModelPath = $ModelLine.Substring('WAVMG_MODEL_PATH='.Length)

$env:WAVMG_RCCL_DLL = Join-Path $ProjectRoot 'build\rccl-windows\rccl.dll'
$env:WAVMG_D3D12_DLL = Join-Path $ProjectRoot 'build\native\wavmg_d3d12_v1.dll'
$env:WAVMG_HIP_DLL = Join-Path $ProjectRoot 'build\native\wavmg_hip_v1.dll'
$env:NCCL_DEBUG = 'WARN'
$env:NCCL_RAS_ENABLE = '0'
$env:NCCL_SHM_DISABLE = '1'
$env:NCCL_P2P_DISABLE = '1'
$env:NCCL_HOSTID = 'windows-local'
$env:NCCL_COMM_BLOCKING = '1'
$env:NCCL_ALGO = 'Ring'
$env:NCCL_PROTO = 'Simple'
$env:WAVMG_TRACE_COLLECTIVES = '1'
$env:VLLM_ROCM_USE_RDNA_W4A16 = '1'

Write-Host "Running the pinned 24B AWQ W4A16 model with TP2: $ModelPath" -ForegroundColor Cyan
& $Python (Join-Path $ProjectRoot 'probes\vllm_tp_output.py') `
    --tp 2 `
    --model $ModelPath `
    --quantization awq `
    --max-model-len $MaxModelLen `
    --max-num-batched-tokens $MaxModelLen `
    --max-tokens $MaxTokens `
    --gpu-memory-utilization $GpuMemoryUtilization `
    --use-rccl `
    --use-d3d12
if ($LASTEXITCODE -ne 0) {
    throw "Large-model TP2 validation failed with exit code $LASTEXITCODE."
}
