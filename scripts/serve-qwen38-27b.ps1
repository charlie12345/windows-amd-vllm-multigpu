#Requires -Version 5.1
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Carlo Pasquale and project contributors

[CmdletBinding()]
param(
    [ValidateSet('Single', 'Rccl')]
    [string]$Mode = 'Single',
    [string]$ModelPath = 'G:\AI-models\Qwen3.8-27B-AWQ-INT4',
    [string]$MtpModelPath = '',
    [ValidateRange(1, 65535)]
    [int]$Port = 8001,
    [ValidateRange(1, 64)]
    [int]$MaxNumSeqs = 32,
    [ValidateRange(512, 4096)]
    [int]$MaxNumBatchedTokens = 512,
    [ValidateRange(0.5, 0.95)]
    [double]$GpuMemoryUtilization = 0.75
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Vllm = Join-Path $ProjectRoot '.venv-vllm\Scripts\vllm.exe'
$RcclDll = Join-Path $ProjectRoot 'build\rccl-windows\rccl.dll'
foreach ($Required in ($Vllm, $ModelPath)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Missing prerequisite: $Required"
    }
}

$TensorParallelSize = if ($Mode -eq 'Rccl') { 2 } else { 1 }
if ($TensorParallelSize -eq 2) {
    & (Join-Path $PSScriptRoot 'assert-windows-amd-gpu-health.ps1') `
        -RequiredCount 2
}
if ($TensorParallelSize -eq 2 -and -not (Test-Path -LiteralPath $RcclDll)) {
    throw "Missing RCCL transport: $RcclDll"
}
if ($MtpModelPath) {
    if (-not (Test-Path -LiteralPath $MtpModelPath)) {
        throw "Missing MTP checkpoint: $MtpModelPath"
    }
    if ($MaxNumBatchedTokens -lt (2 * $MaxNumSeqs)) {
        throw 'MTP requires scheduler room for target and draft token slots.'
    }
}

$env:PYTHONUTF8 = '1'
$env:HIP_VISIBLE_DEVICES = if ($TensorParallelSize -eq 2) { '0,1' } else { '0' }
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
$env:VLLM_WORKER_MULTIPROC_METHOD = 'spawn'
$env:VLLM_DISTRIBUTED_USE_SPLIT_GROUP = '0'
$env:VLLM_PLUGINS = 'windows_amd_multigpu'
$env:VLLM_ROCM_USE_RDNA_W4A16 = '1'
$env:WAVMG_ENABLE = '1'
$env:WAVMG_USE_RCCL = if ($TensorParallelSize -eq 2) { '1' } else { '0' }
$env:WAVMG_USE_D3D12 = '0'
$env:WAVMG_TRACE_COLLECTIVES = if ($TensorParallelSize -eq 2) { '1' } else { '0' }
$env:HF_HUB_OFFLINE = '1'
$env:NCCL_DEBUG = 'WARN'
$env:NCCL_RAS_ENABLE = '0'
$env:NCCL_SHM_DISABLE = '1'
$env:NCCL_P2P_DISABLE = '1'
$env:NCCL_HOSTID = 'windows-local'
$env:NCCL_COMM_BLOCKING = '1'
$env:NCCL_ALGO = 'Ring'
$env:NCCL_PROTO = 'Simple'
if ($TensorParallelSize -eq 2) {
    $env:WAVMG_RCCL_DLL = $RcclDll
}

$ServeArgs = @(
    'serve', $ModelPath,
    '--served-model-name', 'Qwen3.8-27B-AWQ-INT4',
    '--host', '127.0.0.1',
    '--port', $Port,
    '--dtype', 'bfloat16',
    '--tensor-parallel-size', $TensorParallelSize,
    '--distributed-executor-backend', 'mp',
    '--max-model-len', '512',
    '--max-num-batched-tokens', $MaxNumBatchedTokens,
    '--max-num-seqs', $MaxNumSeqs,
    '--gpu-memory-utilization', $GpuMemoryUtilization,
    '--attention-backend', 'auto',
    '--async-scheduling',
    '--enforce-eager',
    '-O0',
    '--no-enable-prefix-caching',
    '--generation-config', 'vllm',
    '--uvicorn-log-level', 'warning',
    '--shutdown-timeout', '30'
)
if ($MtpModelPath) {
    $ServeArgs += @(
        '--speculative-config.method', 'mtp',
        '--speculative-config.model', $MtpModelPath,
        '--speculative-config.num-speculative-tokens', '1'
    )
} else {
    $ServeArgs += '--disable-log-stats'
}

$SpecLabel = if ($MtpModelPath) { 'MTP-1' } else { 'no MTP' }
Write-Host (
    "Serving Qwen3.8-27B native W4A16 on port $Port with " +
    "TP$TensorParallelSize ($Mode), $SpecLabel."
) -ForegroundColor Cyan

& $Vllm @ServeArgs
exit $LASTEXITCODE
