#Requires -Version 5.1

[CmdletBinding()]
param(
    [ValidateSet('Single', 'Rccl')]
    [string]$Mode = 'Single',
    [string]$ModelPath = 'G:\AI-models\Ling-3.0-tiny-b61f4338',
    [ValidateRange(1, 65535)]
    [int]$Port = 8001,
    [ValidateRange(1, 32)]
    [int]$MaxNumSeqs = 32
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
$GpuMemoryUtilization = if ($TensorParallelSize -eq 2) { '0.70' } else { '0.80' }
if ($TensorParallelSize -eq 2 -and -not (Test-Path -LiteralPath $RcclDll)) {
    throw "Missing RCCL transport: $RcclDll"
}

$env:PYTHONUTF8 = '1'
$env:HIP_VISIBLE_DEVICES = if ($TensorParallelSize -eq 2) { '0,1' } else { '0' }
$env:VLLM_WORKER_MULTIPROC_METHOD = 'spawn'
$env:VLLM_DISTRIBUTED_USE_SPLIT_GROUP = '0'
$env:VLLM_PLUGINS = 'windows_amd_multigpu'
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
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
if ($TensorParallelSize -eq 2) {
    $env:WAVMG_RCCL_DLL = $RcclDll
}

Write-Host (
    "Serving Ling BF16 on port $Port with TP$TensorParallelSize ($Mode)."
) -ForegroundColor Cyan

& $Vllm serve $ModelPath `
    --served-model-name Ling-3.0-tiny `
    --trust-remote-code `
    --host 127.0.0.1 `
    --port $Port `
    --dtype bfloat16 `
    --tensor-parallel-size $TensorParallelSize `
    --distributed-executor-backend mp `
    --max-model-len 512 `
    --max-num-batched-tokens 511 `
    --max-num-seqs $MaxNumSeqs `
    --gpu-memory-utilization $GpuMemoryUtilization `
    --async-scheduling `
    -O1 `
    --no-enable-prefix-caching `
    --generation-config vllm `
    --disable-log-stats `
    --uvicorn-log-level warning `
    --shutdown-timeout 30

exit $LASTEXITCODE
