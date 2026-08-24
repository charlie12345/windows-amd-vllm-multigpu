param(
    [string]$Model = "Qwen/Qwen3-0.6B",
    [ValidateSet(1, 2)]
    [int]$TensorParallelSize = 2,
    [int]$MaxModelLen = 256,
    [int]$InputLen = 8,
    [int]$OutputLen = 1
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VllmExecutable = Join-Path $ProjectRoot ".venv-vllm\Scripts\vllm.exe"
if (-not (Test-Path $VllmExecutable)) {
    throw "Missing .venv-vllm. Run scripts\bootstrap-vllm.ps1 first."
}

$env:HIP_VISIBLE_DEVICES = "0,1"
$env:CUDA_VISIBLE_DEVICES = "0,1"
$env:VLLM_WORKER_MULTIPROC_METHOD = "spawn"
$env:VLLM_USE_V2_MODEL_RUNNER = "0"
$env:VLLM_DISTRIBUTED_USE_SPLIT_GROUP = "0"
$env:WAVMG_ENABLE = "1"
$env:VLLM_PLUGINS = "windows_amd_multigpu"
$env:HF_HUB_OFFLINE = "1"

& $VllmExecutable bench latency `
    --model $Model `
    --tensor-parallel-size $TensorParallelSize `
    --distributed-executor-backend mp `
    --dtype bfloat16 `
    --max-model-len $MaxModelLen `
    --input-len $InputLen `
    --output-len $OutputLen `
    --batch-size 1 `
    --enforce-eager `
    -O0 `
    --no-async-scheduling

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
