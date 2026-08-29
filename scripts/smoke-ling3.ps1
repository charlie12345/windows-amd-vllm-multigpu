param(
    [ValidateSet(1, 2)]
    [int]$TensorParallelSize = 2,
    [int]$MaxModelLen = 128,
    [int]$InputLen = 8,
    [int]$OutputLen = 2,
    [bool]$UseDummyWeights = $true,
    [bool]$UseRccl = $true,
    [bool]$UseD3D12 = $true
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Pins = Get-Content (Join-Path $ProjectRoot "pins\rocm10-vllm-v0.28.0.json") |
    ConvertFrom-Json
$VllmExecutable = Join-Path $ProjectRoot ".venv-vllm\Scripts\vllm.exe"

if (-not (Test-Path $VllmExecutable)) {
    throw "Missing .venv-vllm. Run scripts\bootstrap-vllm.ps1 first."
}
if ($TensorParallelSize -eq 2) {
    & (Join-Path $PSScriptRoot 'assert-windows-amd-gpu-health.ps1') `
        -RequiredCount 2
}

$env:HIP_VISIBLE_DEVICES = if ($TensorParallelSize -eq 2) { "0,1" } else { "0" }
$env:VLLM_WORKER_MULTIPROC_METHOD = "spawn"
$env:VLLM_DISTRIBUTED_USE_SPLIT_GROUP = "0"
$env:VLLM_PLUGINS = "windows_amd_multigpu"
$env:WAVMG_ENABLE = "1"
$env:WAVMG_TRACE_COLLECTIVES = "1"
$env:WAVMG_USE_RCCL = if ($UseRccl -and $TensorParallelSize -eq 2) { "1" } else { "0" }
$env:WAVMG_USE_D3D12 = if ($UseD3D12 -and $TensorParallelSize -eq 2) { "1" } else { "0" }
$env:WAVMG_RCCL_DLL = Join-Path $ProjectRoot "build\rccl-windows\rccl.dll"
$env:WAVMG_D3D12_DLL = Join-Path $ProjectRoot "build\native\wavmg_d3d12_v1.dll"
$env:WAVMG_HIP_DLL = Join-Path $ProjectRoot "build\native\wavmg_hip_v1.dll"
$env:HF_HUB_DISABLE_XET = "1"
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
Remove-Item Env:VLLM_USE_V2_MODEL_RUNNER -ErrorAction SilentlyContinue

if ($TensorParallelSize -eq 2) {
    foreach ($Required in (
        $env:WAVMG_RCCL_DLL,
        $env:WAVMG_D3D12_DLL,
        $env:WAVMG_HIP_DLL
    )) {
        if (-not (Test-Path $Required)) {
            throw "Missing native transport: $Required"
        }
    }
}

$BenchArgs = @(
    "bench", "latency",
    "--model", $Pins.ling3_tiny_bf16_model,
    "--revision", $Pins.ling3_tiny_bf16_revision,
    "--tokenizer-revision", $Pins.ling3_tiny_bf16_revision,
    "--trust-remote-code",
    "--tensor-parallel-size", $TensorParallelSize,
    "--distributed-executor-backend", "mp",
    "--dtype", "bfloat16",
    "--max-model-len", $MaxModelLen,
    "--input-len", $InputLen,
    "--output-len", $OutputLen,
    "--batch-size", "1",
    "--gpu-memory-utilization", "0.70",
    "--enforce-eager",
    "-O0",
    "--no-enable-prefix-caching",
    "--no-async-scheduling"
)
if ($UseDummyWeights) {
    $BenchArgs += @("--load-format", "dummy")
}

Write-Host (
    "Ling 3.0 tiny smoke: TP$TensorParallelSize, " +
    "dummy_weights=$UseDummyWeights, D3D12=$UseD3D12, RCCL=$UseRccl"
) -ForegroundColor Cyan
& $VllmExecutable @BenchArgs
if ($LASTEXITCODE -ne 0) {
    throw "Ling 3.0 tiny smoke failed with exit code $LASTEXITCODE."
}
