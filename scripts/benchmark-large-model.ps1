#Requires -Version 5.1

[CmdletBinding()]
param(
    [ValidateSet('Hybrid', 'Rccl', 'Both')]
    [string]$Mode = 'Both',
    [ValidateRange(1, 4096)]
    [int]$InputLen = 32,
    [ValidateRange(1, 512)]
    [int]$OutputLen = 16,
    [ValidateRange(1, 32)]
    [int]$BatchSize = 1,
    [ValidateRange(0, 100)]
    [int]$WarmupIterations = 3,
    [ValidateRange(1, 1000)]
    [int]$Iterations = 10,
    [ValidateRange(128, 4096)]
    [int]$MaxModelLen = 512,
    [ValidateRange(128, 32768)]
    [int]$MaxNumBatchedTokens = 512,
    [ValidateRange(0.5, 0.99)]
    [double]$GpuMemoryUtilization = 0.92
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $ProjectRoot '.venv-vllm\Scripts\python.exe'
$Vllm = Join-Path $ProjectRoot '.venv-vllm\Scripts\vllm.exe'
$Resolver = Join-Path $ProjectRoot 'scripts\download-large-test-model.py'
foreach ($Required in ($Python, $Vllm, $Resolver)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Missing prerequisite: $Required"
    }
}
if (($InputLen + $OutputLen) -gt $MaxModelLen) {
    throw 'InputLen plus OutputLen cannot exceed MaxModelLen.'
}
if (($InputLen * $BatchSize) -gt $MaxNumBatchedTokens) {
    throw 'InputLen times BatchSize cannot exceed MaxNumBatchedTokens.'
}

$ResolverOutput = @(& $Python $Resolver --local-files-only)
if ($LASTEXITCODE -ne 0) {
    throw 'Download the pinned model with scripts\download-large-test-model.py first.'
}
$ModelLine = $ResolverOutput | Where-Object { $_ -like 'WAVMG_MODEL_PATH=*' } |
    Select-Object -Last 1
if (-not $ModelLine) {
    throw 'Could not resolve the pinned large-model snapshot.'
}
$ModelPath = $ModelLine.Substring('WAVMG_MODEL_PATH='.Length)

$RcclDll = Join-Path $ProjectRoot 'build\rccl-windows\rccl.dll'
$D3D12Dll = Join-Path $ProjectRoot 'build\native\wavmg_d3d12_v1.dll'
foreach ($Required in ($RcclDll, $D3D12Dll)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Missing native transport: $Required"
    }
}

$env:HIP_VISIBLE_DEVICES = '0,1'
$env:CUDA_VISIBLE_DEVICES = '0,1'
$env:VLLM_WORKER_MULTIPROC_METHOD = 'spawn'
$env:VLLM_USE_V2_MODEL_RUNNER = '0'
$env:VLLM_DISTRIBUTED_USE_SPLIT_GROUP = '0'
$env:VLLM_PLUGINS = 'windows_amd_multigpu'
$env:VLLM_ROCM_USE_RDNA_W4A16 = '1'
$env:WAVMG_ENABLE = '1'
$env:WAVMG_USE_RCCL = '1'
$env:WAVMG_RCCL_DLL = $RcclDll
$env:WAVMG_D3D12_DLL = $D3D12Dll
$env:WAVMG_TRACE_COLLECTIVES = '1'
$env:HF_HUB_OFFLINE = '1'
$env:NCCL_DEBUG = 'WARN'
$env:NCCL_RAS_ENABLE = '0'
$env:NCCL_SHM_DISABLE = '1'
$env:NCCL_P2P_DISABLE = '1'
$env:NCCL_HOSTID = 'windows-local'
$env:NCCL_COMM_BLOCKING = '1'
$env:NCCL_ALGO = 'Ring'
$env:NCCL_PROTO = 'Simple'

$RunRoot = Join-Path $ProjectRoot (
    'logs\large-model-benchmark-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
)
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

function Invoke-LatencyCase {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][bool]$UseD3D12
    )

    $env:WAVMG_USE_D3D12 = if ($UseD3D12) { '1' } else { '0' }
    $OutputJson = Join-Path $RunRoot "$Name.json"
    Write-Host (
        "Benchmarking ${Name}: batch=$BatchSize input=$InputLen output=$OutputLen " +
        "warmup=$WarmupIterations iterations=$Iterations"
    ) -ForegroundColor Cyan
    & $Vllm bench latency `
        --model $ModelPath `
        --tensor-parallel-size 2 `
        --distributed-executor-backend mp `
        --dtype bfloat16 `
        --quantization awq `
        --max-model-len $MaxModelLen `
        --max-num-batched-tokens $MaxNumBatchedTokens `
        --gpu-memory-utilization $GpuMemoryUtilization `
        --input-len $InputLen `
        --output-len $OutputLen `
        --batch-size $BatchSize `
        --num-iters-warmup $WarmupIterations `
        --num-iters $Iterations `
        --output-json $OutputJson `
        --shutdown-timeout 30 `
        --disable-log-stats `
        --enforce-eager `
        -O0 `
        --no-async-scheduling
    if ($LASTEXITCODE -ne 0) {
        throw "$Name benchmark failed with exit code $LASTEXITCODE."
    }
    Write-Host "$Name result: $OutputJson" -ForegroundColor Green
}

if ($Mode -in ('Hybrid', 'Both')) {
    Invoke-LatencyCase -Name 'hybrid-d3d12-rccl' -UseD3D12 $true
}
if ($Mode -in ('Rccl', 'Both')) {
    Invoke-LatencyCase -Name 'rccl-only' -UseD3D12 $false
}

if ($Mode -eq 'Both') {
    $Hybrid = Get-Content -Raw -LiteralPath (
        Join-Path $RunRoot 'hybrid-d3d12-rccl.json'
    ) | ConvertFrom-Json
    $Rccl = Get-Content -Raw -LiteralPath (
        Join-Path $RunRoot 'rccl-only.json'
    ) | ConvertFrom-Json
    $OutputTokens = $BatchSize * $OutputLen
    @(
        [pscustomobject]@{
            Transport = 'Hybrid D3D12 + RCCL'
            AverageSeconds = [math]::Round($Hybrid.avg_latency, 4)
            P50Seconds = [math]::Round($Hybrid.percentiles.'50', 4)
            OutputTokensPerSecond = [math]::Round(
                $OutputTokens / $Hybrid.avg_latency, 2
            )
        },
        [pscustomobject]@{
            Transport = 'RCCL only'
            AverageSeconds = [math]::Round($Rccl.avg_latency, 4)
            P50Seconds = [math]::Round($Rccl.percentiles.'50', 4)
            OutputTokensPerSecond = [math]::Round(
                $OutputTokens / $Rccl.avg_latency, 2
            )
        }
    ) | Format-Table -AutoSize
    $Speedup = $Rccl.avg_latency / $Hybrid.avg_latency
    $LatencyReduction = 100.0 * (1.0 - $Hybrid.avg_latency / $Rccl.avg_latency)
    Write-Host (
        'Hybrid versus RCCL-only: {0:N2}x faster, {1:N1}% lower latency.' -f
        $Speedup, $LatencyReduction
    ) -ForegroundColor Green
}

Write-Host "Benchmark results: $RunRoot" -ForegroundColor Green
