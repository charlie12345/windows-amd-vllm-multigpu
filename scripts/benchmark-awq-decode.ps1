#Requires -Version 5.1

[CmdletBinding()]
param(
    [ValidateSet('Single', 'Hybrid', 'Rccl', 'All')]
    [string]$Mode = 'All',
    [ValidateSet('Baseline', 'AsyncEager', 'TunedO1', 'TunedO2')]
    [string]$Profile = 'Baseline',
    [ValidateRange(1, 4096)]
    [int]$InputLen = 32,
    [ValidateRange(1, 512)]
    [int]$OutputLen = 32,
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
    [double]$GpuMemoryUtilization = 0.92,
    [ValidateSet('bfloat16', 'float16')]
    [string]$Dtype = 'bfloat16',
    [ValidateSet('auto', 'ROCM_ATTN', 'TRITON_ATTN')]
    [string]$AttentionBackend = 'auto',
    [ValidateSet('auto', 'fp8')]
    [string]$KvCacheDtype = 'auto'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $ProjectRoot '.venv-vllm\Scripts\python.exe'
$Vllm = Join-Path $ProjectRoot '.venv-vllm\Scripts\vllm.exe'
$Resolver = Join-Path $ProjectRoot 'scripts\download-large-test-model.py'
$RcclDll = Join-Path $ProjectRoot 'build\rccl-windows\rccl.dll'
$D3D12Dll = Join-Path $ProjectRoot 'build\native\wavmg_d3d12_v1.dll'
$HipDll = Join-Path $ProjectRoot 'build\native\wavmg_hip_v1.dll'
foreach ($Required in ($Python, $Vllm, $Resolver, $RcclDll, $D3D12Dll, $HipDll)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Missing prerequisite: $Required"
    }
}
if ($Mode -ne 'Single') {
    & (Join-Path $PSScriptRoot 'assert-windows-amd-gpu-health.ps1') `
        -RequiredCount 2
}
if (($InputLen + $OutputLen) -gt $MaxModelLen) {
    throw 'InputLen plus OutputLen cannot exceed MaxModelLen.'
}
if (($InputLen * $BatchSize) -gt $MaxNumBatchedTokens) {
    throw 'InputLen times BatchSize cannot exceed MaxNumBatchedTokens.'
}

$ResolverOutput = @(& $Python $Resolver --local-files-only)
if ($LASTEXITCODE -ne 0) {
    throw 'Download the pinned AWQ model with scripts\download-large-test-model.py first.'
}
$ModelLine = $ResolverOutput | Where-Object { $_ -like 'WAVMG_MODEL_PATH=*' } |
    Select-Object -Last 1
if (-not $ModelLine) {
    throw 'Could not resolve the pinned AWQ model snapshot.'
}
$ModelPath = $ModelLine.Substring('WAVMG_MODEL_PATH='.Length)

$env:VLLM_WORKER_MULTIPROC_METHOD = 'spawn'
$env:VLLM_USE_V2_MODEL_RUNNER = '0'
$env:VLLM_DISTRIBUTED_USE_SPLIT_GROUP = '0'
$env:VLLM_PLUGINS = 'windows_amd_multigpu'
$env:VLLM_ROCM_USE_RDNA_W4A16 = '1'
$env:WAVMG_RCCL_DLL = $RcclDll
$env:WAVMG_D3D12_DLL = $D3D12Dll
$env:WAVMG_HIP_DLL = $HipDll
$env:WAVMG_TRACE_COLLECTIVES = '0'
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
    'logs\awq-decode-benchmark-' + $Profile.ToLowerInvariant() + '-' +
    (Get-Date -Format 'yyyyMMdd-HHmmss')
)
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

function Invoke-LatencyCase {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][int]$TensorParallelSize,
        [Parameter(Mandatory = $true)][bool]$EnableTransport,
        [Parameter(Mandatory = $true)][bool]$UseD3D12
    )

    if ($TensorParallelSize -eq 1) {
        $env:HIP_VISIBLE_DEVICES = '0'
        $env:CUDA_VISIBLE_DEVICES = '0'
    } else {
        $env:HIP_VISIBLE_DEVICES = '0,1'
        $env:CUDA_VISIBLE_DEVICES = '0,1'
    }
    $env:WAVMG_ENABLE = if ($EnableTransport) { '1' } else { '0' }
    $env:WAVMG_USE_RCCL = if ($EnableTransport) { '1' } else { '0' }
    $env:WAVMG_USE_D3D12 = if ($UseD3D12) { '1' } else { '0' }

    $OutputJson = Join-Path $RunRoot "$Name.json"
    $BenchArgs = @(
        'bench', 'latency',
        '--model', $ModelPath,
        '--tensor-parallel-size', $TensorParallelSize,
        '--distributed-executor-backend', 'mp',
        '--dtype', $Dtype,
        '--kv-cache-dtype', $KvCacheDtype,
        '--quantization', 'awq',
        '--max-model-len', $MaxModelLen,
        '--max-num-batched-tokens', $MaxNumBatchedTokens,
        '--gpu-memory-utilization', $GpuMemoryUtilization,
        '--input-len', $InputLen,
        '--output-len', $OutputLen,
        '--batch-size', $BatchSize,
        '--num-iters-warmup', $WarmupIterations,
        '--num-iters', $Iterations,
        '--output-json', $OutputJson,
        '--shutdown-timeout', '30',
        '--disable-log-stats',
        '--no-enable-prefix-caching'
    )
    if ($AttentionBackend -ne 'auto') {
        $BenchArgs += @('--attention-backend', $AttentionBackend)
    }
    switch ($Profile) {
        'Baseline' {
            $BenchArgs += @('--enforce-eager', '-O0', '--no-async-scheduling')
        }
        'AsyncEager' {
            # Isolate scheduler pipelining from torch.compile. This keeps the
            # same eager/O0 execution path used by the baseline.
            $BenchArgs += @('--enforce-eager', '-O0', '--async-scheduling')
        }
        'TunedO1' {
            # The pinned vLLM nightly leaves compile_sizes as None at O1/O2,
            # which makes its piecewise range lookup return early. An empty
            # list enables dynamic-range lookup without adding static shapes.
            $CompileConfig = '{"compile_sizes":[]}'
            $BenchArgs += @(
                '-O1', '--async-scheduling',
                '--compilation-config', $CompileConfig
            )
        }
        'TunedO2' {
            $CompileConfig = '{"compile_sizes":[]}'
            $BenchArgs += @(
                '-O2', '--async-scheduling',
                '--compilation-config', $CompileConfig
            )
        }
    }

    Write-Host (
        "Benchmarking ${Name}: profile=$Profile tp=$TensorParallelSize " +
        "batch=$BatchSize input=$InputLen output=$OutputLen dtype=$Dtype " +
        "attention=$AttentionBackend kv=$KvCacheDtype"
    ) -ForegroundColor Cyan
    & $Vllm @BenchArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$Name benchmark failed with exit code $LASTEXITCODE."
    }
    Write-Host "$Name result: $OutputJson" -ForegroundColor Green
}

if ($Mode -in ('Single', 'All')) {
    Invoke-LatencyCase -Name 'single-gpu' -TensorParallelSize 1 `
        -EnableTransport $false -UseD3D12 $false
}
if ($Mode -in ('Hybrid', 'All')) {
    Invoke-LatencyCase -Name 'tp2-hybrid-d3d12-rccl' -TensorParallelSize 2 `
        -EnableTransport $true -UseD3D12 $true
}
if ($Mode -in ('Rccl', 'All')) {
    Invoke-LatencyCase -Name 'tp2-rccl-only' -TensorParallelSize 2 `
        -EnableTransport $true -UseD3D12 $false
}

$Results = foreach ($ResultFile in Get-ChildItem -LiteralPath $RunRoot -Filter '*.json') {
    $Result = Get-Content -Raw -LiteralPath $ResultFile.FullName | ConvertFrom-Json
    [pscustomobject]@{
        Case = $ResultFile.BaseName
        AverageSeconds = [math]::Round($Result.avg_latency, 4)
        P50Seconds = [math]::Round($Result.percentiles.'50', 4)
        P90Seconds = [math]::Round($Result.percentiles.'90', 4)
        OutputTokensPerSecond = [math]::Round(
            ($BatchSize * $OutputLen) / $Result.avg_latency, 2
        )
    }
}
$Results | Sort-Object Case | Format-Table -AutoSize
Write-Host "Benchmark results: $RunRoot" -ForegroundColor Green
