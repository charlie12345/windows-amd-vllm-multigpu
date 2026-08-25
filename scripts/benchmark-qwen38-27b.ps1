#Requires -Version 5.1

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,
    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$ModelLabel = 'bf16',
    [ValidateSet('Hybrid', 'Rccl')]
    [string]$Transport = 'Hybrid',
    [ValidateSet(
        'Baseline', 'AsyncEager', 'AsyncEagerMtp', 'TunedO1', 'TunedO1Mtp'
    )]
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
    [string]$AttentionBackend = 'auto'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Vllm = Join-Path $ProjectRoot '.venv-vllm\Scripts\vllm.exe'
$RcclDll = Join-Path $ProjectRoot 'build\rccl-windows\rccl.dll'
$D3D12Dll = Join-Path $ProjectRoot 'build\native\wavmg_d3d12_v1.dll'
foreach ($Required in ($Vllm, $RcclDll, $D3D12Dll, $ModelPath)) {
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

$env:HIP_VISIBLE_DEVICES = '0,1'
$env:CUDA_VISIBLE_DEVICES = '0,1'
$env:VLLM_WORKER_MULTIPROC_METHOD = 'spawn'
$env:VLLM_USE_V2_MODEL_RUNNER = '0'
$env:VLLM_DISTRIBUTED_USE_SPLIT_GROUP = '0'
$env:VLLM_PLUGINS = 'windows_amd_multigpu'
$env:VLLM_ROCM_USE_RDNA_W4A16 = '1'
$env:WAVMG_ENABLE = '1'
$env:WAVMG_USE_RCCL = '1'
$env:WAVMG_USE_D3D12 = if ($Transport -eq 'Hybrid') { '1' } else { '0' }
$env:WAVMG_RCCL_DLL = $RcclDll
$env:WAVMG_D3D12_DLL = $D3D12Dll
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
    'logs\qwen38-27b-' + $ModelLabel + '-' + $Transport.ToLowerInvariant() +
    '-' + $Profile.ToLowerInvariant() + '-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
)
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$OutputJson = Join-Path $RunRoot 'latency.json'

$BenchArgs = @(
    'bench', 'latency',
    '--model', (Resolve-Path -LiteralPath $ModelPath).Path,
    '--tensor-parallel-size', '2',
    '--distributed-executor-backend', 'mp',
    '--dtype', $Dtype,
    '--attention-backend', $AttentionBackend,
    '--language-model-only',
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
switch ($Profile) {
    'Baseline' {
        $BenchArgs += @('--enforce-eager', '-O0', '--no-async-scheduling')
    }
    'AsyncEager' {
        $BenchArgs += @('--enforce-eager', '-O0', '--async-scheduling')
    }
    'AsyncEagerMtp' {
        $BenchArgs += @(
            '--enforce-eager', '-O0', '--async-scheduling',
            '--speculative-config',
            '{"method":"mtp","num_speculative_tokens":1}'
        )
    }
    'TunedO1' {
        $BenchArgs += @(
            '-O1', '--async-scheduling',
            '--compilation-config', '{"compile_sizes":[]}'
        )
    }
    'TunedO1Mtp' {
        $BenchArgs += @(
            '-O1', '--async-scheduling',
            '--compilation-config', '{"compile_sizes":[]}',
            '--speculative-config',
            '{"method":"mtp","num_speculative_tokens":1}'
        )
    }
}

Write-Host (
    "Benchmarking Qwen3.8-27B ${ModelLabel}: transport=$Transport " +
    "profile=$Profile batch=$BatchSize input=$InputLen output=$OutputLen"
) -ForegroundColor Cyan
& $Vllm @BenchArgs
if ($LASTEXITCODE -ne 0) {
    throw "Qwen3.8 benchmark failed with exit code $LASTEXITCODE."
}

$Result = Get-Content -Raw -LiteralPath $OutputJson | ConvertFrom-Json
[pscustomobject]@{
    Model = $ModelLabel
    Transport = $Transport
    Profile = $Profile
    BatchSize = $BatchSize
    AverageSeconds = [math]::Round($Result.avg_latency, 4)
    P50Seconds = [math]::Round($Result.percentiles.'50', 4)
    P90Seconds = [math]::Round($Result.percentiles.'90', 4)
    OutputTokensPerSecond = [math]::Round(
        ($BatchSize * $OutputLen) / $Result.avg_latency,
        2
    )
} | Format-List
Write-Host "Benchmark result: $OutputJson" -ForegroundColor Green
