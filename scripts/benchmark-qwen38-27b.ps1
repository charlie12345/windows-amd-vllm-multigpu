#Requires -Version 5.1

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,
    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$ModelLabel = 'bf16',
    [ValidateSet('Single', 'Hybrid', 'Rccl')]
    [string]$Transport = 'Hybrid',
    [ValidateSet(
        'Baseline', 'EagerMtp', 'AsyncEager', 'AsyncEagerMtp',
        'TunedO1', 'TunedO1Mtp', 'TunedO2', 'TunedO2Mtp',
        'V2AsyncEager', 'DFlash2Eager', 'DFlash2TunedO1'
    )]
    [string]$Profile = 'Baseline',
    [string]$SpeculativeModelPath = '',
    [ValidateSet('balanced', 'interactivity', 'throughput')]
    [string]$PerformanceMode = 'balanced',
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
    [ValidateRange(1, 256)]
    [int]$MaxNumSeqs = 8,
    [ValidateRange(0.5, 0.99)]
    [double]$GpuMemoryUtilization = 0.92,
    [ValidateSet('bfloat16', 'float16')]
    [string]$Dtype = 'bfloat16',
    [ValidateSet('auto', 'ROCM_ATTN', 'TRITON_ATTN')]
    [string]$AttentionBackend = 'auto',
    [ValidateSet('ROCM_ATTN', 'TRITON_ATTN')]
    [string]$DFlashAttentionBackend = 'TRITON_ATTN',
    [ValidateRange(1, 16)]
    [int]$DFlashSpeculativeTokens = 7,
    [ValidateSet('auto', 'dummy')]
    [string]$LoadFormat = 'auto',
    [ValidateRange(0, 3600)]
    [int]$DistributedTimeoutSeconds = 0,
    [switch]$TraceCollectives
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Vllm = Join-Path $ProjectRoot '.venv-vllm\Scripts\vllm.exe'
$RcclDll = Join-Path $ProjectRoot 'build\rccl-windows\rccl.dll'
$D3D12Dll = Join-Path $ProjectRoot 'build\native\wavmg_d3d12_v1.dll'
$RequiredPaths = @($Vllm, $ModelPath)
if ($Transport -ne 'Single') {
    $RequiredPaths += @($RcclDll, $D3D12Dll)
}
foreach ($Required in $RequiredPaths) {
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
$UsesDFlash2 = $Profile -in @('DFlash2Eager', 'DFlash2TunedO1')
$UsesV2Runner = $UsesDFlash2 -or $Profile -eq 'V2AsyncEager'
if ($UsesDFlash2) {
    if ([string]::IsNullOrWhiteSpace($SpeculativeModelPath)) {
        throw "Profile $Profile requires -SpeculativeModelPath."
    }
    if (-not (Test-Path -LiteralPath $SpeculativeModelPath)) {
        throw "Missing DFlash2 checkpoint: $SpeculativeModelPath"
    }
    # A target/drafter backend mismatch can create incompatible shared KV-cache
    # layouts on ROCm. Match them by default; callers can explicitly select the
    # native ROCM_ATTN pair for experiments.
    if ($AttentionBackend -eq 'auto') {
        $AttentionBackend = $DFlashAttentionBackend
    }
}

$TensorParallelSize = if ($Transport -eq 'Single') { 1 } else { 2 }
$VisibleDevices = if ($TensorParallelSize -eq 1) { '0' } else { '0,1' }
$env:HIP_VISIBLE_DEVICES = $VisibleDevices
$env:CUDA_VISIBLE_DEVICES = $VisibleDevices
$env:VLLM_WORKER_MULTIPROC_METHOD = 'spawn'
$env:VLLM_HOST_IP = '127.0.0.1'
$env:VLLM_USE_V2_MODEL_RUNNER = if ($UsesV2Runner) { '1' } else { '0' }
$env:VLLM_DISTRIBUTED_USE_SPLIT_GROUP = '0'
$env:VLLM_PLUGINS = 'windows_amd_multigpu'
$env:VLLM_ROCM_USE_AITER = '0'
$env:VLLM_ROCM_USE_RDNA_W4A16 = '1'
$env:WAVMG_ENABLE = if ($TensorParallelSize -eq 2) { '1' } else { '0' }
$env:WAVMG_USE_RCCL = if ($TensorParallelSize -eq 2) { '1' } else { '0' }
$env:WAVMG_USE_D3D12 = if ($Transport -eq 'Hybrid') { '1' } else { '0' }
$env:WAVMG_RCCL_DLL = $RcclDll
$env:WAVMG_D3D12_DLL = $D3D12Dll
$env:WAVMG_TRACE_COLLECTIVES = if ($TraceCollectives) { '1' } else { '0' }
$env:HF_HUB_OFFLINE = '1'
$env:NCCL_DEBUG = 'WARN'
$env:NCCL_RAS_ENABLE = '0'
$env:NCCL_SHM_DISABLE = '1'
$env:NCCL_P2P_DISABLE = '1'
$env:NCCL_HOSTID = 'windows-local'
$env:NCCL_COMM_BLOCKING = '1'
$env:NCCL_ALGO = 'Ring'
$env:NCCL_PROTO = 'Simple'

$DFlash2Args = @()
if ($UsesDFlash2) {
    # Use dotted JSON arguments because Windows PowerShell 5.1 strips the
    # embedded quotes from a JSON object passed to a native executable.
    $DFlash2Args = @(
        '--speculative-config.method', 'dflash',
        '--speculative-config.model',
        (Resolve-Path -LiteralPath $SpeculativeModelPath).Path,
        '--speculative-config.num-speculative-tokens', $DFlashSpeculativeTokens,
        # A mixed ROCM_ATTN target and TRITON_ATTN drafter currently produce
        # incompatible shared KV-cache layouts. Keep this value matched to the
        # target backend selected above.
        '--speculative-config.attention-backend', $DFlashAttentionBackend
    )
}

$RunRoot = Join-Path $ProjectRoot (
    'logs\qwen38-27b-' + $ModelLabel + '-' + $Transport.ToLowerInvariant() +
    '-' + $Profile.ToLowerInvariant() + '-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
)
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$OutputJson = Join-Path $RunRoot 'latency.json'
$ConsoleLog = Join-Path $RunRoot 'console.log'

$BenchArgs = @(
    'bench', 'latency',
    '--model', (Resolve-Path -LiteralPath $ModelPath).Path,
    '--tensor-parallel-size', $TensorParallelSize,
    '--dtype', $Dtype,
    '--load-format', $LoadFormat,
    '--attention-backend', $AttentionBackend,
    '--language-model-only',
    '--max-model-len', $MaxModelLen,
    '--max-num-batched-tokens', $MaxNumBatchedTokens,
    '--max-num-seqs', $MaxNumSeqs,
    '--gpu-memory-utilization', $GpuMemoryUtilization,
    '--performance-mode', $PerformanceMode,
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
if ($TensorParallelSize -eq 2) {
    $BenchArgs += @('--distributed-executor-backend', 'mp')
}
if ($DistributedTimeoutSeconds -gt 0) {
    $BenchArgs += @(
        '--distributed-timeout-seconds', $DistributedTimeoutSeconds,
        '--cpu-distributed-timeout-seconds', $DistributedTimeoutSeconds
    )
}
switch ($Profile) {
    'Baseline' {
        $BenchArgs += @('--enforce-eager', '-O0', '--no-async-scheduling')
    }
    'EagerMtp' {
        $BenchArgs += @(
            '--enforce-eager', '-O0', '--no-async-scheduling',
            '--speculative-config',
            '{"method":"mtp","num_speculative_tokens":1}'
        )
    }
    'AsyncEager' {
        $BenchArgs += @('--enforce-eager', '-O0', '--async-scheduling')
    }
    'V2AsyncEager' {
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
    'TunedO2' {
        $BenchArgs += @(
            '-O2', '--async-scheduling',
            '--compilation-config',
            '{"cudagraph_capture_sizes":[1,8,32],"max_cudagraph_capture_size":32}'
        )
    }
    'TunedO2Mtp' {
        $BenchArgs += @(
            '-O2', '--async-scheduling',
            '--compilation-config',
            '{"cudagraph_capture_sizes":[1,8,32],"max_cudagraph_capture_size":32}',
            '--speculative-config',
            '{"method":"mtp","num_speculative_tokens":1}'
        )
    }
    'DFlash2Eager' {
        $BenchArgs += @(
            '--enforce-eager', '-O0', '--async-scheduling'
        )
        $BenchArgs += $DFlash2Args
    }
    'DFlash2TunedO1' {
        $BenchArgs += @(
            '-O1', '--async-scheduling',
            '--compilation-config', '{"compile_sizes":[]}'
        )
        $BenchArgs += $DFlash2Args
    }
}

Write-Host (
    "Benchmarking Qwen3.8-27B ${ModelLabel}: transport=$Transport " +
    "tp=$TensorParallelSize profile=$Profile performance=$PerformanceMode " +
    "load=$LoadFormat max-seqs=$MaxNumSeqs " +
    "batch=$BatchSize input=$InputLen output=$OutputLen"
) -ForegroundColor Cyan
# Windows PowerShell 5.1 converts native stderr records into PowerShell error
# records when stderr is merged into the success stream. PyTorch emits a
# harmless c10d IPv4-mapped IPv6 warning on stderr during Windows Gloo startup;
# with the script-wide Stop preference that warning terminated this launcher,
# killed the vLLM client, and orphaned an otherwise healthy EngineCore process.
# Keep strict failure handling for PowerShell code, but let the native process
# finish and use its real exit code as the authoritative result.
$PreviousErrorActionPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Continue'
    & $Vllm @BenchArgs 2>&1 |
        ForEach-Object {
            if ($_ -is [System.Management.Automation.ErrorRecord]) {
                $_.Exception.Message
            }
            else {
                $_
            }
        } |
        Tee-Object -FilePath $ConsoleLog
    $VllmExitCode = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $PreviousErrorActionPreference
}
if ($VllmExitCode -ne 0) {
    throw (
        "Qwen3.8 benchmark failed with exit code $VllmExitCode. " +
        "Full console log: $ConsoleLog"
    )
}

$Result = Get-Content -Raw -LiteralPath $OutputJson | ConvertFrom-Json
[pscustomobject]@{
    Model = $ModelLabel
    Transport = $Transport
    TensorParallelSize = $TensorParallelSize
    Profile = $Profile
    PerformanceMode = $PerformanceMode
    LoadFormat = $LoadFormat
    BatchSize = $BatchSize
    AverageSeconds = [math]::Round($Result.avg_latency, 4)
    P50Seconds = [math]::Round($Result.percentiles.'50', 4)
    P90Seconds = [math]::Round($Result.percentiles.'90', 4)
    OutputTokensPerSecond = [math]::Round(
        ($BatchSize * $OutputLen) / $Result.avg_latency,
        2
    )
    ApproxInputTokensPerSecond = [math]::Round(
        ($BatchSize * $InputLen) / $Result.avg_latency,
        2
    )
} | Format-List
Write-Host "Benchmark result: $OutputJson" -ForegroundColor Green
Write-Host "Console log: $ConsoleLog" -ForegroundColor Green
