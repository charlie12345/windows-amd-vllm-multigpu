#Requires -Version 5.1

[CmdletBinding()]
param(
    [ValidateSet('Baseline', 'AsyncEager', 'TunedO1')]
    [string]$Profile = 'TunedO1',
    [ValidateRange(128, 32768)]
    [int]$MaxModelLen = 32768,
    [ValidateRange(128, 32768)]
    [int]$MaxNumBatchedTokens = 2048,
    [ValidateRange(1, 1024)]
    [int]$MaxNumSeqs = 128,
    [ValidateRange(0.5, 0.99)]
    [double]$GpuMemoryUtilization = 0.92,
    [ValidateRange(1, 65535)]
    [int]$Port = 8000,
    [string]$HostAddress = '127.0.0.1',
    [switch]$DisablePrefixCaching
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($MaxNumBatchedTokens -lt $MaxNumSeqs) {
    throw 'MaxNumBatchedTokens must be greater than or equal to MaxNumSeqs.'
}

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $ProjectRoot '.venv-vllm\Scripts\python.exe'
$Vllm = Join-Path $ProjectRoot '.venv-vllm\Scripts\vllm.exe'
$Resolver = Join-Path $ProjectRoot 'scripts\download-large-test-model.py'
$RcclDll = Join-Path $ProjectRoot 'build\rccl-windows\rccl.dll'
foreach ($Required in ($Python, $Vllm, $Resolver, $RcclDll)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Missing prerequisite: $Required"
    }
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

$env:HIP_VISIBLE_DEVICES = '0,1'
$env:CUDA_VISIBLE_DEVICES = '0,1'
$env:VLLM_WORKER_MULTIPROC_METHOD = 'spawn'
$env:VLLM_USE_V2_MODEL_RUNNER = '0'
$env:VLLM_DISTRIBUTED_USE_SPLIT_GROUP = '0'
$env:VLLM_PLUGINS = 'windows_amd_multigpu'
$env:VLLM_ROCM_USE_RDNA_W4A16 = '1'
$env:WAVMG_ENABLE = '1'
$env:WAVMG_USE_RCCL = '1'
$env:WAVMG_USE_D3D12 = '0'
$env:WAVMG_RCCL_DLL = $RcclDll
$env:HF_HUB_OFFLINE = '1'
$env:NCCL_DEBUG = 'WARN'
$env:NCCL_RAS_ENABLE = '0'
$env:NCCL_SHM_DISABLE = '1'
$env:NCCL_P2P_DISABLE = '1'
$env:NCCL_HOSTID = 'windows-local'
$env:NCCL_COMM_BLOCKING = '1'
$env:NCCL_ALGO = 'Ring'
$env:NCCL_PROTO = 'Simple'

$ServeArgs = @(
    'serve', $ModelPath,
    '--served-model-name', 'mistral-small-24b-awq',
    '--host', $HostAddress,
    '--port', $Port,
    '--tensor-parallel-size', '2',
    '--distributed-executor-backend', 'mp',
    '--dtype', 'bfloat16',
    '--quantization', 'awq',
    '--attention-backend', 'auto',
    '--max-model-len', $MaxModelLen,
    '--max-num-batched-tokens', $MaxNumBatchedTokens,
    '--max-num-seqs', $MaxNumSeqs,
    '--gpu-memory-utilization', $GpuMemoryUtilization
)
if ($DisablePrefixCaching) {
    $ServeArgs += '--no-enable-prefix-caching'
} else {
    $ServeArgs += '--enable-prefix-caching'
}
switch ($Profile) {
    'Baseline' {
        $ServeArgs += @('--enforce-eager', '-O0', '--no-async-scheduling')
    }
    'AsyncEager' {
        $ServeArgs += @('--enforce-eager', '-O0', '--async-scheduling')
    }
    'TunedO1' {
        $ServeArgs += @(
            '-O1', '--async-scheduling',
            '--compilation-config', '{"compile_sizes":[]}'
        )
    }
}

Write-Host (
    "Serving mistral-small-24b-awq at http://${HostAddress}:$Port/v1 " +
    "with TP2 RCCL, profile=$Profile"
) -ForegroundColor Cyan
& $Vllm @ServeArgs
if ($LASTEXITCODE -ne 0) {
    throw "vLLM server exited with code $LASTEXITCODE."
}
