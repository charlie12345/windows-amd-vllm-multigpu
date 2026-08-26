#Requires -Version 5.1

[CmdletBinding()]
param(
    [ValidateSet('Single', 'Rccl', 'Hybrid', 'All')]
    [string]$Mode = 'All',
    [string]$ModelPath = 'G:\AI-models\Ling-3.0-tiny-b61f4338',
    [ValidateRange(1, 4096)]
    [int]$InputLen = 32,
    [ValidateRange(1, 511)]
    [int]$OutputLen = 128,
    [ValidateRange(0, 100)]
    [int]$WarmupIterations = 3,
    [ValidateRange(1, 1000)]
    [int]$Iterations = 5
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Vllm = Join-Path $ProjectRoot '.venv-vllm\Scripts\vllm.exe'
$RcclDll = Join-Path $ProjectRoot 'build\rccl-windows\rccl.dll'
$D3D12Dll = Join-Path $ProjectRoot 'build\native\wavmg_d3d12_v1.dll'
foreach ($Required in ($Vllm, $ModelPath)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Missing prerequisite: $Required"
    }
}
if (($InputLen + $OutputLen) -gt 512) {
    throw 'InputLen plus OutputLen cannot exceed 512.'
}

$RunRoot = Join-Path $ProjectRoot (
    'logs\ling3-bf16-benchmark-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
)
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$env:VLLM_WORKER_MULTIPROC_METHOD = 'spawn'
$env:VLLM_DISTRIBUTED_USE_SPLIT_GROUP = '0'
$env:VLLM_PLUGINS = 'windows_amd_multigpu'
$env:WAVMG_ENABLE = '1'
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
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue

function Invoke-LingBenchmark {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][ValidateSet(1, 2)][int]$Tp,
        [Parameter(Mandatory = $true)][bool]$UseD3D12
    )

    $env:HIP_VISIBLE_DEVICES = if ($Tp -eq 2) { '0,1' } else { '0' }
    $env:WAVMG_USE_RCCL = if ($Tp -eq 2) { '1' } else { '0' }
    $env:WAVMG_USE_D3D12 = if ($Tp -eq 2 -and $UseD3D12) { '1' } else { '0' }
    if ($Tp -eq 2) {
        foreach ($Required in ($RcclDll, $D3D12Dll)) {
            if (-not (Test-Path -LiteralPath $Required)) {
                throw "Missing native transport: $Required"
            }
        }
        $env:WAVMG_RCCL_DLL = $RcclDll
        $env:WAVMG_D3D12_DLL = $D3D12Dll
    }

    $OutputJson = Join-Path $RunRoot "$Name.json"
    Write-Host "Benchmarking Ling BF16 $Name" -ForegroundColor Cyan
    & $Vllm bench latency `
        --model $ModelPath `
        --trust-remote-code `
        --tensor-parallel-size $Tp `
        --distributed-executor-backend mp `
        --dtype bfloat16 `
        --max-model-len 512 `
        --max-num-batched-tokens 511 `
        --gpu-memory-utilization $(if ($Tp -eq 2) { 0.70 } else { 0.80 }) `
        --input-len $InputLen `
        --output-len $OutputLen `
        --batch-size 1 `
        --num-iters-warmup $WarmupIterations `
        --num-iters $Iterations `
        --output-json $OutputJson `
        --shutdown-timeout 30 `
        --disable-log-stats `
        --async-scheduling `
        -O1 `
        --no-enable-prefix-caching
    if ($LASTEXITCODE -ne 0) {
        throw "$Name benchmark failed with exit code $LASTEXITCODE."
    }

    $Result = Get-Content -Raw -LiteralPath $OutputJson | ConvertFrom-Json
    [pscustomobject]@{
        Configuration = $Name
        AverageSeconds = [math]::Round($Result.avg_latency, 4)
        P50Seconds = [math]::Round($Result.percentiles.'50', 4)
        OutputTokensPerSecond = [math]::Round(
            $OutputLen / $Result.avg_latency,
            2
        )
    }
}

$Results = @()
if ($Mode -in ('Single', 'All')) {
    $Results += Invoke-LingBenchmark -Name 'tp1' -Tp 1 -UseD3D12 $false
}
if ($Mode -in ('Rccl', 'All')) {
    $Results += Invoke-LingBenchmark -Name 'tp2-rccl' -Tp 2 -UseD3D12 $false
}
if ($Mode -in ('Hybrid', 'All')) {
    $Results += Invoke-LingBenchmark -Name 'tp2-hybrid' -Tp 2 -UseD3D12 $true
}

$Results | Format-Table -AutoSize
Write-Host "Benchmark results: $RunRoot" -ForegroundColor Green
