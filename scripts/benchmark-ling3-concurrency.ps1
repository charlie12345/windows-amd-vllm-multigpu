#Requires -Version 5.1

[CmdletBinding()]
param(
    [string]$BaseUrl = 'http://127.0.0.1:8001',
    [string]$ServedModelName = 'Ling-3.0-tiny',
    [string]$ModelPath = 'G:\AI-models\Ling-3.0-tiny-b61f4338',
    [ValidateNotNullOrEmpty()]
    [int[]]$Concurrency = @(1, 2, 4, 8, 16),
    [ValidateRange(1, 4096)]
    [int]$InputLen = 32,
    [ValidateRange(1, 511)]
    [int]$OutputLen = 128,
    [string]$ResultDir = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Vllm = Join-Path $ProjectRoot '.venv-vllm\Scripts\vllm.exe'
foreach ($Required in ($Vllm, $ModelPath)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Missing prerequisite: $Required"
    }
}
if (($InputLen + $OutputLen) -gt 512) {
    throw 'InputLen plus OutputLen cannot exceed the 512-token test server limit.'
}
foreach ($Value in $Concurrency) {
    if ($Value -lt 1 -or $Value -gt 32) {
        throw "Concurrency must be between 1 and 32; received $Value."
    }
}

if ([string]::IsNullOrWhiteSpace($ResultDir)) {
    $ResultDir = Join-Path $ProjectRoot (
        'logs\ling3-concurrency-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
    )
}
New-Item -ItemType Directory -Force -Path $ResultDir | Out-Null

try {
    $Health = Invoke-WebRequest -UseBasicParsing -Uri "$BaseUrl/health"
} catch {
    throw "The vLLM server at $BaseUrl is not healthy: $($_.Exception.Message)"
}
if ($Health.StatusCode -ne 200) {
    throw "The vLLM server returned health status $($Health.StatusCode)."
}

$Rows = foreach ($Value in $Concurrency) {
    $NumPrompts = if ($Value -ge 8) { 32 } else { 16 }
    $OutputJson = Join-Path $ResultDir "concurrency-$Value.json"
    Write-Host "Benchmarking concurrency $Value ($NumPrompts requests)" `
        -ForegroundColor Cyan

    & $Vllm bench serve `
        --backend openai `
        --base-url $BaseUrl `
        --endpoint /v1/completions `
        --model $ServedModelName `
        --tokenizer $ModelPath `
        --trust-remote-code `
        --dataset-name random `
        --num-prompts $NumPrompts `
        --num-warmups $Value `
        --request-rate inf `
        --max-concurrency $Value `
        --random-input-len $InputLen `
        --random-output-len $OutputLen `
        --ignore-eos `
        --save-result `
        --result-dir $ResultDir `
        --result-filename (Split-Path -Leaf $OutputJson)
    if ($LASTEXITCODE -ne 0) {
        throw "Concurrency $Value benchmark failed with exit code $LASTEXITCODE."
    }

    $Result = Get-Content -Raw -LiteralPath $OutputJson | ConvertFrom-Json
    [pscustomobject]@{
        Concurrency = $Value
        SuccessfulRequests = $Result.completed
        OutputTokensPerSecond = $Result.output_throughput
        TotalTokensPerSecond = $Result.total_token_throughput
        MeanTTFTMilliseconds = $Result.mean_ttft_ms
        MeanTPOTMilliseconds = $Result.mean_tpot_ms
        P99ITLMilliseconds = $Result.p99_itl_ms
    }
}

$Rows | Format-Table -AutoSize
Write-Host "Benchmark JSON: $ResultDir" -ForegroundColor Green
