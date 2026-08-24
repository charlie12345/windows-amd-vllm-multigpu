$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Pins = Get-Content (Join-Path $ProjectRoot "pins\nightly-2026-07-28.json") |
    ConvertFrom-Json
$VllmRoot = Join-Path $ProjectRoot "sandbox\vllm"
$Patch = Join-Path $ProjectRoot "patches\vllm\fb9fb8c5-windows-pipeconnection.patch"

$ActualCommit = (git -C $VllmRoot rev-parse HEAD).Trim()
if ($ActualCommit -ne $Pins.vllm_commit) {
    throw "Refusing to patch vLLM $ActualCommit; expected $($Pins.vllm_commit)"
}

git -C $VllmRoot apply --check $Patch 2>$null
if ($LASTEXITCODE -eq 0) {
    git -C $VllmRoot apply $Patch
    if ($LASTEXITCODE -ne 0) { throw "vLLM patch application failed" }
    Write-Host "Applied the pinned Windows vLLM compatibility patch."
    return
}

git -C $VllmRoot apply --reverse --check $Patch 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "The pinned Windows vLLM compatibility patch is already applied."
    return
}

throw "The pinned vLLM patch neither applies nor appears already applied."
