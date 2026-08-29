#Requires -Version 5.1

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Pins = Get-Content -Raw -LiteralPath (
    Join-Path $ProjectRoot 'pins\rocm10-vllm-v0.28.0.json'
) | ConvertFrom-Json
$Checkout = Join-Path $ProjectRoot 'sandbox\rocm-systems-hip-p2p'
$Patch = Join-Path $ProjectRoot 'patches\hip-p2p\44be71b-force-pal-peer-probe.patch'

if (-not (Test-Path -LiteralPath (Join-Path $Checkout '.git'))) {
    throw 'Run scripts\sync-hip-p2p-runtime.ps1 first.'
}
if (-not (Test-Path -LiteralPath $Patch -PathType Leaf)) {
    throw "HIP peer-probe patch is missing: $Patch"
}

$Current = (& git -C $Checkout rev-parse HEAD).Trim()
if ($Current -ne $Pins.hip_runtime_rocm_systems_commit) {
    throw "HIP peer-probe patch requires $($Pins.hip_runtime_rocm_systems_commit); found $Current."
}

& git -C $Checkout apply --reverse --check $Patch 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host 'The pinned experimental HIP/PAL patch is already applied.' -ForegroundColor Green
    exit 0
}

$Dirty = @(& git -C $Checkout status --porcelain)
if ($Dirty.Count -gt 0) {
    throw 'Refusing to apply the HIP/PAL patch over an already modified checkout.'
}

& git -C $Checkout apply --check $Patch
if ($LASTEXITCODE -ne 0) {
    throw 'The HIP/PAL patch does not apply cleanly to the pinned source.'
}
& git -C $Checkout apply $Patch
if ($LASTEXITCODE -ne 0) {
    throw 'Applying the HIP/PAL patch failed.'
}

Write-Host 'Applied the opt-in experimental HIP/PAL peer-probe patch.' -ForegroundColor Green
