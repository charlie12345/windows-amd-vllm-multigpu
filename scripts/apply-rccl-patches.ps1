#Requires -Version 5.1

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Pins = Get-Content -Raw -LiteralPath (
    Join-Path $ProjectRoot 'pins\rocm10-vllm-v0.28.0.json'
) | ConvertFrom-Json
$Checkout = Join-Path $ProjectRoot 'sandbox\rocm-systems'
$Patch = Join-Path $ProjectRoot 'patches\rccl\ee3bae9-windows-native.patch'

if (-not (Test-Path -LiteralPath (Join-Path $Checkout '.git'))) {
    throw 'Run scripts\sync-upstreams.ps1 first.'
}
if (-not (Test-Path -LiteralPath $Patch)) {
    throw "RCCL patch is missing: $Patch"
}

$Current = (& git -C $Checkout rev-parse HEAD).Trim()
if ($Current -ne $Pins.rocm_systems_commit) {
    throw "RCCL patch requires $($Pins.rocm_systems_commit); found $Current."
}

& git -C $Checkout apply --reverse --check $Patch 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host 'The pinned RCCL Windows patch is already applied.' -ForegroundColor Green
    exit 0
}

$Dirty = @(& git -C $Checkout status --porcelain)
if ($Dirty.Count -gt 0) {
    throw 'Refusing to apply the RCCL patch over an already modified checkout.'
}

& git -C $Checkout apply --check $Patch
if ($LASTEXITCODE -ne 0) {
    throw 'The RCCL patch does not apply cleanly to the pinned source.'
}
& git -C $Checkout apply $Patch
if ($LASTEXITCODE -ne 0) {
    throw 'Applying the RCCL Windows patch failed.'
}

Write-Host 'Applied the pinned native-Windows RCCL patch.' -ForegroundColor Green
