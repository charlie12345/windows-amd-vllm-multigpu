<#
.SYNOPSIS
    Verify the exact clean Windows AMD vLLM host expected by this plugin.

.DESCRIPTION
    This script is read-only. It verifies the pinned Git revision and rejects
    tracked source edits. The Direct-RCCL/D3D12 package is an external vLLM
    plugin and must not patch or copy files into the vLLM source tree.
#>
#Requires -Version 5.1

[CmdletBinding()]
param(
    [string]$VllmRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$PinPath = Join-Path $ProjectRoot 'pins\rocm10-vllm-v0.28.0.json'
$Pins = Get-Content -Raw -LiteralPath $PinPath | ConvertFrom-Json

if ([string]::IsNullOrWhiteSpace($VllmRoot)) {
    $VllmRoot = Join-Path $ProjectRoot 'sandbox\vllm'
}
$VllmRoot = [IO.Path]::GetFullPath($VllmRoot)

if (-not (Test-Path -LiteralPath (Join-Path $VllmRoot '.git'))) {
    throw "Expected a Git clone of vLLM at $VllmRoot."
}
if ($Pins.vllm_source_patches_required -ne $false) {
    throw 'The current pin must declare vllm_source_patches_required=false.'
}

$ActualCommit = (& git -C $VllmRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Could not read the vLLM revision at $VllmRoot."
}
if ($ActualCommit -ne $Pins.vllm_commit) {
    throw @"
Refusing an unvalidated vLLM host.
  actual:   $ActualCommit
  expected: $($Pins.vllm_commit)
Build or check out the exact pinned charlie12345/vLLM_for_AMD revision.
"@
}

$TrackedChanges = @(& git -C $VllmRoot status --porcelain --untracked-files=no)
if ($LASTEXITCODE -ne 0) {
    throw "Could not inspect the vLLM worktree at $VllmRoot."
}
if ($TrackedChanges.Count -gt 0) {
    throw @"
The vLLM host has tracked source changes. This plugin does not patch vLLM.
Review or preserve those changes in their own branch, then use a clean checkout.
$($TrackedChanges -join "`n")
"@
}

$UpstreamVersionPath = Join-Path $VllmRoot 'UPSTREAM_VERSION'
if (-not (Test-Path -LiteralPath $UpstreamVersionPath -PathType Leaf)) {
    throw "The Windows fork marker is missing: $UpstreamVersionPath"
}
$UpstreamVersion = (Get-Content -Raw -LiteralPath $UpstreamVersionPath).Trim()
if ($UpstreamVersion -ne $Pins.vllm_upstream_tag) {
    throw "Expected upstream $($Pins.vllm_upstream_tag); found $UpstreamVersion."
}

Write-Host 'Windows AMD vLLM host verification passed.' -ForegroundColor Green
Write-Host "  source:   $VllmRoot"
Write-Host "  commit:   $ActualCommit"
Write-Host "  upstream: $UpstreamVersion"
Write-Host '  patches:  none (external plugin integration)'
