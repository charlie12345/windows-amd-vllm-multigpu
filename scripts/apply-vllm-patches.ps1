<#
.SYNOPSIS
    Compatibility wrapper for the former vLLM patch installer.

.DESCRIPTION
    v0.2.0-rc2 no longer applies source patches to vLLM. Required Windows host
    changes live in charlie12345/vLLM_for_AMD. This wrapper remains so older
    automation fails closed through the read-only host verifier instead of
    silently modifying the engine checkout.
#>
#Requires -Version 5.1

[CmdletBinding()]
param(
    [string]$VllmRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Write-Warning (
    'apply-vllm-patches.ps1 is retired: the plugin now applies zero vLLM ' +
    'patches. Running the read-only host verifier.'
)
& (Join-Path $PSScriptRoot 'verify-vllm-host.ps1') -VllmRoot $VllmRoot
if ($LASTEXITCODE -ne 0) {
    throw 'Windows AMD vLLM host verification failed.'
}
