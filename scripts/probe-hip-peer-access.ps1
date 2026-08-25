#Requires -Version 5.1

[CmdletBinding()]
param(
    [string]$HipRuntimeDll,
    [string[]]$DependencyDirectory = @(),
    [ValidateRange(1, 10000)]
    [int]$Iterations = 10,
    [switch]$ForcePalPeerProbe
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $ProjectRoot '.venv-vllm\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Missing vLLM Python environment: $Python"
}

$env:PYTHONPATH = Join-Path $ProjectRoot 'src'
$Arguments = @(
    (Join-Path $ProjectRoot 'probes\hip_peer_access_probe.py'),
    '--iterations',
    $Iterations
)
if ($HipRuntimeDll) {
    $Arguments += @('--runtime-dll', $HipRuntimeDll)
}
foreach ($Directory in $DependencyDirectory) {
    $Arguments += @('--dependency-dir', $Directory)
}
if ($ForcePalPeerProbe) {
    if (-not $HipRuntimeDll) {
        throw '-ForcePalPeerProbe requires -HipRuntimeDll to protect the installed runtime.'
    }
    $env:GPU_FORCE_P2P_COMPAT = '1'
    $Arguments += '--fail-closed-pal-peer-probe'
}

& $Python @Arguments
exit $LASTEXITCODE
