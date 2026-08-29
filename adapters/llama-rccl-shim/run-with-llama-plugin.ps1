#Requires -Version 5.1

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Executable,

    [Parameter(Mandatory = $true)]
    [string]$RocmRoot,

    [ValidateSet('hybrid', 'd3d12', 'rccl')]
    [string]$Mode = 'hybrid',

    [switch]$EnableExperimentalLlamaRccl,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExecutableArguments
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$PluginRoot = $PSScriptRoot
$PluginBin = Join-Path $PluginRoot 'bin'
$RocmBin = Join-Path $RocmRoot 'bin'
$RocmLlvmBin = Join-Path $RocmRoot 'lib\llvm\bin'

foreach ($Path in @($Executable, $PluginBin, $RocmBin, $RocmLlvmBin)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required path does not exist: $Path"
    }
}
foreach ($Dll in @('rccl.dll', 'rccl-real.dll')) {
    $DllPath = Join-Path $PluginBin $Dll
    if (-not (Test-Path -LiteralPath $DllPath)) {
        throw "Plugin DLL does not exist: $DllPath"
    }
}

$Adapters = @(Get-PnpDevice -Class Display -PresentOnly -ErrorAction Stop |
    Where-Object { $_.FriendlyName -match 'AMD|Radeon' })
if ($Adapters.Count -lt 2) {
    throw "Detected $($Adapters.Count) AMD adapter(s); two are required."
}
$Unhealthy = @($Adapters | Where-Object { $_.Status -ne 'OK' })
if ($Unhealthy.Count -gt 0) {
    $Details = ($Unhealthy | ForEach-Object {
        '{0}: status={1}, problem={2}' -f $_.FriendlyName, $_.Status, $_.Problem
    }) -join '; '
    throw "Refusing multi-GPU launch with an unhealthy AMD adapter: $Details"
}

$env:ROCM_PATH = $RocmRoot
$env:HIP_PATH = $RocmRoot
$env:HIP_PLATFORM = 'amd'
$env:HIP_DEVICE_LIB_PATH = Join-Path $RocmRoot 'lib\llvm\amdgcn\bitcode'
$env:GGML_CUDA_ALLREDUCE = 'nccl'
$env:WAC_MODE = $Mode
$env:WAC_LLAMA_RCCL_EXPERIMENTAL = if ($EnableExperimentalLlamaRccl) { '1' } else { '0' }
$env:PATH = "$PluginBin;$RocmBin;$RocmLlvmBin;$env:PATH"

if ($Mode -eq 'rccl' -and -not $EnableExperimentalLlamaRccl) {
    throw 'Mode rccl requires -EnableExperimentalLlamaRccl; it is not a validated llama.cpp transport.'
}

& $Executable @ExecutableArguments
exit $LASTEXITCODE
