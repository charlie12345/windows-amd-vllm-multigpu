#Requires -Version 5.1

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Executable,

    [Parameter(Mandatory = $true)]
    [string]$RocmRoot,

    [ValidateSet('hybrid', 'd3d12', 'rccl')]
    [string]$Mode = 'hybrid',

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

$env:ROCM_PATH = $RocmRoot
$env:HIP_PATH = $RocmRoot
$env:HIP_PLATFORM = 'amd'
$env:HIP_DEVICE_LIB_PATH = Join-Path $RocmRoot 'lib\llvm\amdgcn\bitcode'
$env:GGML_CUDA_ALLREDUCE = 'nccl'
$env:WAC_MODE = $Mode
$env:PATH = "$PluginBin;$RocmBin;$RocmLlvmBin;$env:PATH"

& $Executable @ExecutableArguments
exit $LASTEXITCODE
