#Requires -Version 5.1

[CmdletBinding()]
param(
    [ValidateRange(1, 64)]
    [int]$Jobs = 4,
    [string]$Target = 'rccl'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$CMake = Join-Path $ProjectRoot '.venv\Scripts\cmake.exe'
$Build = Join-Path $ProjectRoot 'build\rccl-windows'

if (-not (Test-Path -LiteralPath (Join-Path $Build 'build.ninja'))) {
    throw 'Run scripts\configure-rccl-windows.ps1 first.'
}

$RocmRoot = (& $Python -m rocm_sdk path --root).Trim()
if (-not $RocmRoot) {
    throw 'Could not locate the ROCm SDK wheel root.'
}
$env:ROCM_PATH = $RocmRoot.Replace('\', '/')
$env:HIP_PATH = $env:ROCM_PATH
$env:HIP_PLATFORM = 'amd'
$env:PATH = (Join-Path $RocmRoot 'bin') + ';' +
    (Join-Path $RocmRoot 'lib\llvm\bin') + ';' +
    (Join-Path $ProjectRoot 'tools') + ';' +
    'C:\Program Files\Git\bin;' +
    (Join-Path $ProjectRoot '.venv\Scripts') + ';' + $env:PATH

$VsDevCmd = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat'
if (-not (Test-Path -LiteralPath $VsDevCmd)) {
    throw "Visual Studio build environment was not found at $VsDevCmd"
}

$Command = 'call "' + $VsDevCmd + '" -arch=x64 -host_arch=x64 >nul && ' +
    '"' + $CMake + '" --build "' + $Build + '" --target "' + $Target + '" --parallel ' + $Jobs
& $env:COMSPEC /d /s /c $Command
exit $LASTEXITCODE
