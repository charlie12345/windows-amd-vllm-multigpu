#Requires -Version 5.1

[CmdletBinding()]
param(
    [ValidateRange(1, 64)]
    [int]$Jobs = 4,
    [switch]$Fresh,
    [switch]$ConfigureOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Pins = Get-Content -Raw -LiteralPath (
    Join-Path $ProjectRoot 'pins\rocm10-vllm-v0.28.0.json'
) | ConvertFrom-Json
$SourceRoot = Join-Path $ProjectRoot 'sandbox\rocm-systems-hip-p2p'
$Source = Join-Path $SourceRoot 'projects\clr'
$Build = Join-Path $ProjectRoot 'build\hip-p2p-runtime'
$Install = Join-Path $Build 'install'
$Python = Join-Path $ProjectRoot '.venv-vllm\Scripts\python.exe'
$CMake = Join-Path $ProjectRoot '.venv\Scripts\cmake.exe'
$VsDevCmd = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat'

foreach ($RequiredFile in @(
    $Python,
    $CMake,
    $VsDevCmd,
    (Join-Path $Source 'CMakeLists.txt'),
    (Join-Path $SourceRoot 'shared\amdgpu-windows-interop\pal\lib\Release\x64\pal.lib')
)) {
    if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
        throw "Required HIP/PAL build input is missing: $RequiredFile"
    }
}

$Current = (& git -C $SourceRoot rev-parse HEAD).Trim()
if ($Current -ne $Pins.hip_runtime_rocm_systems_commit) {
    throw "HIP runtime build requires $($Pins.hip_runtime_rocm_systems_commit); found $Current."
}
$Patch = Join-Path $ProjectRoot 'patches\hip-p2p\44be71b-force-pal-peer-probe.patch'
& git -C $SourceRoot apply --reverse --check $Patch 2>$null
if ($LASTEXITCODE -ne 0) {
    throw 'Run scripts\apply-hip-p2p-patches.ps1 before building.'
}

if ($Fresh -and (Test-Path -LiteralPath $Build)) {
    $ResolvedBuild = (Resolve-Path -LiteralPath $Build).Path
    $ExpectedBuild = [IO.Path]::GetFullPath((Join-Path $ProjectRoot 'build\hip-p2p-runtime'))
    if ($ResolvedBuild -ne $ExpectedBuild) {
        throw "Refusing to remove unexpected build directory: $ResolvedBuild"
    }
    Remove-Item -LiteralPath $ResolvedBuild -Recurse -Force
}

$RocmRoot = (& $Python -m rocm_sdk path --root).Trim()
if (-not $RocmRoot) {
    throw 'Could not locate the pinned ROCm SDK wheel root.'
}
$RocmRootForward = $RocmRoot.Replace('\', '/')
$env:ROCM_PATH = $RocmRoot
$env:HIP_PATH = $RocmRoot
$env:HIP_PLATFORM = 'amd'
$env:HIP_DEVICE_LIB_PATH = Join-Path $RocmRoot 'lib\llvm\amdgcn\bitcode'
$env:CMAKE_BUILD_PARALLEL_LEVEL = [string]$Jobs
$env:PATH = (Join-Path $RocmRoot 'bin') + ';' +
    (Join-Path $RocmRoot 'lib\llvm\bin') + ';' +
    (Join-Path $ProjectRoot '.venv\Scripts') + ';' + $env:PATH

$Arguments = @(
    '-S', $Source,
    '-B', $Build,
    '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release',
    '-DCLR_BUILD_HIP=ON',
    '-DCLR_BUILD_OCL=OFF',
    "-DHIP_COMMON_DIR=$((Join-Path $SourceRoot 'projects\hip').Replace('\', '/'))",
    "-DHIPCC_BIN_DIR=$((Join-Path $RocmRoot 'bin').Replace('\', '/'))",
    "-DROCM_PATH=$RocmRootForward",
    "-DCMAKE_PREFIX_PATH=$RocmRootForward",
    "-DCMAKE_INSTALL_PREFIX=$($Install.Replace('\', '/'))",
    '-D__HIP_ENABLE_PCH=OFF',
    '-DROCCLR_ENABLE_HSA=ON',
    '-DROCCLR_ENABLE_PAL=ON',
    '-D__HIP_ENABLE_RTC=ON',
    '-DUSE_PROF_API=OFF',
    '-DROCR_DLL_LOAD=OFF',
    "-DAMD_COMPUTE_WIN=$((Join-Path $SourceRoot 'shared\amdgpu-windows-interop').Replace('\', '/'))",
    "-DPython3_EXECUTABLE=$($Python.Replace('\', '/'))"
)
$QuotedArguments = $Arguments | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' }
$ConfigureCommand = 'call "' + $VsDevCmd + '" -arch=x64 -host_arch=x64 >nul && ' +
    '"' + $CMake + '" ' + ($QuotedArguments -join ' ')
& $env:COMSPEC /d /s /c $ConfigureCommand
if ($LASTEXITCODE -ne 0) {
    throw 'Configuring the experimental HIP/PAL runtime failed.'
}
if ($ConfigureOnly) {
    exit 0
}

$BuildCommand = 'call "' + $VsDevCmd + '" -arch=x64 -host_arch=x64 >nul && ' +
    '"' + $CMake + '" --build "' + $Build + '" --config Release --target install --parallel ' + $Jobs
& $env:COMSPEC /d /s /c $BuildCommand
if ($LASTEXITCODE -ne 0) {
    throw 'Building the experimental HIP/PAL runtime failed.'
}

$Runtime = Join-Path $Install 'bin\amdhip64_7.dll'
if (-not (Test-Path -LiteralPath $Runtime -PathType Leaf)) {
    throw "The build completed without the expected runtime: $Runtime"
}
Write-Host "Experimental HIP runtime built at $Runtime" -ForegroundColor Green
