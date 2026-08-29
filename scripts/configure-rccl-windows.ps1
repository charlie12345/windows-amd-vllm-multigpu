#Requires -Version 5.1

[CmdletBinding()]
param(
    [switch]$Fresh,
    [ValidateSet('Full', 'Vllm', 'Minimal')]
    [string]$FunctionProfile = 'Full',
    [ValidateSet(
        'gfx1030',
        'gfx1100', 'gfx1101', 'gfx1102', 'gfx1103',
        'gfx1150', 'gfx1151', 'gfx1152', 'gfx1153',
        'gfx1200', 'gfx1201'
    )]
    [string]$GpuArch = 'gfx1201'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$CMake = Join-Path $ProjectRoot '.venv\Scripts\cmake.exe'
$Source = Join-Path $ProjectRoot 'sandbox\rocm-systems\projects\rccl'
$Build = Join-Path $ProjectRoot 'build\rccl-windows'
$Toolchain = Join-Path $ProjectRoot 'cmake\windows-rocm-wheel.cmake'
$HipifyLauncher = (Join-Path $ProjectRoot 'tools\hipify-perl.cmd').Replace('\', '/')

$OnlyFunctions = switch ($FunctionProfile) {
    'Full' { '' }
    'Vllm' {
        'AllReduce RING SIMPLE Sum f16/f32/bf16|' +
        'AllGather RING SIMPLE Sum i8|' +
        'ReduceScatter RING SIMPLE Sum f16/f32/bf16|' +
        'Broadcast RING SIMPLE Sum i8|' +
        'SendRecv RING SIMPLE Sum i8'
    }
    'Minimal' { 'AllReduce RING SIMPLE Sum f32|SendRecv RING SIMPLE Sum i8' }
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw 'Run scripts\bootstrap-nightly.ps1 first.'
}
if (-not (Test-Path -LiteralPath (Join-Path $Source 'CMakeLists.txt'))) {
    throw 'The pinned rocm-systems checkout is missing under sandbox\rocm-systems.'
}
if ($Fresh -and (Test-Path -LiteralPath $Build)) {
    $resolvedBuild = (Resolve-Path -LiteralPath $Build).Path
    $expectedBuild = [IO.Path]::GetFullPath((Join-Path $ProjectRoot 'build\rccl-windows'))
    if ($resolvedBuild -ne $expectedBuild) {
        throw "Refusing to remove unexpected build directory: $resolvedBuild"
    }
    Remove-Item -LiteralPath $resolvedBuild -Recurse -Force
}

if ($env:ROCM_ROOT) {
    $RocmRoot = [IO.Path]::GetFullPath($env:ROCM_ROOT)
} else {
    $RocmRoot = (& $Python -m rocm_sdk path --root).Trim()
}
if (-not $RocmRoot) {
    throw 'Could not locate ROCm. Set ROCM_ROOT or install rocm[devel].'
}
$RocmRootForward = $RocmRoot.Replace('\', '/')
$ToolchainForward = $Toolchain.Replace('\', '/')
$env:ROCM_PATH = $RocmRootForward
$env:HIP_PATH = $RocmRootForward
$env:HIP_PLATFORM = 'amd'
$env:PATH = (Join-Path $RocmRoot 'bin') + ';' +
    (Join-Path $RocmRoot 'lib\llvm\bin') + ';' +
    (Join-Path $ProjectRoot 'tools') + ';' +
    'C:\Program Files\Git\cmd;' +
    'C:\Program Files\Git\usr\bin;' +
    (Join-Path $ProjectRoot '.venv\Scripts') + ';' +
    'C:\Program Files (x86)\Microsoft Visual Studio\Installer;' +
    'C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem;' +
    'C:\Windows\System32\WindowsPowerShell\v1.0'

$VsDevCmd = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat'
if (-not (Test-Path -LiteralPath $VsDevCmd)) {
    throw "Visual Studio build environment was not found at $VsDevCmd"
}

$Arguments = @(
    '-S', $Source,
    '-B', $Build,
    '-G', 'Ninja',
    "-DCMAKE_TOOLCHAIN_FILE=$ToolchainForward",
    "-DROCM_PATH=$RocmRootForward",
    "-DROCMCORE_PATH=$RocmRootForward",
    "-Dhipify-perl_executable=$HipifyLauncher",
    '-DEXPLICIT_ROCM_VERSION=10.0.0',
    '-DCMAKE_BUILD_TYPE=Release',
    "-DGPU_TARGETS=$GpuArch",
    '-DBUILD_SHARED_LIBS=ON',
    '-DBUILD_TESTS=OFF',
    '-DBUILD_PROFILER_INSPECTOR=OFF',
    '-DBUILD_PLUGIN_EXAMPLES=OFF',
    '-DBUILD_NCCL4PY=OFF',
    '-DENABLE_DEVICE_LINKER=OFF',
    '-DDISABLE_KERNARG_PRELOAD=OFF',
    "-DONLY_FUNCS=$OnlyFunctions",
    '-DGENERATE_SYM_KERNELS=OFF',
    '-DENABLE_ROCSHMEM=OFF',
    '-DENABLE_ROCSHMEM_GIN=OFF',
    '-DRCCL_HIP_FATBIN_AT_TAIL=OFF',
    '-DRCCL_ROCPROFILER_REGISTER=OFF',
    '-DROCTX=OFF',
    '-DFAULT_INJECTION=OFF',
    '-DENABLE_COMPRESS=OFF'
)

$QuotedArguments = $Arguments | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' }
$Command = 'call "' + $VsDevCmd + '" -arch=x64 -host_arch=x64 >nul && ' +
    '"' + $CMake + '" ' + ($QuotedArguments -join ' ')
& $env:COMSPEC /d /s /c $Command
exit $LASTEXITCODE
