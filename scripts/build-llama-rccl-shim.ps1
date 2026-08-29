#Requires -Version 5.1

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RocmRoot,

    [Parameter(Mandatory = $true)]
    [string]$RcclDll,

    [Parameter(Mandatory = $true)]
    [string]$RcclIncludeDirectory,

    [string]$BuildDirectory,
    [string]$InstallDirectory,
    [string]$GpuArchitecture = 'gfx1201',
    [string]$CMakeExecutable,
    [string]$NinjaExecutable,
    [switch]$AllowDirty,

    [ValidateRange(1, 256)]
    [int]$Jobs = 8
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Source = Join-Path $ProjectRoot 'adapters\llama-rccl-shim'
$RepositoryStatus = @(& git -C $ProjectRoot status --porcelain --untracked-files=all)
$RepositoryDirty = $RepositoryStatus.Count -gt 0
if ($RepositoryDirty -and -not $AllowDirty) {
    throw 'Refusing a release-style build from a dirty repository. Commit/review the source or pass -AllowDirty for compile-only validation.'
}
if (-not $BuildDirectory) {
    $BuildDirectory = Join-Path $ProjectRoot 'build\llama-rccl-shim'
}
if (-not $InstallDirectory) {
    $InstallDirectory = Join-Path $ProjectRoot 'artifacts\llama-rccl-shim'
}

$RocmRoot = (Resolve-Path -LiteralPath $RocmRoot).Path
$RcclDll = (Resolve-Path -LiteralPath $RcclDll).Path
$RcclIncludeDirectory = (Resolve-Path -LiteralPath $RcclIncludeDirectory).Path
$BuildDirectory = [IO.Path]::GetFullPath($BuildDirectory)
$InstallDirectory = [IO.Path]::GetFullPath($InstallDirectory)

foreach ($RequiredPath in @(
    (Join-Path $RocmRoot 'bin'),
    (Join-Path $RocmRoot 'lib\llvm\bin\clang++.exe'),
    (Join-Path $RcclIncludeDirectory 'nccl.h')
)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required path does not exist: $RequiredPath"
    }
}

if (-not $CMakeExecutable) {
    $ProjectCMake = Join-Path $ProjectRoot '.venv\Scripts\cmake.exe'
    if (Test-Path -LiteralPath $ProjectCMake) {
        $CMakeExecutable = $ProjectCMake
    } else {
        $CMakeExecutable = (Get-Command cmake -ErrorAction Stop).Source
    }
}
if (-not $NinjaExecutable) {
    $ProjectNinja = Join-Path $ProjectRoot '.venv\Scripts\ninja.exe'
    if (Test-Path -LiteralPath $ProjectNinja) {
        $NinjaExecutable = $ProjectNinja
    } else {
        $NinjaExecutable = (Get-Command ninja -ErrorAction Stop).Source
    }
}
$CMakeExecutable = (Resolve-Path -LiteralPath $CMakeExecutable).Path
$NinjaExecutable = (Resolve-Path -LiteralPath $NinjaExecutable).Path
$env:ROCM_PATH = $RocmRoot
$env:HIP_PATH = $RocmRoot
$env:HIP_PLATFORM = 'amd'
$env:HIP_DEVICE_LIB_PATH = Join-Path $RocmRoot 'lib\llvm\amdgcn\bitcode'
$env:PATH = (Join-Path $RocmRoot 'bin') + ';' +
    (Join-Path $RocmRoot 'lib\llvm\bin') + ';' + $env:PATH

$ConfigureArguments = @(
    '-S', $Source,
    '-B', $BuildDirectory,
    '-G', 'Ninja',
    "-DCMAKE_MAKE_PROGRAM=$NinjaExecutable",
    '-DCMAKE_BUILD_TYPE=Release',
    "-DCMAKE_PREFIX_PATH=$RocmRoot",
    "-DCMAKE_HIP_COMPILER=$(Join-Path $RocmRoot 'lib\llvm\bin\clang++.exe')",
    "-DWAC_RCCL_DLL=$RcclDll",
    "-DWAC_RCCL_INCLUDE_DIR=$RcclIncludeDirectory",
    "-DWAC_HIP_ARCHITECTURES=$GpuArchitecture",
    "-DCMAKE_INSTALL_PREFIX=$InstallDirectory"
)

& $CMakeExecutable @ConfigureArguments
if ($LASTEXITCODE -ne 0) {
    throw 'Configuring the llama RCCL shim failed.'
}
& $CMakeExecutable --build $BuildDirectory --config Release --parallel $Jobs
if ($LASTEXITCODE -ne 0) {
    throw 'Building the llama RCCL shim failed.'
}
& $CMakeExecutable --install $BuildDirectory --config Release
if ($LASTEXITCODE -ne 0) {
    throw 'Installing the llama RCCL shim failed.'
}

$Commit = (& git -C $ProjectRoot rev-parse HEAD).Trim()
$Pins = Get-Content -Raw -LiteralPath (Join-Path $ProjectRoot 'pins\rocm10-vllm-v0.28.0.json') |
    ConvertFrom-Json
$RcclPatch = Join-Path $ProjectRoot 'patches\rccl\ee3bae9-windows-native.patch'
$RcclBuildCache = Join-Path (Split-Path -Parent $RcclDll) 'CMakeCache.txt'
$InputRcclRocmVersion = 'unknown'
if (Test-Path -LiteralPath $RcclBuildCache) {
    $VersionMatch = Select-String -LiteralPath $RcclBuildCache -Pattern '^EXPLICIT_ROCM_VERSION:[^=]+=(.+)$' |
        Select-Object -First 1
    if ($VersionMatch) {
        $InputRcclRocmVersion = $VersionMatch.Matches[0].Groups[1].Value
    }
}
$RocmVersionFile = Join-Path $RocmRoot '.info\version'
$RocmVersion = if (Test-Path -LiteralPath $RocmVersionFile) {
    (Get-Content -Raw -LiteralPath $RocmVersionFile).Trim()
} else {
    'unknown'
}
$InstalledShim = Join-Path $InstallDirectory 'bin\rccl.dll'
$InstalledRealRccl = Join-Path $InstallDirectory 'bin\rccl-real.dll'
$SourceHashes = [ordered]@{}
foreach ($SourceFile in @(
    'adapters\llama-rccl-shim\CMakeLists.txt',
    'adapters\llama-rccl-shim\cmake\rccl-config.cmake.in',
    'adapters\llama-rccl-shim\src\d3d12_transport.cu',
    'adapters\llama-rccl-shim\src\d3d12_transport.hpp',
    'adapters\llama-rccl-shim\src\rccl_shim.cpp',
    'adapters\llama-rccl-shim\src\rccl_shim.def'
)) {
    $SourcePath = Join-Path $ProjectRoot $SourceFile
    $SourceHashes[$SourceFile.Replace('\', '/')] =
        (Get-FileHash -Algorithm SHA256 -LiteralPath $SourcePath).Hash
}
$Provenance = [ordered]@{
    schema = 1
    component = 'wavmg-llama-rccl-shim'
    component_version = '0.2.0-rc2'
    repository_commit = $Commit
    repository_dirty = $RepositoryDirty
    shim_build_rocm_version = $RocmVersion
    shim_build_rocm_root = $RocmRoot
    gpu_architecture = $GpuArchitecture
    input_rccl_dll = $RcclDll
    input_rccl_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $RcclDll).Hash
    input_rccl_build_rocm_version = $InputRcclRocmVersion
    repository_rccl_pin = $Pins.rocm_systems_commit
    repository_rccl_version = $Pins.rccl_version
    repository_rccl_patch_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $RcclPatch).Hash
    adapter_source_sha256 = $SourceHashes
    shim_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $InstalledShim).Hash
    packaged_real_rccl_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $InstalledRealRccl).Hash
    runtime_validation = 'not-run-by-build-script'
}
$Provenance | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath (Join-Path $InstallDirectory 'PROVENANCE.json') -Encoding utf8

Write-Host "Installed llama RCCL shim: $InstallDirectory" -ForegroundColor Green
Write-Host 'Runtime validation is still required before publishing a stable binary.' -ForegroundColor Yellow
