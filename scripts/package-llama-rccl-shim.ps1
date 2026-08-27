#Requires -Version 5.1

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$InstallDirectory,

    [string]$Version = '0.2.0-rc1',
    [string]$RocmLabel = 'rocm10.0',
    [string]$GpuArchitecture = 'gfx1201',
    [string]$OutputDirectory,
    [switch]$AllowUnvalidated
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$InstallDirectory = (Resolve-Path -LiteralPath $InstallDirectory).Path
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $ProjectRoot 'dist'
}
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)

foreach ($RequiredRelativePath in @(
    'bin\rccl.dll',
    'bin\rccl-real.dll',
    'lib\rccl.lib',
    'lib\cmake\rccl\rccl-config.cmake',
    'include\nccl.h',
    'run-with-llama-plugin.ps1',
    'PROVENANCE.json',
    'docs\install-llama-rocmfpx.md',
    'licenses\LICENSE',
    'licenses\NOTICE',
    'licenses\LICENSES\RCCL-UPSTREAM-LICENSE.txt'
)) {
    $RequiredPath = Join-Path $InstallDirectory $RequiredRelativePath
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Release input is incomplete: $RequiredPath"
    }
}

$Provenance = Get-Content -Raw -LiteralPath (Join-Path $InstallDirectory 'PROVENANCE.json') |
    ConvertFrom-Json
if ($Provenance.repository_dirty -and -not $AllowUnvalidated) {
    throw 'Refusing to package a build produced from a dirty repository. Rebuild from the reviewed release commit.'
}
if ($Provenance.runtime_validation -ne 'passed' -and -not $AllowUnvalidated) {
    throw 'Refusing to package a runtime-unvalidated build. Pass -AllowUnvalidated only for a clearly labeled prerelease.'
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$ArchiveBase = "wavmg-llama-shim-$Version-$RocmLabel-$GpuArchitecture"
$ArchivePath = Join-Path $OutputDirectory "$ArchiveBase.zip"
if (Test-Path -LiteralPath $ArchivePath) {
    throw "Refusing to overwrite existing release archive: $ArchivePath"
}

$TemporaryRoot = Join-Path ([IO.Path]::GetTempPath()) (
    'wavmg-release-' + [Guid]::NewGuid().ToString('N')
)
$StagedRoot = Join-Path $TemporaryRoot $ArchiveBase
try {
    New-Item -ItemType Directory -Force -Path $StagedRoot | Out-Null
    Get-ChildItem -LiteralPath $InstallDirectory -Force |
        Copy-Item -Destination $StagedRoot -Recurse -Force
    Compress-Archive -LiteralPath $StagedRoot -DestinationPath $ArchivePath -CompressionLevel Optimal
} finally {
    if (Test-Path -LiteralPath $TemporaryRoot) {
        $ResolvedTemporaryRoot = (Resolve-Path -LiteralPath $TemporaryRoot).Path
        $ExpectedTempPrefix = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
        if (-not $ResolvedTemporaryRoot.StartsWith($ExpectedTempPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to remove unexpected staging directory: $ResolvedTemporaryRoot"
        }
        Remove-Item -LiteralPath $ResolvedTemporaryRoot -Recurse -Force
    }
}

$ArchiveHash = Get-FileHash -Algorithm SHA256 -LiteralPath $ArchivePath
$ChecksumPath = "$ArchivePath.sha256"
('{0}  {1}' -f $ArchiveHash.Hash.ToLowerInvariant(), (Split-Path -Leaf $ArchivePath)) |
    Set-Content -LiteralPath $ChecksumPath -Encoding ascii

Write-Host "Created $ArchivePath" -ForegroundColor Green
Write-Host "Created $ChecksumPath" -ForegroundColor Green
