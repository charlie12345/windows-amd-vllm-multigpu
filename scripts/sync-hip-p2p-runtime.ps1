#Requires -Version 5.1

[CmdletBinding()]
param(
    [switch]$AcceptAmdInteropEula
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Pins = Get-Content -Raw -LiteralPath (
    Join-Path $ProjectRoot 'pins\nightly-2026-07-28.json'
) | ConvertFrom-Json
$Checkout = Join-Path $ProjectRoot 'sandbox\rocm-systems-hip-p2p'
$ToolsEnvironment = Join-Path $ProjectRoot 'build\hip-p2p-tools'
$Python = Join-Path $ToolsEnvironment 'Scripts\python.exe'
$Dvc = Join-Path $ToolsEnvironment 'Scripts\dvc.exe'

if (-not (Test-Path -LiteralPath (Join-Path $Checkout '.git'))) {
    New-Item -ItemType Directory -Force -Path (Split-Path $Checkout) | Out-Null
    & git clone --filter=blob:none --no-checkout $Pins.rocm_systems_repo $Checkout
    if ($LASTEXITCODE -ne 0) {
        throw 'Cloning rocm-systems failed.'
    }
}

$Dirty = @(& git -C $Checkout status --porcelain)
$Current = (& git -C $Checkout rev-parse HEAD 2>$null).Trim()
if (($Dirty.Count -gt 0) -and ($Current -ne $Pins.hip_runtime_rocm_systems_commit)) {
    throw "Refusing to change a dirty checkout at $Checkout."
}
if (($Dirty.Count -eq 0) -and ($Current -ne $Pins.hip_runtime_rocm_systems_commit)) {
    & git -C $Checkout fetch --depth 1 origin $Pins.hip_runtime_rocm_systems_commit
    if ($LASTEXITCODE -ne 0) {
        throw 'Fetching the pinned HIP runtime source failed.'
    }
    & git -C $Checkout checkout --detach $Pins.hip_runtime_rocm_systems_commit
    if ($LASTEXITCODE -ne 0) {
        throw 'Checking out the pinned HIP runtime source failed.'
    }
}

& git -C $Checkout sparse-checkout init --cone
& git -C $Checkout sparse-checkout set `
    .dvc `
    projects/clr `
    projects/hip `
    projects/rocr-runtime `
    shared/amdgpu-windows-interop
if ($LASTEXITCODE -ne 0) {
    throw 'Configuring the sparse HIP runtime checkout failed.'
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    py -3.12 -m venv $ToolsEnvironment
    if ($LASTEXITCODE -ne 0) {
        throw 'Creating the isolated DVC environment failed.'
    }
}
if (-not (Test-Path -LiteralPath $Dvc -PathType Leaf)) {
    & $Python -m pip install --disable-pip-version-check 'dvc[s3]'
    if ($LASTEXITCODE -ne 0) {
        throw 'Installing DVC into the isolated tools environment failed.'
    }
}

$InteropEula = Join-Path $Checkout 'shared\amdgpu-windows-interop\LICENSE'
if (-not (Test-Path -LiteralPath $InteropEula -PathType Leaf)) {
    throw "The AMD Windows interop EULA was not found at $InteropEula"
}
if (-not $AcceptAmdInteropEula) {
    throw (
        "Review $InteropEula, then rerun with -AcceptAmdInteropEula to download " +
        'the EULA-governed prebuilt PAL/WKMI objects. The objects are never added to this repository.'
    )
}

& $Dvc --cd $Checkout pull
if ($LASTEXITCODE -ne 0) {
    throw 'Downloading the pinned public Windows interop libraries failed.'
}

Write-Host "Pinned HIP/PAL source and DVC objects are ready at $Checkout" -ForegroundColor Green
