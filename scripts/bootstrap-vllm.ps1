<#
.SYNOPSIS
    Build the pinned Windows AMD vLLM host and install this external plugin.

.DESCRIPTION
    Clones the exact charlie12345/vLLM_for_AMD revision, requires a clean
    source tree, delegates the ROCm 10 build to that fork's guarded setup
    script, then installs this package into the same Python environment.
    No vLLM source patch or file overlay is applied.
#>
#Requires -Version 5.1

[CmdletBinding()]
param(
    [ValidateSet(
        'gfx1030',
        'gfx1100', 'gfx1101', 'gfx1102', 'gfx1103',
        'gfx1150', 'gfx1151', 'gfx1152', 'gfx1153',
        'gfx1200', 'gfx1201'
    )]
    [string]$GpuArch = 'gfx1201',

    [ValidateRange(1, 256)]
    [int]$MaxJobs = 8,

    [switch]$SkipBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$PinPath = Join-Path $ProjectRoot 'pins\rocm10-vllm-v0.28.0.json'
$Pins = Get-Content -Raw -LiteralPath $PinPath | ConvertFrom-Json
$VllmRoot = Join-Path $ProjectRoot 'sandbox\vllm'
$Venv = Join-Path $ProjectRoot '.venv-vllm'
$Python = Join-Path $Venv 'Scripts\python.exe'
$TransportPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$Uv = (Get-Command uv -CommandType Application -ErrorAction Stop |
        Select-Object -First 1).Source

$NewClone = $false
if (-not (Test-Path -LiteralPath (Join-Path $VllmRoot '.git'))) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $VllmRoot) |
        Out-Null
    & git clone --filter=blob:none --no-checkout $Pins.vllm_fork $VllmRoot
    if ($LASTEXITCODE -ne 0) {
        throw 'Cloning the Windows AMD vLLM fork failed.'
    }
    $NewClone = $true
}

if (-not $NewClone) {
    $TrackedChanges = @(& git -C $VllmRoot status --porcelain --untracked-files=no)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not inspect $VllmRoot."
    }
    if ($TrackedChanges.Count -gt 0) {
        $ChangeList = $TrackedChanges -join [Environment]::NewLine
        throw @"
Refusing to update a vLLM checkout with tracked changes:
$ChangeList
Preserve those changes on their own branch and use a clean checkout.
"@
    }
}

& git -C $VllmRoot fetch --depth 1 origin $Pins.vllm_commit
if ($LASTEXITCODE -ne 0) {
    throw "Fetching pinned vLLM commit $($Pins.vllm_commit) failed."
}
& git -C $VllmRoot checkout --detach $Pins.vllm_commit
if ($LASTEXITCODE -ne 0) {
    throw 'Checking out the pinned vLLM commit failed.'
}

& (Join-Path $PSScriptRoot 'verify-vllm-host.ps1') -VllmRoot $VllmRoot

if (-not $SkipBuild) {
    $PreviousVenv = [Environment]::GetEnvironmentVariable('VLLM_VENV', 'Process')
    try {
        $env:VLLM_VENV = $Venv
        $SetupArgs = @{
            GpuArch = $GpuArch
            MaxJobs = $MaxJobs
        }
        & (Join-Path $VllmRoot 'setup_windows_rocm.ps1') @SetupArgs
        if ($LASTEXITCODE -ne 0) {
            throw 'The guarded Windows AMD vLLM build failed.'
        }
    }
    finally {
        [Environment]::SetEnvironmentVariable(
            'VLLM_VENV',
            $PreviousVenv,
            'Process'
        )
    }
}
elseif (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "-SkipBuild requires an existing vLLM environment at $Venv."
}

if (-not (Test-Path -LiteralPath $TransportPython -PathType Leaf)) {
    & (Join-Path $PSScriptRoot 'bootstrap-nightly.ps1') -GpuArch $GpuArch
    if ($LASTEXITCODE -ne 0) {
        throw 'The transport environment bootstrap failed.'
    }
}

$PreviousArch = [Environment]::GetEnvironmentVariable('WAVMG_GPU_ARCH', 'Process')
try {
    $env:WAVMG_GPU_ARCH = $GpuArch
    & (Join-Path $PSScriptRoot 'build-native.cmd')
    if ($LASTEXITCODE -ne 0) {
        throw 'The transport native-kernel build failed.'
    }
}
finally {
    [Environment]::SetEnvironmentVariable(
        'WAVMG_GPU_ARCH',
        $PreviousArch,
        'Process'
    )
}

& $Uv pip install --python $Python --editable $ProjectRoot --no-build-isolation --no-deps
if ($LASTEXITCODE -ne 0) {
    throw 'Installing the communicator plugin failed.'
}

$env:WAVMG_ENABLE = '1'
$env:VLLM_PLUGINS = 'windows_amd_multigpu'
& $Python -c @'
import json
import torch
import vllm
from importlib.metadata import entry_points
from vllm.platforms import current_platform

plugins = {
    entry.name: entry.value
    for entry in entry_points(group="vllm.platform_plugins")
}
if "windows_amd_multigpu" not in plugins:
    raise SystemExit("windows_amd_multigpu entry point is missing")

print(json.dumps({
    "vllm": vllm.__version__,
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "platform": type(current_platform).__name__,
    "communicator": current_platform.get_device_communicator_cls(),
    "plugin": plugins["windows_amd_multigpu"],
}, indent=2))
'@
if ($LASTEXITCODE -ne 0) {
    throw 'The installed vLLM/plugin verification failed.'
}

Write-Host 'Pinned Windows AMD vLLM and external plugin are ready.' -ForegroundColor Green
Write-Host "vLLM source: $VllmRoot"
Write-Host "Python:      $Python"
