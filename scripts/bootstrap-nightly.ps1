<#
.SYNOPSIS
    Create the isolated ROCm 10 PyTorch/native-transport environment.

.DESCRIPTION
    Installs the exact packages recorded in pins/rocm10-vllm-v0.28.0.json
    into this repository's .venv. No global packages or engine source tree is
    changed. The historical filename of this script is retained for compatible
    automation; it no longer installs a nightly ROCm stack.
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
    [string]$GpuArch = 'gfx1201'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$PinPath = Join-Path $ProjectRoot 'pins\rocm10-vllm-v0.28.0.json'
$ProbeVenv = Join-Path $ProjectRoot '.venv'
$ProbePython = Join-Path $ProbeVenv 'Scripts\python.exe'
$Pins = Get-Content -Raw -LiteralPath $PinPath | ConvertFrom-Json
$Uv = (Get-Command uv -CommandType Application -ErrorAction Stop |
        Select-Object -First 1).Source

if (-not (Test-Path -LiteralPath $ProbePython)) {
    & $Uv venv --python $Pins.python $ProbeVenv
    if ($LASTEXITCODE -ne 0) {
        throw "uv failed to create $ProbeVenv."
    }
}

$BuildToolArgs = @(
    'pip', 'install',
    '--python', $ProbePython,
    "cmake==$($Pins.cmake)",
    "ninja==$($Pins.ninja)"
)
& $Uv @BuildToolArgs
if ($LASTEXITCODE -ne 0) {
    throw 'Installing the pinned native build tools failed.'
}

$RuntimeArgs = @(
    'pip', 'install',
    '--python', $ProbePython,
    '--extra-index-url', $Pins.pytorch_index_url,
    '--extra-index-url', $Pins.rocm_index_url,
    '--index-strategy', 'unsafe-best-match',
    "numpy==$($Pins.numpy)",
    "torch[device-$GpuArch]==$($Pins.torch)",
    "rocm[devel,device-$GpuArch]==$($Pins.rocm)"
)
& $Uv @RuntimeArgs
if ($LASTEXITCODE -ne 0) {
    throw 'The pinned AMD ROCm 10 installation failed.'
}

& $Uv pip install --python $ProbePython --no-deps --editable $ProjectRoot
if ($LASTEXITCODE -ne 0) {
    throw 'Installing the local transport package failed.'
}

$env:WAVMG_GPU_ARCH = $GpuArch
& $ProbePython (Join-Path $ProjectRoot 'probes\environment_probe.py')
if ($LASTEXITCODE -ne 0) {
    throw 'The environment probe failed.'
}
