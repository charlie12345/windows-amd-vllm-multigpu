<#
.SYNOPSIS
    Create an isolated AMD Windows PyTorch/Gloo probe environment.

.DESCRIPTION
    Installs the exact packages recorded in pins/nightly-2026-07-28.json into
    this repository's .venv. No global packages or vLLM checkout are changed.
#>
#Requires -Version 5.1

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$PinPath = Join-Path $ProjectRoot 'pins\nightly-2026-07-28.json'
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

$Packages = @(
    "numpy==$($Pins.numpy)",
    "torch==$($Pins.torch)",
    "rocm[$($Pins.rocm_extra)]==$($Pins.rocm)",
    $Pins.amd_torch_device
)

& $Uv pip install `
    --python $ProbePython `
    --index-url $Pins.index_url `
    --index-strategy unsafe-best-match `
    @Packages
if ($LASTEXITCODE -ne 0) {
    throw 'The pinned AMD nightly installation failed.'
}

& $Uv pip install --python $ProbePython --no-deps --editable $ProjectRoot
if ($LASTEXITCODE -ne 0) {
    throw 'Installing the local transport package failed.'
}

& $ProbePython (Join-Path $ProjectRoot 'probes\environment_probe.py')
if ($LASTEXITCODE -ne 0) {
    throw 'The environment probe failed.'
}
