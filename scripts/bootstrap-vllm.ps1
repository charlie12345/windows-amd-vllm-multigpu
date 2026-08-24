param(
    [ValidateRange(1, 256)]
    [int]$MaxJobs = 16,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Pins = Get-Content (Join-Path $ProjectRoot "pins\nightly-2026-07-28.json") |
    ConvertFrom-Json
$VllmRoot = Join-Path $ProjectRoot "sandbox\vllm"
$Venv = Join-Path $ProjectRoot ".venv-vllm"
$Python = Join-Path $Venv "Scripts\python.exe"
$VenvJunction = Join-Path $VllmRoot ".venv211"
$TransportPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path (Join-Path $VllmRoot ".git"))) {
    New-Item -ItemType Directory -Force (Split-Path -Parent $VllmRoot) | Out-Null
    git clone --branch $Pins.vllm_branch --single-branch $Pins.vllm_fork $VllmRoot
    if ($LASTEXITCODE -ne 0) { throw "vLLM clone failed" }
}

$ActualCommit = (git -C $VllmRoot rev-parse HEAD).Trim()
if ($ActualCommit -ne $Pins.vllm_commit) {
    throw "Sandbox vLLM is $ActualCommit; expected pinned $($Pins.vllm_commit)"
}

& (Join-Path $PSScriptRoot "apply-vllm-patches.ps1")

if (-not (Test-Path $Python)) {
    uv venv --python $Pins.python $Venv
    if ($LASTEXITCODE -ne 0) { throw "vLLM virtual environment creation failed" }
}

uv pip install --python $Python `
    "cmake==$($Pins.cmake)" "ninja==$($Pins.ninja)" `
    "setuptools>=77,<80" "setuptools-scm>=8" "setuptools-rust>=1.9" `
    wheel "jinja2>=3.1.6" packaging
if ($LASTEXITCODE -ne 0) { throw "vLLM build dependency installation failed" }

uv pip install --python $Python `
    --index-url $Pins.index_url `
    --prerelease allow `
    --index-strategy unsafe-best-match `
    "torch==$($Pins.torch)" `
    "torchvision==$($Pins.torchvision)" `
    "rocm[$($Pins.rocm_extra)]==$($Pins.rocm)" `
    $Pins.amd_torch_device
if ($LASTEXITCODE -ne 0) { throw "AMD nightly installation failed" }

uv pip install --python $Python "triton-windows==3.6.0.post26" winloop
if ($LASTEXITCODE -ne 0) { throw "Triton installation failed" }
uv pip install --python $Python -r (Join-Path $VllmRoot "requirements\common.txt")
if ($LASTEXITCODE -ne 0) { throw "vLLM Python dependency installation failed" }

if (-not (Test-Path $VenvJunction)) {
    New-Item -ItemType Junction -Path $VenvJunction -Target $Venv | Out-Null
}
$ResolvedJunction = (Get-Item $VenvJunction).Target
if ((Resolve-Path $ResolvedJunction).Path -ne (Resolve-Path $Venv).Path) {
    throw "$VenvJunction does not point to $Venv"
}

if (-not (Test-Path $TransportPython)) {
    & (Join-Path $PSScriptRoot "bootstrap-nightly.ps1")
    if ($LASTEXITCODE -ne 0) { throw "transport environment bootstrap failed" }
}
& (Join-Path $PSScriptRoot "build-native.cmd")
if ($LASTEXITCODE -ne 0) { throw "transport native-kernel build failed" }

if (-not $SkipBuild) {
    $PreviousRoot = $env:VLLM_ROOT
    $PreviousJobs = $env:MAX_JOBS
    try {
        $env:VLLM_ROOT = $VllmRoot + "\"
        $env:MAX_JOBS = [string]$MaxJobs
        & (Join-Path $VllmRoot "build_windows_rocm.cmd")
        if ($LASTEXITCODE -ne 0) { throw "vLLM native extension build failed" }
        & (Join-Path $VllmRoot "install_windows_rocm.cmd")
        if ($LASTEXITCODE -ne 0) { throw "vLLM editable install failed" }
    }
    finally {
        $env:VLLM_ROOT = $PreviousRoot
        $env:MAX_JOBS = $PreviousJobs
    }
}

uv pip install --python $Python -e $ProjectRoot --no-build-isolation
if ($LASTEXITCODE -ne 0) { throw "communicator plugin installation failed" }

$env:WAVMG_ENABLE = "1"
$env:VLLM_PLUGINS = "windows_amd_multigpu"
& $Python -c @'
import json
import torch
import vllm
from vllm.platforms import current_platform
print(json.dumps({
    "vllm": vllm.__version__,
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "platform": type(current_platform).__name__,
    "communicator": current_platform.get_device_communicator_cls(),
}, indent=2))
'@
if ($LASTEXITCODE -ne 0) { throw "installed vLLM/plugin verification failed" }
