# Install the bridge in Windows AMD vLLM

## Required host: the Windows AMD vLLM fork

This repository is the multi-GPU transport plugin. It does **not** make stock
upstream vLLM run on native Windows. The supported host is Carlo Pasquale's
[`charlie12345/vLLM_for_AMD`](https://github.com/charlie12345/vLLM_for_AMD)
Windows ROCm fork. The reproducible build below uses the branch and exact
commit recorded in `pins/rocm10-vllm-v0.28.0.json`; newer fork revisions must be
rebuilt and validated before being declared compatible.

> [!WARNING]
> This is development software. A failed GPU collective or driver fault can
> hang worker processes, leave VRAM charged to WDDM, or require a Windows
> restart. Save work in other GPU applications first. The launch and validation
> scripts fail closed when an AMD adapter is not healthy, but no user-mode
> watchdog can recover a wedged display driver or GPU firmware.

There are therefore three separate pieces:

1. the Windows AMD vLLM fork, which provides native-Windows vLLM and ROCm
   support;
2. this repository's Python platform/communicator plugin; and
3. this repository's full Windows RCCL and D3D12/HIP DLLs.

Install the Python plugin into the exact interpreter that launches the Windows
AMD vLLM build. Do not copy the communicator sources into the vLLM checkout,
and do not use the six-symbol llama RCCL shim with vLLM.

The vLLM adapter uses one worker process per GPU. Gloo remains the CPU control
plane, full native Windows RCCL handles AllGather, ReduceScatter, Broadcast,
and the RCCL-only fallback, and the D3D12 DLL handles eligible two-rank
AllReduce calls.

Do not replace the vLLM RCCL DLL with the llama shim. The shim implements only
the six symbols imported by the tested single-process llama backend.

## Reproducible source build

From a PowerShell prompt:

```powershell
git clone https://github.com/charlie12345/windows-amd-vllm-multigpu.git
Set-Location .\windows-amd-vllm-multigpu

.\scripts\bootstrap-nightly.ps1
# Optional: use an unpacked portable ROCm 10 SDK instead of the wheel SDK.
$env:ROCM_ROOT = 'G:\path\to\rocm-10-sdk'
.\scripts\build-native.cmd
.\scripts\sync-upstreams.ps1
.\scripts\apply-rccl-patches.ps1
.\scripts\configure-rccl-windows.ps1 -FunctionProfile Vllm
.\scripts\build-rccl-windows.ps1 -Jobs 8
.\scripts\run-rccl-validation.ps1 -SkipBuild
.\.venv\Scripts\python.exe .\probes\d3d12_cross_process_probe.py
.\.venv\Scripts\python.exe .\probes\d3d12_all_reduce_probe.py
.\scripts\bootstrap-vllm.ps1 -MaxJobs 8
```

The last script installs this repository as an editable vLLM platform plugin.
It checks out and builds the pinned Windows vLLM fork in `sandbox\vllm`. It
requires a clean tracked worktree and applies **zero vLLM patches**. Required
Windows host fixes live in `charlie12345/vLLM_for_AMD`; transport code remains
in this repository.

The two runtime binaries are:

- `build\rccl-windows\rccl.dll` — full native Windows RCCL;
- `build\native\wavmg_d3d12_v1.dll` — cross-process D3D12 AllReduce.

The D3D12 transport also requires `build\native\wavmg_hip_v1.dll`, which
contains the HIP publish and reduction kernels.

## Install into an existing Windows vLLM environment

Use this route only for an already working native-Windows installation of
`charlie12345/vLLM_for_AMD`. Build the native DLLs as above with the same ROCm
runtime generation used by that vLLM environment, then install the Python
package into that environment. The package registers the
`windows_amd_multigpu` platform entry point; it does not copy Python files into
the vLLM source tree.

```powershell
$bridgeRoot = (Resolve-Path .).Path
$vllmRoot = 'C:\path\to\vLLM_for_AMD'
$vllmPython = 'C:\path\to\the-vllm-environment\Scripts\python.exe'

# Confirm that this is the intended Windows AMD vLLM source and interpreter.
git -C $vllmRoot remote -v
git -C $vllmRoot rev-parse HEAD
.\scripts\verify-vllm-host.ps1 -VllmRoot $vllmRoot
& $vllmPython -c "import sys, torch, vllm; print(sys.executable); print(vllm.__file__); print(torch.__version__, torch.version.hip)"

# Install only the external bridge into that exact environment.
uv pip install --python $vllmPython --editable $bridgeRoot --no-build-isolation --no-deps

# Verify that vLLM can discover the plugin entry point.
& $vllmPython -c "from importlib.metadata import entry_points; print([(e.name, e.value) for e in entry_points(group='vllm.platform_plugins')])"
```

The final command must include `windows_amd_multigpu`. If `vllm.__file__`
points to another environment or the entry point is absent, stop: launching
from that interpreter will not use the plugin.

The validated vLLM fork revision is the `vllm_commit` value in the pin file.
An exact mismatch is not automatically broken, but it is unvalidated until the
native probes and deterministic TP2 test pass.

## Launch TP2

Close other GPU-heavy applications, then run the read-only health gate. Both
intended adapters must report status `OK` before any TP=2 process starts:

```powershell
.\scripts\assert-windows-amd-gpu-health.ps1 -RequiredCount 2
```

If it reports `CM_PROB_FAILED_ADD`, do not bypass it. Restart Windows; if the
error persists, cold-power-cycle and repair/reinstall the matching AMD driver.

Use the checked-in launcher:

```powershell
.\scripts\run-vllm-tp2.ps1 `
    -Model Qwen/Qwen3-0.6B `
    -TensorParallelSize 2 `
    -UseRccl $true `
    -UseD3D12 $true
```

The equivalent integration contract is:

```powershell
$env:HIP_VISIBLE_DEVICES = '0,1'
$env:CUDA_VISIBLE_DEVICES = '0,1'
$env:VLLM_WORKER_MULTIPROC_METHOD = 'spawn'
$env:WAVMG_ENABLE = '1'
$env:VLLM_PLUGINS = 'windows_amd_multigpu'
$env:WAVMG_USE_RCCL = '1'
$env:WAVMG_RCCL_DLL = 'C:\path\to\rccl.dll'
$env:WAVMG_USE_D3D12 = '1'
$env:WAVMG_D3D12_DLL = 'C:\path\to\wavmg_d3d12_v1.dll'
$env:WAVMG_HIP_DLL = 'C:\path\to\wavmg_hip_v1.dll'
$env:WAVMG_D3D12_MIN_BYTES = '32768'
$env:WAVMG_D3D12_MAX_BYTES = '67108864'
$env:WAVMG_TRACE_COLLECTIVES = '1'
```

Launch vLLM from `$vllmPython` (or the `vllm.exe` beside it), not from a global
Python installation. Use one engine with `--tensor-parallel-size 2` and
`--distributed-executor-backend mp`; do not start two unrelated servers.

For a TP=2 MoE configuration whose vLLM parallel config does not require
All-to-All, opt in separately:

```powershell
$env:WAVMG_ALLOW_TP_EXPERT_PARALLEL = '1'
```

This permits TP-local expert ownership only. The platform still rejects data
parallel, context-parallel, and every expert-parallel layout for which
`parallel_config.use_all2all` is true. The default remains fail-closed.

### Large-model Windows commit limit

Dedicated VRAM allocations can consume Windows commit backing under WDDM even
when the weights remain in VRAM. A 128 GiB RAM machine with only a 32 GiB
pagefile therefore does not necessarily have enough commit for 64 GiB of VRAM
allocations plus large CPU-resident weights.

This was observed with the 119.8 GiB Qwen3.8 Flash-Next W4A16/FP8-PLE
checkpoint. Its PLE tensors total 47.745 GiB. During TP=2/EP=2 construction,
physical RAM still had approximately 90 GiB available while Windows commit
headroom fell to 0.95 GiB. Continuing that load would risk a hard OOM.

On a machine with adequate free space on an NTFS secondary drive, configure a
larger pagefile from an elevated PowerShell window and reboot:

```powershell
.\scripts\configure-windows-pagefile.ps1 `
    -PagefileDrive G `
    -InitialSizeMiB 98304 `
    -MaximumSizeMiB 131072
```

The script retains a system-managed pagefile on C:, records the previous
settings under `build`, and supports
`.\scripts\configure-windows-pagefile.ps1 -RestoreAutomatic`. Use pagefile
capacity as commit backing, not as a substitute for measuring CPU-offload I/O;
heavy paging during decode would still make the configuration impractical.

Then start one vLLM engine with `--tensor-parallel-size 2` and
`--distributed-executor-backend mp`. This is one synchronized model, not two
unrelated servers.

Both ranks should report D3D12 only for eligible AllReduce payloads in the
configured range, RCCL fallback outside that range, and RCCL for the other
tensor collectives. The default lower bound keeps small decode reductions on
RCCL. Disable D3D12 first when isolating a failure; RCCL-only is the reference
path.

The D3D12 probe includes the exact 235,520-byte speculative-decode payload
that exposed the ROCm 10 `hipMemcpyAsync` external-pointer crash. Do not call a
ROCm 10 build validated unless that case and all three dtypes pass.

## Update policy

Keep the bridge repository separate from vLLM. Update
`charlie12345/vLLM_for_AMD` through its upstream-sync PR workflow and build it
in an isolated branch. Do not apply `patches/vllm`; that directory is retained
only as v0.27 development provenance. Rebuild, run the native probes, compare
deterministic TP2 output through RCCL-only and hybrid modes, then advance the
exact `vllm_commit` pin. Transport changes stay in this plugin; generally
useful Windows vLLM fixes belong in the host fork and should be proposed
upstream when portable.

## Other AMD architectures

The source build accepts `gfx1030`, `gfx1100`-`gfx1103`, `gfx1150`-`gfx1153`,
and `gfx1200`-`gfx1201`. Pass the same architecture to both bootstrap and RCCL
configuration, for example RX 7900 XTX (RDNA 3, `gfx1100`):

```powershell
.\scripts\bootstrap-nightly.ps1 -GpuArch gfx1100
$env:WAVMG_GPU_ARCH = 'gfx1100'
.\scripts\build-native.cmd
.\scripts\configure-rccl-windows.ps1 -FunctionProfile Vllm -GpuArch gfx1100
.\scripts\bootstrap-vllm.ps1 -GpuArch gfx1100 -MaxJobs 8
```

Only the dual-R9700 `gfx1201` configuration has passed end-to-end TP=2 tests.
The listed RDNA 2/3/3.5/4 targets are build-selectable, not validated
performance or compatibility claims.
