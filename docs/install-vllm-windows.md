# Install the bridge in Windows AMD vLLM

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
.\scripts\build-native.cmd
.\scripts\sync-upstreams.ps1
.\scripts\apply-rccl-patches.ps1
.\scripts\configure-rccl-windows.ps1 -FunctionProfile Vllm
.\scripts\build-rccl-windows.ps1 -Jobs 8
.\scripts\run-rccl-validation.ps1 -SkipBuild
.\.venv\Scripts\python.exe .\probes\d3d12_cross_process_probe.py
.\.venv\Scripts\python.exe .\probes\d3d12_all_reduce_probe.py
.\scripts\bootstrap-vllm.ps1 -MaxJobs 16
```

The last script installs this repository as an editable vLLM platform plugin.
It builds the pinned Windows vLLM fork in `sandbox\vllm`; it does not inject
the communicator sources into that checkout.

The two runtime binaries are:

- `build\rccl-windows\rccl.dll` — full native Windows RCCL;
- `build\native\wavmg_d3d12_v1.dll` — cross-process D3D12 AllReduce.

## Launch TP2

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
```

Then start one vLLM engine with `--tensor-parallel-size 2` and
`--distributed-executor-backend mp`. This is one synchronized model, not two
unrelated servers.

Set `WAVMG_TRACE_COLLECTIVES=1` for routing markers. Both ranks should report
D3D12 for eligible AllReduce payloads and RCCL for the other tensor
collectives. Disable D3D12 first when isolating a failure; RCCL-only is the
reference path.

## Update policy

Keep the bridge repository separate from vLLM. Update the pinned vLLM fork in
an isolated branch, reapply only `patches/vllm`, rebuild, and run the native
probes plus a deterministic TP2 model test before advancing the pin. Transport
changes should stay in this plugin unless the change is a general upstream
vLLM Windows fix.
