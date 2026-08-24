# Windows AMD vLLM Multi-GPU

Experimental native-Windows RCCL collectives and tensor parallelism for AMD
GPUs and vLLM. This is an independent repository: it does not modify or build
inside the Windows AMD vLLM port at `C:\AI\vllm`.

## Milestone reached

The reference machine now builds native Windows RCCL 2.30.7 and runs a real
vLLM tensor-parallel model across two Radeon AI PRO R9700 GPUs and two worker
processes. The port passed exact FP16, FP32, and BF16 AllReduce; AllGather;
ReduceScatter; Broadcast; a 100-iteration 1,048,576-element stress test; and
non-default PyTorch HIP-stream tests. Qwen3-0.6B then produced exactly the same
token IDs through RCCL and the hybrid fast path as the known-good TP2 run.

The default hybrid uses D3D12 cross-adapter heaps/fences for its hot two-rank
AllReduce and native RCCL for the remaining collective API. With an 8-token
input, one generated token, 10 warmups, and 30 measured iterations:

| Configuration | Average latency | Relative to TP1 |
| --- | ---: | ---: |
| TP1, one R9700 | 77.75 ms | baseline |
| TP2, RCCL socket transport | 61.82 ms | 1.26x faster |
| TP2, D3D12 AllReduce + RCCL | 59.00 ms | 1.32x faster |

This is true model sharding and concurrent GPU execution. It is not direct
VRAM-to-VRAM P2P: the pinned Windows HIP driver reports no peer access and
rejects cross-device HIP IPC, so RCCL selects NET/Socket through system memory.
D3D12 does expose coherent cross-adapter heaps and GPU fences, but the D3D12
specification requires those heaps to reside in L0/system memory on discrete
adapters. The hybrid moves payloads with GPU copy engines and reduces in local
VRAM; the CPU does not copy payload data. Gloo is rendezvous/control.

Above-4G decoding, Re-Size BAR, and CSM-off are the correct firmware settings,
but firmware cannot make a Windows driver expose `hipDeviceCanAccessPeer`.
The repository therefore does not label the present transport GPU-direct.

## Reproduce native Windows RCCL

From a PowerShell prompt:

```powershell
.\scripts\bootstrap-nightly.ps1
.\scripts\build-native.cmd
.\scripts\sync-upstreams.ps1
.\scripts\apply-rccl-patches.ps1
.\scripts\configure-rccl-windows.ps1 -FunctionProfile Vllm
.\scripts\build-rccl-windows.ps1 -Jobs 8
.\scripts\run-rccl-validation.ps1 -SkipBuild
.\.venv\Scripts\python.exe .\probes\d3d12_cross_process_probe.py
.\.venv\Scripts\python.exe .\probes\d3d12_all_reduce_probe.py
```

`Vllm` builds the operations and dtypes exercised by the adapter. `Full`, the
configure script's default, builds RCCL's complete generated function table;
`Minimal` is intended only for compiler bring-up. The build output is
`build\rccl-windows\rccl.dll`. It is intentionally not committed or bundled in
the Python wheel.

All upstream sources, virtual environments, logs, and build output are kept in
ignored directories inside this repository. The RCCL and HIPIFY commits and
the ROCm wheel train are exact pins, and the Windows RCCL changes are stored as
a patch that is checked before application.

## Reproduce the mapped-host fallback

```powershell
.\scripts\bootstrap-nightly.ps1
.\scripts\build-native.cmd
.\scripts\run-probes.ps1
```

## Reproduce vLLM TP2

The vLLM bootstrap creates an isolated `.venv-vllm`, clones the pinned Windows
AMD vLLM fork into ignored `sandbox\vllm`, applies the version-pinned Windows
compatibility patch, builds it, and installs this plugin:

```powershell
.\scripts\bootstrap-vllm.ps1
.\scripts\run-vllm-tp2.ps1 -UseRccl $true -UseD3D12 $true
```

Compare one and two GPUs with:

```powershell
.\scripts\run-vllm-tp2.ps1 -TensorParallelSize 1 -UseRccl $true -UseD3D12 $true
.\scripts\run-vllm-tp2.ps1 -TensorParallelSize 2 -UseRccl $true -UseD3D12 $true
```

The launch script selects the hybrid D3D12 + RCCL path by default. Pass
`-UseD3D12 $false` for RCCL-only, or pass both `-UseD3D12 $false` and
`-UseRccl $false` for the mapped-host fallback. The validated vLLM source commit is
recorded in `pins/nightly-2026-07-28.json`. The adapter deliberately rejects
unsupported layouts: TP is currently limited to 1 or 2, while PP, DP, DCP,
PCP, and EP must remain 1 or disabled.

The fork's Windows installer intentionally omits ROCm's Linux-only and optional
feature packages. The bootstrap installs the dependency set exercised by the
validated text-generation path; quantization, model streaming, speculative
decoding, multimodal models, and other optional vLLM features are not yet part
of this compatibility claim.

## Repository layout

- `native/` contains the HIP mapped-memory and D3D12 cross-adapter DLLs.
- `src/` contains the RCCL wrapper, fallback transport, and vLLM plugin.
- `probes/` contains correctness, capability, performance, and model tests.
- `scripts/` contains repeatable setup, build, patch, and launch commands.
- `patches/` contains explicit, version-pinned vLLM and RCCL patches.
- `cmake/` and `tools/` contain the Windows ROCm toolchain integration.
- `pins/` records the exact successful PyTorch, ROCm, RCCL, HIPIFY, and vLLM
  versions.

See [docs/architecture.md](docs/architecture.md),
[docs/results.md](docs/results.md), and
[docs/upstream-status.md](docs/upstream-status.md) for design details, full
measurements, limitations, and upstream context.

## License and third-party notices

Original code in this repository is licensed under Apache-2.0; see
[LICENSE](LICENSE). Compatibility patches retain the licenses and notices of
their upstream projects. In particular, RCCL/NCCL-derived patches and any
distributed `rccl.dll` remain covered by the upstream terms reproduced in
[LICENSES/RCCL-UPSTREAM-LICENSE.txt](LICENSES/RCCL-UPSTREAM-LICENSE.txt),
[LICENSES/RCCL-UPSTREAM-NOTICES.txt](LICENSES/RCCL-UPSTREAM-NOTICES.txt), and
[LICENSES/RCCL-UPSTREAM-ThirdPartyNotices.txt](LICENSES/RCCL-UPSTREAM-ThirdPartyNotices.txt).
Repository-specific attribution is in [NOTICE](NOTICE).

Binary release archives must include `LICENSE`, `NOTICE`, and the complete
`LICENSES` directory next to the binaries.
