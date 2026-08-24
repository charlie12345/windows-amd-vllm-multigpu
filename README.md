# Windows AMD vLLM Multi-GPU

Native-Windows AMD tensor parallelism for vLLM using a port of RCCL plus a
D3D12 cross-adapter AllReduce fast path. This is a separate project and does
not modify or build inside the Windows AMD vLLM checkout at `C:\AI\vllm`.

The validated target is Windows 11 with two AMD Radeon AI PRO R9700
(`gfx1201`) GPUs. Other AMD GPUs and software versions require their own
validation.

## What works

- real vLLM model sharding across two GPU worker processes;
- concurrent kernel execution on both GPUs;
- native Windows RCCL 2.30.7 AllReduce, AllGather, ReduceScatter, and
  Broadcast on the current PyTorch HIP stream;
- a faster two-rank D3D12 AllReduce for FP16, FP32, and BF16;
- deterministic TP2 text generation; and
- a mapped-host fallback when RCCL or D3D12 is disabled.

The default hybrid uses the following division of work:

| Operation | Data-plane backend | Purpose |
| --- | --- | --- |
| AllReduce SUM | D3D12 cross-adapter heap/fence + HIP | Hot vLLM reduction path |
| AllGather, ReduceScatter, Broadcast | Native Windows RCCL | Remaining GPU collectives |
| Bootstrap and capability agreement | PyTorch Gloo | CPU control plane only |
| Unsupported cases | Pinned/mapped host transport | Correctness fallback |

AllReduce tensors larger than the configured D3D12 heap automatically route
to RCCL instead of failing; smaller supported tensors continue on D3D12.

This is true tensor parallelism, but it is not direct VRAM-to-VRAM P2P. The
current Windows HIP driver reports no peer access and rejects cross-device HIP
IPC. RCCL consequently selects its NET/Socket transport. D3D12 cross-adapter
heaps work, but Microsoft specifies them as L0/system-memory allocations on
discrete adapters.

## How the hybrid AllReduce works

For every supported two-rank AllReduce, each worker enqueues this sequence on
the current PyTorch HIP stream:

1. copy its local tensor shard from VRAM into its own D3D12 cross-adapter heap;
2. signal a D3D12 timeline fence from the GPU stream;
3. wait on the peer GPU's fence without a CPU payload copy;
4. copy the peer's cross-adapter heap into temporary local VRAM;
5. launch a local HIP kernel that sums the local and peer tensors; and
6. signal consumption so neither rank reuses its heap too early.

The copy engines and reduction kernels do the payload work. Gloo exchanges
object names and RCCL unique IDs, but never receives a HIP tensor. RCCL handles
collectives that are not routed to the D3D12 AllReduce. See
[docs/architecture.md](docs/architecture.md) for the ABI, lifetime, and
fallback details.

## Measured reference result

Qwen3-0.6B produced the same deterministic TP2 token IDs through RCCL and the
hybrid path. With batch size 1, an 8-token input, one generated token, 10
warmups, and 30 measured iterations:

| Configuration | Average latency | Relative to TP1 |
| --- | ---: | ---: |
| TP1, one R9700 | 77.75 ms | baseline |
| TP2, RCCL socket transport | 61.82 ms | 1.26x faster |
| TP2, D3D12 AllReduce + RCCL | 59.00 ms | 1.32x faster |

At a 64 MiB FP16 payload, the D3D12 AllReduce measured 4.63 ms versus
36.73 ms for RCCL NET/Socket on this machine. These are reference-machine
measurements, not general performance guarantees.

## Prerequisites

Install these before cloning:

- Windows 11 x64;
- two supported AMD GPUs and an AMD driver compatible with the pinned ROCm
  wheel train;
- PowerShell 5.1 or newer;
- Git for Windows;
- [uv](https://docs.astral.sh/uv/);
- Visual Studio 2022 Build Tools with Desktop development with C++, MSVC x64,
  and a Windows SDK; and
- Vulkan SDK 1.4.350.0 for the included Vulkan capability probe.

The scripts currently target `gfx1201` and use the standard Visual Studio and
Vulkan locations shown in [scripts/build-native.cmd](scripts/build-native.cmd)
and [scripts/configure-rccl-windows.ps1](scripts/configure-rccl-windows.ps1).
Change those pins deliberately for another GPU or toolchain.

Recommended firmware settings are Above-4G decoding enabled, Re-Size BAR
enabled, and CSM disabled. They do not force HIP P2P support; the capability
probes remain authoritative.

## Clean build from a new clone

Open PowerShell:

```powershell
git clone https://github.com/charlie12345/windows-amd-vllm-multigpu.git
Set-Location .\windows-amd-vllm-multigpu

# 1. Create the isolated ROCm/PyTorch build environment.
.\scripts\bootstrap-nightly.ps1

# 2. Build the mapped-memory and D3D12/HIP native components.
.\scripts\build-native.cmd

# 3. Fetch exact RCCL and HIPIFY commits and apply the version-locked patch.
.\scripts\sync-upstreams.ps1
.\scripts\apply-rccl-patches.ps1

# 4. Configure and build the RCCL functions used by vLLM.
.\scripts\configure-rccl-windows.ps1 -FunctionProfile Vllm
.\scripts\build-rccl-windows.ps1 -Jobs 8

# 5. Validate native RCCL and D3D12.
.\scripts\run-rccl-validation.ps1 -SkipBuild
.\.venv\Scripts\python.exe .\probes\d3d12_cross_process_probe.py
.\.venv\Scripts\python.exe .\probes\d3d12_all_reduce_probe.py

# 6. Create the isolated vLLM environment and build the pinned Windows fork.
.\scripts\bootstrap-vllm.ps1 -MaxJobs 16
```

The RCCL output is `build\rccl-windows\rccl.dll`. The D3D12 transport is
`build\native\wavmg_d3d12_v1.dll`. Both are local build products and are
intentionally excluded from Git and the Python wheel.

`Vllm` builds the exact operations and dtypes used by the adapter. Use
`-FunctionProfile Full` to build RCCL's complete generated function table.
`Minimal` is only for compiler bring-up.

All cloned upstreams, virtual environments, logs, model output, and compiled
artifacts stay in ignored directories under this repository or in the user's
Hugging Face cache.

## Run vLLM with TP2

The launch script defaults to D3D12 AllReduce plus RCCL:

```powershell
.\scripts\run-vllm-tp2.ps1 `
    -Model Qwen/Qwen3-0.6B `
    -TensorParallelSize 2 `
    -UseRccl $true `
    -UseD3D12 $true
```

Use `-UseD3D12 $false` for RCCL-only. Disable both flags for the mapped-host
fallback. The adapter currently accepts TP1 or TP2 and rejects pipeline, data,
decode-context, prefill-context, and expert-parallel layouts.

Set `WAVMG_TRACE_COLLECTIVES=1` to print one routing marker per operation and
backend on each rank. It is enabled by the large-model validation script.

## Large-model proof of tensor parallelism

The pinned large test is
[`mistralai/Mistral-Small-24B-Instruct-2501`](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501)
at revision `9527884be6e5616bdd54de542f9ae13384489724`. It is Apache-2.0,
contains 24B BF16 parameters, and its model card reports roughly 55 GB of GPU
memory. The selected ten sharded weight files total 47,144,848,872 bytes
(43.9 GiB), which exceeds one R9700's 31.86 GiB VRAM.

Download only the required sharded representation; the script deliberately
excludes the duplicate 47 GB `consolidated.safetensors`:

```powershell
.\.venv-vllm\Scripts\python.exe .\scripts\download-large-test-model.py
.\scripts\run-large-model-tp2.ps1
```

Allow about 50 GiB of free disk space. Enabling Windows Developer Mode avoids
Hugging Face's degraded no-symlink cache warning but is not required for one
revision.

The validated TP2 run loaded all 43.91 GiB of checkpoint weights and reported
21.96 GiB of model memory per worker. Each GPU used 22.11 GiB for weights plus
runtime state, peaked at 0.45 GiB of activation memory, and retained 6.76 GiB
of KV cache at 92% configured utilization. Both ranks logged D3D12 AllReduce
and RCCL AllGather. The model generated:

- `The capital of France is` → `Paris. It is known for its iconic landmarks
  such as the Eiffel Tower`
- `Two plus two equals` → `four. This is a mathematical fact. It is not a
  matter of opinion.`

This proves model sharding because the unquantized checkpoint cannot reside on
either individual GPU. A separate exact-value probe also passed 1 MiB through
D3D12 and automatically routed an 80 MiB operation through RCCL when it
exceeded the configured 64 MiB D3D12 heap.

## Speed testing

Run a warmed end-to-end comparison of the hybrid and RCCL-only paths on the
24B model:

```powershell
.\scripts\benchmark-large-model.ps1 `
    -Mode Both `
    -InputLen 32 `
    -OutputLen 16 `
    -BatchSize 1 `
    -WarmupIterations 3 `
    -Iterations 10
```

The script loads the exact same TP2 model/configuration for each case, prints
average and percentile latency, and writes JSON files under
`logs\large-model-benchmark-*`. `Hybrid` uses D3D12 AllReduce plus RCCL;
`Rccl` disables only D3D12. The 24B BF16 model cannot be benchmarked at TP1 on
a 32 GiB card, so this comparison measures transport improvement, not TP1
versus TP2 scaling.

Use `-BatchSize 4` to emphasize aggregate throughput, provided
`InputLen * BatchSize` stays within `MaxNumBatchedTokens`. For low-noise results,
close games and GPU applications, keep the power/clock policy fixed, and run
the command at least three times.

Reference result from the command above (one paired run on the reference
machine):

| Transport | Average | p50 | p90 | Approx. output tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Hybrid D3D12 + RCCL | 1.530 s | 1.538 s | 1.544 s | 10.46 |
| RCCL NET/Socket only | 2.594 s | 2.584 s | 2.666 s | 6.17 |

The hybrid was 1.69x faster with 41.0% lower average latency. Approximate
output tokens/s is `BatchSize * OutputLen / average latency`; it includes both
prefill and decode time and is not a server-concurrency benchmark.

To isolate collective performance from model compute:

```powershell
# D3D12 exact-value latency across sizes through 64 MiB.
.\.venv\Scripts\python.exe .\probes\d3d12_all_reduce_probe.py

# RCCL FP16 AllReduce at 64 MiB for 100 measured iterations.
.\scripts\benchmark-rccl.ps1 -Dtype f16 -Count 33554432 -Iterations 100
```

## Reproducibility and safety limits

Exact PyTorch, ROCm, RCCL, HIPIFY, vLLM, and large-model revisions are recorded
in [pins/nightly-2026-07-28.json](pins/nightly-2026-07-28.json). The RCCL and
vLLM patch scripts verify both the upstream commit and patch applicability
before modifying ignored sandbox clones.

This remains experimental. Current limits include exactly two ranks for the
D3D12 fast path, no driver-level VRAM P2P, no peer-death GPU watchdog, no
custom PyTorch ProcessGroup, and limited long-duration/model coverage. Do not
interpret a successful ReBAR setup as GPU-direct support.

## Repository layout

- `native/`: mapped-memory kernels, D3D12/HIP DLL, and capability probes.
- `src/`: RCCL wrapper, D3D12 transport, fallbacks, and vLLM plugin.
- `probes/`: exact-value, stream-ordering, model, and performance tests.
- `scripts/`: pinned bootstrap, native build, validation, and launch commands.
- `patches/`: version-locked vLLM and RCCL source patches.
- `cmake/` and `tools/`: Windows ROCm build integration.
- `pins/`: the exact validated dependency and model revisions.
- `LICENSES/`: verbatim upstream licenses/notices plus derived-code notices.

Full results and upstream constraints are in
[docs/results.md](docs/results.md) and
[docs/upstream-status.md](docs/upstream-status.md).

## Licensing and attribution

Original code in this repository is Apache-2.0; see [LICENSE](LICENSE).
Version-pinned patches remain under their upstream licenses. Exact vLLM and
RCCL license/notice texts, third-party attributions, provenance, and binary
redistribution requirements are documented in
[docs/licensing.md](docs/licensing.md) and stored under [LICENSES](LICENSES).

Model weights are downloaded from Hugging Face and are not committed,
repackaged, or relicensed by this project. Anyone distributing binaries must
place `LICENSE`, `NOTICE`, and the complete `LICENSES` directory next to
those binaries.
