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

## Experimental HIP/PAL peer-VRAM probe

This repository now contains a separate, version-locked build of AMD's Windows
HIP/CLR runtime. It does **not** replace the ROCm wheel's `amdhip64_7.dll`, and
it does not alter `C:\AI\vllm`. The patch adds an opt-in capability-gate probe
and, critically, disables PAL's staged-copy fallback while that probe is active.
That makes a failed peer-VRAM mapping observable instead of allowing a correct
host-staged copy to masquerade as direct P2P.

The probe is diagnostic, not a production runtime. On the validated pair of
R9700s and AMD driver `32.0.31035.1003`:

| Test | Result |
| --- | --- |
| Unmodified PAL capability, GPU 0 to 1 and 1 to 0 | `false`; enable returns HIP 101 |
| Forced PAL capability gate | capability and enable return success |
| Fail-closed PAL peer-resource creation | fails; PAL logs `Video memory allocation failed` |
| Same-GPU cross-process IPC, GPU 0 to GPU 0 | passes with correct bytes |
| Cross-GPU IPC, GPU 0 to GPU 1 | fails at import with HIP 17 |
| Direct VRAM P2P DMA proven | **no** |

The same-GPU IPC control proves that handle duplication and the IPC test itself
work. The cross-GPU import reaches Windows WDDM but the importing GPU's
`D3DKMTQueryResourceInfoFromNtHandle` is rejected with
`STATUS_INVALID_PARAMETER`. The two GPUs also enumerate under separate PCIe
root-complex branches, and HIP reports no Large BAR. These results put the
remaining boundary in AMD PAL/KMD (`amdkmdag.sys`), where peer allocation,
page-table mapping, residency, and coherency are owned. A user-mode HIP or RCCL
patch cannot safely grant those kernel-driver capabilities.

Reproduce the isolated build and tests:

```powershell
# First run prepares the exact sparse checkout and stops at the AMD interop EULA.
.\scripts\sync-hip-p2p-runtime.ps1
Get-Content .\sandbox\rocm-systems-hip-p2p\shared\amdgpu-windows-interop\LICENSE

# Continue only after personally accepting those upstream binary terms.
.\scripts\sync-hip-p2p-runtime.ps1 -AcceptAmdInteropEula
.\scripts\apply-hip-p2p-patches.ps1
.\scripts\build-hip-p2p-runtime.ps1 -Jobs 4

$Runtime = '.\build\hip-p2p-runtime\install\bin\amdhip64_7.dll'
$RocmBin = (& .\.venv-vllm\Scripts\python.exe -m rocm_sdk path --bin).Trim()
.\scripts\probe-hip-peer-access.ps1 `
    -HipRuntimeDll $Runtime `
    -DependencyDirectory $RocmBin `
    -ForcePalPeerProbe

$env:PYTHONPATH = '.\src'
$env:GPU_FORCE_P2P_COMPAT = '1'
.\.venv-vllm\Scripts\python.exe .\probes\hip_ipc_probe.py `
    --runtime-dll $Runtime --dependency-dir $RocmBin `
    --export-device 0 --import-device 0
.\.venv-vllm\Scripts\python.exe .\probes\hip_ipc_probe.py `
    --runtime-dll $Runtime --dependency-dir $RocmBin `
    --export-device 0 --import-device 1
```

The expected forced peer-access probe currently exits nonzero because the
fail-closed copy is deliberately incorrect after PAL rejects both peer
mappings. Never copy the experimental DLL over the wheel or system runtime.
The build output is ignored by Git and is not redistributed because AMD's
prebuilt Windows PAL/WKMI objects have separate binary terms. See `NOTICE` and
`LICENSES/ROCM-SYSTEMS-CLR-MIT.txt`.

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

## Portable ComfyUI integration

The transport package also exposes `D3D12TensorBridge`, a single-process,
one-way tensor transport for ComfyUI's built-in multi-GPU CFG work scheduler.
It uses the same D3D12 cross-adapter heap and GPU-scheduled HIP copies, but it
does not perform an AllReduce: a secondary GPU's completed model output is
returned to the primary GPU for sampling aggregation.

The separately versioned `ComfyUI-AMD-MultiGPU-Bridge` custom node installs a
small output-transport hook in `comfy/samplers.py`. On the reference dual-R9700
machine, clean GPU1-to-GPU0 runs measured 0.254 ms versus 0.477 ms for a 1 MiB
payload and 2.686 ms versus 6.099 ms for 64 MiB. The custom node defaults to a
64 MiB D3D12 heap and falls back to PyTorch outside its configured range.

This ComfyUI path is genuine concurrent CFG work-unit execution on both GPUs,
not tensor parallel sharding of a single diffusion layer. It needs two
independent conditioning work units and does not accelerate CFG=1 workflows
that produce only one unit.

The Windows RCCL DLL loads under portable ComfyUI's ROCm 7.2 runtime, but the
current same-process/two-rank collective test hangs at the first AllReduce.
RCCL therefore remains disabled in the ComfyUI custom node until that test is
fixed. The working one-process-per-GPU RCCL path remains available to vLLM.

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
[`stelterlab/Mistral-Small-24B-Instruct-2501-AWQ`](https://huggingface.co/stelterlab/Mistral-Small-24B-Instruct-2501-AWQ)
at revision `cbda099649a0188dd888d44f0e4964d8d982dc9a`. It is an Apache-2.0
AWQ quantization of Mistral Small 24B with 4-bit weights, 16-bit activations,
128-value groups, and asymmetric zero points. Its three selected weight shards
total 14,234,370,648 bytes (13.26 GiB).

Download and validate the exact pinned revision:

```powershell
.\.venv-vllm\Scripts\python.exe .\scripts\download-large-test-model.py
.\scripts\run-large-model-tp2.ps1
```

Allow about 15 GiB of free disk space. Model files remain in the Hugging Face
cache and are never added to this repository. The downloader verifies the
revision, exact shard count and byte total, and AWQ quantization metadata.

The validated TP2 run selected vLLM's native RDNA hybrid W4A16 kernel, loaded
6.76 GiB of model memory per worker, and retained about 21.5 GiB of KV cache
per GPU at 92% configured utilization. Both ranks logged real GPU collective
traffic. TP1 and TP2 produced identical deterministic token IDs and text:

- `The capital of France is` → `Paris. It is known for its iconic landmarks
  such as the Eiffel Tower`
- `Two plus two equals` → `four. This is a mathematical fact. It is not a
  matter of opinion.`

The earlier 43.91 GiB BF16 test remains recorded in `docs/results.md` as proof
that a checkpoint too large for either 31.86 GiB GPU was successfully sharded.
The AWQ test is now the default because it is much faster and practical for
repeatable benchmarking.

## Speed testing

The benchmark supports TP1, TP2 hybrid, TP2 RCCL-only, and four optimization
profiles. The baseline command below compares all device layouts:

```powershell
.\scripts\benchmark-awq-decode.ps1 `
    -Mode All `
    -Profile Baseline `
    -InputLen 32 `
    -OutputLen 32 `
    -BatchSize 1 `
    -WarmupIterations 3 `
    -Iterations 10
```

Use the tested TP2 decode profile with:

```powershell
.\scripts\benchmark-awq-decode.ps1 `
    -Mode Rccl `
    -Profile TunedO1 `
    -InputLen 32 `
    -OutputLen 32 `
    -BatchSize 8 `
    -WarmupIterations 3 `
    -Iterations 10
```

Start an OpenAI-compatible local server with the same winning TP2 settings:

```powershell
.\scripts\serve-awq-tp2.ps1 -Profile TunedO1
```

The server binds to `127.0.0.1:8000` by default. The default
`MaxNumBatchedTokens=2048` favors interactive inter-token latency; pass
`-MaxNumBatchedTokens 16384` for a throughput-oriented A/B test.

`TunedO1` enables vLLM async scheduling and Inductor `-O1`. Windows TP2 keeps
CUDA/HIP graph capture disabled because the native collective transports are
not graph-safe. The plugin exposes collectives as opaque vLLM custom ops so
Inductor can compile surrounding model code without tracing Python `ctypes`
stream handles. The first process start compiles and caches the graph; compare
steady-state iterations, not startup time.

Reference results on two Radeon AI PRO R9700 GPUs, 32 input and 32 output
tokens, after three warmups:

| Workload | Configuration | Average | p50 | p90 | Output tokens/s |
| --- | --- | ---: | ---: | ---: | ---: |
| Batch 1 | TP1 eager/O0 | 0.9282 s | 0.9283 s | 0.9331 s | 34.47 |
| Batch 1 | TP2 RCCL eager/O0 | 0.8912 s | 0.8930 s | 0.8959 s | 35.91 |
| Batch 1 | TP2 RCCL async + O1 | 0.8424 s | 0.8437 s | 0.8449 s | 37.99 |
| Batch 8 | TP1 eager/O0 | 3.2915 s | 3.1502 s | 3.8286 s | 77.78 |
| Batch 8 | TP2 RCCL eager/O0 | 2.4699 s | 2.4550 s | 2.5282 s | 103.65 |
| Batch 8 | TP2 RCCL async + O1 | 2.3912 s | 2.3069 s | 2.4194 s | 107.06 |

At batch 8, tuned TP2 delivered 37.6% more aggregate output throughput than
the TP1 eager baseline. At batch 1, communication overhead limits scaling, but
tuned TP2 was still 10.2% faster. Use TP1 when the model fits and minimum
single-request latency is the only objective; use TP2 for concurrency, larger
KV cache, models that do not fit one GPU, or the measured batch-throughput
gain.

### Qwen3.8-27B BF16 format trial

The Apache-2.0 [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
checkpoint was tested at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Its 18 weight shards total
55,563,006,776 bytes (51.75 GiB), so the BF16 model cannot fit on either
31.86 GiB R9700 by itself. TP2 loaded 25.24 GiB of model memory per worker,
which directly verifies that vLLM sharded the model across the two GPUs.

These text-only runs used `--language-model-only`, 32 input tokens, 32 output
tokens, automatic `ROCM_ATTN`, no prefix cache, and a 512-token model limit.
The model's GDN linear-attention prefill and recurrent-decode kernels used the
Triton/FLA path.

| Batch | Configuration | Average | p50 | p90 | Output tokens/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | TP2 D3D12 + RCCL eager/O0 | 11.0946 s | 11.0987 s | 11.1249 s | 2.88 |
| 1 | TP2 RCCL eager/O0 | 10.5343 s | 10.4008 s | 10.8281 s | 3.04 |
| 1 | TP2 RCCL async + O1 | 10.2798 s | 10.2957 s | 10.3629 s | **3.11** |
| 1 | TP2 RCCL async eager + one-token MTP | 10.8185 s | 10.9276 s | 11.4470 s | 2.96 |
| 8 | TP2 RCCL async eager | 28.0533 s | 28.0294 s | 28.7700 s | **9.13 aggregate** |

RCCL-only was 5.5% faster than the hybrid at batch 1, and async scheduling
plus O1 improved the hybrid baseline by 8.0%. Qwen's built-in one-token MTP
speculative decoder was 4.8% slower than the winning ordinary decode profile
on this random-token latency workload. The first O1 launch required 429
seconds to compile and cache the 64-layer graph; that startup cost is excluded
from the steady-state table. An O1+MTP compile was not benchmarked because its
separate AOT artifact exhausted the remaining disk during the trial.

Reproduce a format trial with:

```powershell
.\scripts\benchmark-qwen38-27b.ps1 `
    -ModelPath C:\path\to\Qwen3.8-27B `
    -ModelLabel bf16 `
    -Transport Rccl `
    -Profile TunedO1 `
    -BatchSize 1
```

The BF16 checkpoint was deleted after recording the results. It is not
distributed by this repository and can be restored only by downloading the
pinned Hugging Face revision again.

### Decode-related vLLM flags

These settings are verified for this AWQ checkpoint and machine:

- `--quantization awq` plus `VLLM_ROCM_USE_RDNA_W4A16=1` selects the native
  RDNA W4A16 decode kernel. The environment variable currently defaults on,
  but setting it explicitly makes runs reproducible.
- `--async-scheduling -O1 --compilation-config '{"compile_sizes":[]}'` is the
  fastest tested TP2 profile. Async scheduling alone reached 36.43 tokens/s;
  the full profile reached 37.99 tokens/s.
- Keep `--dtype bfloat16`. An otherwise identical FP16 run reached only 36.15
  tokens/s.
- Leave `--attention-backend` on `auto`; it selects `ROCM_ATTN`. Forcing
  `TRITON_ATTN` reached only 35.67 tokens/s.
- Use RCCL-only (`WAVMG_USE_D3D12=0`) for this quantized model. Its small
  reductions do not amortize the D3D12 cross-adapter route. This is
  workload-specific: D3D12 was substantially faster for the earlier BF16
  checkpoint and remains useful for larger collective payloads.

Additional flags can help particular serving workloads but are not universal
decode speedups:

- `--max-num-batched-tokens 2048` favors inter-token latency under mixed
  prefill/decode load. Values above 8192 favor aggregate throughput; benchmark
  them with the intended context lengths and concurrency.
- `--max-num-seqs` raises or caps request concurrency. It increases throughput
  only when enough requests are waiting.
- `--enable-prefix-caching` avoids repeated prefill work for shared prefixes;
  it does not accelerate unique-token decode.
- `--speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":4}'`
  can help prompts whose generated text repeats prompt n-grams. Acceptance rate
  determines whether it wins.
- `--kv-cache-dtype fp8` doubles KV-cache capacity on ROCm and may help
  long-context attention bandwidth. It can affect numerical quality and has
  not improved this short-context test, so keep `auto` unless a long-context
  A/B test wins.

`--gpu-memory-utilization`, `--kv-cache-memory-bytes`, and
`--safetensors-load-strategy=prefetch` tune capacity or startup rather than
steady decode. `--enforce-eager` and `-O0` are useful compatibility baselines,
but they disable the measured Inductor improvement.

For low-noise results, close games and GPU applications, keep the power/clock
policy fixed, and run each command at least three times. JSON results are
written beneath `logs/awq-decode-benchmark-*` and intentionally ignored by Git.

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
