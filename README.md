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

## Ling 3.0 tiny support

The pinned Windows vLLM build now includes upstream support for
[`inclusionAI/Ling-3.0-tiny`](https://huggingface.co/inclusionAI/Ling-3.0-tiny)
(`BailingMoeV3ForCausalLM`), including its hybrid KDA/MLA attention, MoE model,
MTP model class, Ling reasoning/tool parsers, FP8 handling, and routed-expert
MXFP4 support. The source commits and all three checkpoint revisions are pinned
in `pins/nightly-2026-07-28.json`.

The official native vLLM-format checkpoints are:

| Checkpoint | Declared format | Weight bytes | Status in this build |
| --- | --- | ---: | --- |
| `inclusionAI/Ling-3.0-tiny` | BF16 SafeTensors | 15,787,992,416 | Real-weight TP1 and TP2 O1 benchmarks passed |
| `inclusionAI/Ling-3.0-tiny-fp8` | FP8 SafeTensors, dynamic activations, 128x128 blocks | 8,409,692,800 | Upstream load/kernel support ported; real weights not yet tested here |
| `inclusionAI/Ling-3.0-tiny-int4` | compressed-tensors INT4, group size 32 | 5,805,705,224 | Standard non-NVFP4 vLLM format; real weights not yet tested here |

These are not GGUF checkpoints. The repositories declare MIT; weights are not
bundled with this project. Because the pinned Hugging Face repository maps its
custom configuration code, launches currently require `--trust-remote-code`.
Keep the pinned revision and inspect that code before allowing it on another
machine.

Ling exposed a gap on RDNA 4 Windows: upstream could select only AITER or ROCm
FlashAttention for MLA prefill, neither of which supports this installed
`gfx1201` stack. This project adds `TRITON_MLA` as a third, lower-priority ROCm
prefill backend. It supports Ling's 192-wide Q/K and 128-wide V heads,
variable-length batches, causal prefill, and LSE output for chunked-context
merging. Attention payloads remain GPU-resident. AITER and FlashAttention stay
higher priority whenever their runtime feature checks pass.

Run the reproducible BF16 architecture smoke without downloading the 15.8 GB
checkpoint:

```powershell
# One GPU
.\scripts\smoke-ling3.ps1 -TensorParallelSize 1

# Two GPUs: D3D12 AllReduce plus RCCL for the other collectives
.\scripts\smoke-ling3.ps1 -TensorParallelSize 2
```

Set `-UseDummyWeights $false` to download and run the exact pinned BF16
checkpoint. The smoke deliberately uses eager/O0 for a short correctness test;
it is not the recommended performance profile or a speed benchmark.

On the two-R9700 reference system, TP1 dummy generation loaded 16.54 GiB and
completed four tokens. TP2 loaded approximately 8.83 GiB per worker and
completed two tokens. Trace markers proved D3D12 AllReduce and RCCL AllGather
on both ranks; Gloo remained the CPU rendezvous/control plane. The numerical
MLA tests also match a float32 PyTorch reference for causal and chunked-context
attention. Random dummy weights cannot validate model quality, and the long
first-request times were one-time Triton JIT/autotuning rather than steady-state
throughput.

The real pinned BF16 checkpoint was then benchmarked with a 32-token input,
128-token output, batch size one, three warmups, five measured iterations,
async scheduling, and O1 compilation. The reported rate is generated tokens
divided by complete request latency, so it includes the small prefill cost:

| Configuration | Average latency | Output tokens/s |
| --- | ---: | ---: |
| TP1 | 4.8467 s | **26.41** |
| TP2 RCCL-only | 4.8955 s | **26.15** |
| TP2 D3D12 AllReduce + RCCL | 5.8652 s | **21.82** |

For this relatively small MoE model, one GPU and TP2 RCCL-only are effectively
tied, while D3D12 AllReduce overhead makes the hybrid slower. TP2 still reduces
model memory to about 8.84 GiB per GPU and provides much more KV-cache capacity.
Run `scripts\benchmark-ling3.ps1` to reproduce all three cases. O1 required a
local fix so dynamic compile ranges remain selectable when concrete
`compile_sizes` is unset; that fix is part of the version-locked patch stack.

### Ling concurrent-serving benchmark

vLLM continuous batching does not load another copy of the model for every
request. One loaded engine keeps several sequences active and batches their
decode steps. This can greatly increase aggregate output throughput, although
each individual stream can receive tokens more slowly as concurrency rises.

The same pinned BF16 checkpoint was served through the OpenAI-compatible HTTP
endpoint with O1, async scheduling, a 32-token random input, 128 forced output
tokens, and prefix caching disabled. Each concurrency received a complete
warmup wave before measurement. Concurrency 1-4 used 16 measured requests;
concurrency 8-16 used 32. The August 26, 2026 results were:

| Concurrent requests | TP1 output tok/s | TP1 mean TTFT | TP1 mean TPOT | TP2 RCCL output tok/s | TP2 mean TTFT | TP2 mean TPOT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 33.56 | 174.69 ms | 28.66 ms | 25.29 | 160.87 ms | 38.59 ms |
| 2 | 48.91 | 340.34 ms | 38.53 ms | 51.44 | 257.86 ms | 37.12 ms |
| 4 | 106.02 | 296.90 ms | 35.69 ms | 107.42 | 312.04 ms | 35.02 ms |
| 8 | 83.77 | 330.91 ms | 93.62 ms | 89.56 | 433.86 ms | 86.55 ms |
| 16 | **160.59** | 349.10 ms | 97.63 ms | **164.54** | 460.61 ms | 94.24 ms |

On TP1, concurrency 16 produced 4.79 times the aggregate output rate of
concurrency 1. Concurrency 4 was the better latency/throughput compromise,
while 16 maximized total output. The non-monotonic concurrency-8 result is a
real measurement for these RDNA 4 Triton/MoE batch shapes and should be repeated
before treating it as a general rule. TP2 RCCL was only 2.5% faster than TP1 at
concurrency 16, so sharding this model is primarily useful for memory capacity;
RCCL overhead prevents two-times scaling. Gloo performed rendezvous/control
only; the traced GPU tensor collectives used RCCL on both ranks.

Start the matching server in the first PowerShell window, wait for its health
endpoint to become ready, and run the client-side load matrix in a second:

```powershell
# Window 1: replace Single with Rccl for TP2 RCCL-only
.\scripts\serve-ling3.ps1 -Mode Single

# Window 2
.\scripts\benchmark-ling3-concurrency.ps1 `
    -BaseUrl http://127.0.0.1:8001
```

For maximum independent-request throughput when the whole model fits one GPU,
two separate TP1 replicas behind a load balancer are expected to scale better
than TP2. That replica mode is distinct from tensor parallelism and is not
claimed by the table above.

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

### Qwen3.8-27B native FP8 trial

The official Apache-2.0
[`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)
checkpoint was tested at revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`. Its 66 native SafeTensors
weight shards total 30,866,866,928 bytes (28.75 GiB) and declare vLLM's `fp8`
quantization with dynamic activations and 128-by-128 weight blocks. This was
not a GGUF conversion.

TP1 loaded all 28.50 GiB of model data but could not allocate a usable KV
cache on one 31.86 GiB R9700. TP2 loaded about 14.45 GiB per worker and ran
successfully. Collective tracing verified that hybrid mode routed AllReduce
through the D3D12 cross-adapter bridge and AllGather through Windows RCCL.

The sustained decode comparison used 32 input tokens, 128 generated tokens,
batch 1, eager/O0, BF16 activations, automatic `ROCM_ATTN`, a 512-token model
limit, no prefix cache, and 60% GPU-memory utilization. The lower utilization
still provides ample KV capacity for this workload and avoids Windows WDDM
overcommit after repeated engine restarts.

| Configuration | Average | p50 | p90 | Output tokens/s |
| --- | ---: | ---: | ---: | ---: |
| TP2 D3D12 AllReduce + RCCL AllGather | 31.5488 s | 31.5229 s | 31.6158 s | 4.06 |
| TP2 RCCL-only | **30.7916 s** | **30.7916 s** | **30.8047 s** | **4.16** |

RCCL-only was 2.46% faster than hybrid for sustained batch-1 FP8 decode, so it
is the recommended transport for this checkpoint on the tested two-R9700
machine. Short 32-output-token runs varied by about 20% across repeated engine
launches and are retained in `docs/results.md` rather than advertised as the
primary number. Async eager, O1, forced `TRITON_ATTN`, and locally tuned FP8
GEMM configurations all lost to eager/O0 with automatic attention. MTP exposed
a two-rank Windows Triton-cache race and is not recommended for this format.

Reproduce the sustained FP8 run with:

```powershell
.\scripts\benchmark-qwen38-27b.ps1 `
    -ModelPath G:\AI-models\Qwen3.8-27B-FP8 `
    -ModelLabel fp8-longdecode `
    -Transport Rccl `
    -Profile Baseline `
    -InputLen 32 `
    -OutputLen 128 `
    -BatchSize 1 `
    -WarmupIterations 2 `
    -Iterations 4 `
    -MaxModelLen 512 `
    -MaxNumBatchedTokens 512 `
    -GpuMemoryUtilization 0.60
```

The FP8 checkpoint was deleted after this sequential format trial. Model
weights and local benchmark logs are not distributed by this repository.

### Qwen3.8-27B standard 4-bit W4A16 trial

NVFP4 is not the AMD performance target for this project. The native
SafeTensors `unsloth/Qwen3.8-27B-NVFP4` checkpoint loaded correctly, but vLLM
selected `EmulationNvFp4LinearKernel` on the R9700. It reached only 1.05
tokens/s on TP1, 1.71 tokens/s on TP2 RCCL, and 1.75 tokens/s on TP2 hybrid
for a 32-input/32-output batch-1 test. Those results prove tensor-parallel
execution but do not represent optimized AMD 4-bit inference. Full numbers
and test qualifications are in `docs/results.md`.

The replacement is the Apache-2.0
[`abihsoro/Qwen3.8-27B-AWQ-INT4`](https://huggingface.co/abihsoro/Qwen3.8-27B-AWQ-INT4)
checkpoint pinned at `f2e0cac39907e7b1ed7fdb210363dd33cc18f993`.
It is one 17,646,863,912-byte SafeTensors weight file using the standard
compressed-tensors `pack-quantized` W4A16 schema: INT4 group-128 symmetric
weights with BF16 activations. It is not GGUF, NVFP4, or a runtime conversion.

Download and validate the exact checkpoint on `G:` with:

```powershell
.\.venv-vllm\Scripts\python.exe .\scripts\download-qwen38-27b-awq.py
```

The downloader validates the Hugging Face commit, declared Apache-2.0 license,
weight byte count, architecture, packing format, bit width, group size, and
activation scheme. During preflight, do not accept the checkpoint as optimized
unless the runtime reports
`Using RDNAHybridW4A16LinearKernel for CompressedTensorsWNA16`. That kernel
keeps INT4 weights packed and dispatches decode shapes to the HIP skinny GEMM
and larger prefill shapes to the tuned Triton W4A16 GEMM.

The primary 32-input/128-output batch-1 results were:

| Configuration | Average | Output tokens/s |
| --- | ---: | ---: |
| TP1 eager/O0 | 4.7218 s | 27.11 |
| TP1 async eager | 4.0039 s | **31.97** |
| TP1 async + O1 | 5.0841 s | 25.18 |
| TP2 RCCL eager/O0 | 6.0356 s | 21.21 |
| TP2 RCCL async eager | 5.5275 s | 23.16 |
| TP2 RCCL async + O1 | 4.3613 s | **29.35** |
| TP2 hybrid eager/O0 | 6.8936 s | 18.57 |

For one request, TP1 async eager remains 8.2% faster than the best TP2 result.
At batch 8, however, TP2 RCCL reached 81.84 aggregate output tokens/s versus
65.78 on TP1, a 24.4% throughput gain. Tensor parallelism is therefore useful
for capacity and concurrency even when communication overhead loses on a
single short request.

### DFlash2 result

The official Apache-2.0
[`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
drafter is pinned at `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` and can be
downloaded and validated with:

```powershell
.\.venv-vllm\Scripts\python.exe .\scripts\download-qwen38-27b-dflash2.py
```

The pinned vLLM patch adds upstream DFlash2 support and its initialization fix.
On ROCm, the target and drafter must use matching attention backends; an
automatic `ROCM_ATTN` target plus `TRITON_ATTN` drafter creates incompatible
shared KV-cache layouts. The benchmark launcher now matches both to
`TRITON_ATTN` by default and exposes `-DFlashAttentionBackend` for controlled
experiments.

The feature is correct and distributed, but it is not a speedup for this
W4A16/RDNA4 path:

| Configuration | Runs | Average | Output tokens/s |
| --- | ---: | ---: | ---: |
| TP1 V2+Triton, no speculation | 10 | 4.9296 s | **25.97** |
| TP1 DFlash2, depth 1 | 5 | 6.2148 s | 20.60 |
| TP1 DFlash2, depth 7 | 10 | 30.0670 s | 4.26 |
| TP2 RCCL V2+Triton, no speculation | 5 | 8.7366 s | **14.65** |
| TP2 RCCL DFlash2, depth 7 | 5 | 22.3202 s | 5.73 |
| TP2 hybrid DFlash2, depth 7 | 3 | 24.8854 s | 5.14 |

TP2 RCCL improved the seven-token DFlash path by 34.5% over TP1 and both ranks
emitted RCCL AllReduce/AllGather traces, but it was still 60.9% slower than its
matched non-speculative TP2 control. The hybrid used D3D12 AllReduce plus RCCL
AllGather and was 10.3% slower than RCCL-only. DFlash2 remains experimental and
off by default until AMD verification GEMMs and acceptance behavior are faster.

### Decode-related vLLM flags

These settings are verified for this AWQ checkpoint and machine:

- `--quantization awq` plus `VLLM_ROCM_USE_RDNA_W4A16=1` selects the native
  RDNA W4A16 decode kernel. The environment variable currently defaults on,
  but setting it explicitly makes runs reproducible.
- `--async-scheduling -O1 --compilation-config '{"compile_sizes":[]}'` is the
  fastest tested TP2 single-request profile at 29.35 tokens/s. On TP1, async
  eager was faster at 31.97 tokens/s, so do not assume one compile profile wins
  on both topologies.
- Keep `--dtype bfloat16`. An otherwise identical FP16 run reached only 36.15
  tokens/s.
- Leave `--attention-backend` on `auto` for normal decode so it can select
  `ROCM_ATTN`. Use matching target/draft backends for DFlash2; the launcher
  defaults that experiment to `TRITON_ATTN` for ROCm correctness.
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

The Windows vLLM patch set also adds cooperative EngineCore shutdown. Python's
`multiprocessing.Process.terminate()` maps to `TerminateProcess` on Windows and
does not run Python's SIGTERM handlers; without the patch, repeatedly launching
`vllm bench latency` can orphan TP workers and leave their HIP contexts charged
to WDDM even after the PIDs disappear. The patch sends a spawn-safe shutdown
event first, lets EngineCore release workers and GPU resources normally, and
retains forced termination only as a timeout fallback. Validate the primitive
with:

```powershell
.\.venv-vllm\Scripts\python.exe .\probes\probe_windows_engine_shutdown.py
```

The Qwen benchmark launcher also handles a Windows PowerShell 5.1 edge case:
native stderr becomes a PowerShell error record when merged into the log
pipeline. PyTorch's harmless c10d IPv4-mapped IPv6 warning previously met the
script-wide `Stop` policy, terminated the client, and left a healthy EngineCore
orphaned. The launcher now preserves native stderr as text, waits for vLLM, and
uses the native exit code as the authoritative result.

If a run made before this patch already left nonexistent PIDs in the Windows
`GPU Process Memory` counters, a normal reboot or elevated display-adapter
restart is required once to clear those driver allocations.

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
