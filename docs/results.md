# Reference results

Reference machine: Windows 11, two AMD Radeon AI PRO R9700 (`gfx1201`) GPUs.

## Pinned stack

- Python 3.12
- PyTorch `2.11.0+rocm7.15.0a20260728`
- HIP `7.15.26290`
- AMD ROCm wheel train `7.15.0a20260728`
- vLLM `0.1.dev19487+gfb9fb8c5a.rocm715`, source commit
  `fb9fb8c5aeaed96c91eef5cb48743a96f8496907`
- RCCL `2.30.7-develop`, rocm-systems commit
  `ee3bae9a931561506c49dcf82fca52ec4711c34f`
- HIPIFY commit `f1af27c6e0c43e1a9663dc3650dcff54f980e6a6`

`torch.distributed` and Gloo are available; the PyTorch wheel has no built-in
NCCL/RCCL backend. This repository's `ctypes` adapter loads the separately
built native `rccl.dll`. Two GPUs are visible and peer access is false in both
directions.

## Windows HIP/PAL peer-VRAM experiment

An isolated `amdhip64_7.dll` was built from the exact ROCm source commit
`44be71b52284948e58c93f65f46910399773fdcd`, matching the installed TheRock
wheel manifest. The build used AMD's documented public Windows CLR/PAL path;
its output stayed under `build/hip-p2p-runtime` and never replaced the working
wheel runtime.

The default runtime and the isolated runtime with the experimental flag off
both reported `hipDeviceCanAccessPeer=false` in both directions. Peer enable
returned HIP 101. `hipMemcpyPeer` still returned correct data because PAL has
an explicit two-step, 4 MiB host-staging fallback; copy correctness alone was
therefore not accepted as P2P evidence.

The opt-in `GPU_FORCE_P2P_COMPAT=1` patch bypassed only the cached PAL
compatibility result. HIP then reported capability and peer enable as success,
but PAL's attempt to open peer GPU memory logged `Video memory allocation
failed`. A second patch made this mode fail closed instead of entering the
staging fallback. Both directions then produced incorrect destination bytes,
as expected when no peer resource exists. This disproves the apparent success
from capability and enable alone.

Cross-process IPC supplied the decisive control:

| Exporter | Importer | Export | Import | Correct data |
| --- | --- | --- | --- | --- |
| GPU 0 | GPU 0 | pass | pass | yes |
| GPU 0 | GPU 1 | pass | HIP 17 | no |

The same-device pass proves that the NT-handle exchange and probe are valid.
The cross-device path fails when WDDM queries the GPU-0 resource using GPU 1's
device handle; the translated driver result is error 9/
`STATUS_INVALID_PARAMETER`. Peer GPU-VA mapping is never reached.

Topology and BAR evidence on the same machine:

- GPU 0 is under `PCIROOT(20)` and GPU 1 under `PCIROOT(C0)`;
- both endpoints negotiate PCIe Gen5 x16;
- HIP reports `isLargeBar: 0` and an empty peer list for both GPUs; and
- Windows exposes a 256 MiB high MMIO aperture, not a full 31.86 GiB VRAM BAR,
  for each adapter.

Conclusion: true peer-VRAM DMA is not available on this driver/GPU/topology.
The open HIP front end can be patched and rebuilt, but the present AMD Windows
PAL/KMD rejects the required cross-adapter local-memory object. The working
production path remains real TP2 with RCCL plus the D3D12 L0/system-memory
transport; it must not be described as direct VRAM P2P.

## Correctness

Two-rank probes pass CPU Gloo all-reduce; native FP32, FP16, and BF16
shared-memory all-reduce; host-staged all-reduce, all-gather, gather,
reduce-scatter, broadcast, and point-to-point. Direct GPU Gloo is intentionally
excluded because it causes a native Windows access violation.

Native Windows RCCL passed exact byte/value checks for:

- AllReduce sum with FP16, FP32, and BF16;
- AllGather with FP16 and BF16;
- ReduceScatter sum with FP32 and BF16;
- Broadcast with FP32;
- 100 consecutive FP16 AllReduce operations at 1,048,576 elements; and
- all four adapter operations on a non-default PyTorch HIP stream.

The D3D12 cross-process probe passed exact 1 MiB transfers in both directions,
including imported cross-adapter fences. The D3D12 two-rank AllReduce then
passed exact FP16, FP32, and BF16 sums at 4 KiB, 64 KiB, 1 MiB, 8 MiB, and
64 MiB on a non-default PyTorch HIP stream.

The RCCL vLLM TP2 run returned the same token IDs shown below and shut down
cleanly. This establishes correctness and integration, not GPU-direct: RCCL's
topology selected NET/Socket because HIP P2P and SHM device mappings are not
available on the pinned Windows driver.

The vLLM reference test used `Qwen/Qwen3-0.6B`, two prompts, deterministic
sampling, and eight output tokens. TP1 and TP2 returned identical token IDs:

| Prompt | Token IDs from both TP1 and TP2 |
| --- | --- |
| `The capital of France is` | `12095, 13, 576, 6722, 315, 9625, 374, 1083` |
| `Two plus two equals` | `1378, 11, 323, 1378, 5519, 1378, 16819, 1378` |

Each TP2 worker loaded about 0.57 GiB of model weights versus about 1.12 GiB
for TP1, confirming that the model was sharded rather than replicated.

## Mapped-host all-reduce transport latency

Representative latency from the stream-ordered native path follows. Windows
small-message measurements are noisy; these figures establish scale, not a
formal performance guarantee.

| Payload | FP32 | FP16 | BF16 |
| ---: | ---: | ---: | ---: |
| 4 KiB | 0.260 ms | 0.167 ms | 0.172 ms |
| 8 KiB | 0.331 ms | 0.376 ms | 0.167 ms |
| 64 KiB | 0.174 ms | 0.304 ms | 0.262 ms |
| 1 MiB | 0.217 ms | 0.224 ms | 0.359 ms |
| 8 MiB | 0.630 ms | 0.642 ms | 0.869 ms |

For comparison, the original pinned-host Gloo all-reduce measured 0.752 ms at
4 KiB, 0.659 ms at 64 KiB, 9.33 ms at 1 MiB, and 23.57 ms at 8 MiB. The
intermediate Windows shared-memory CPU-add path measured 0.200, 0.207, 0.342,
and 1.285 ms respectively.

Raw transfer probes measured roughly 56 GB/s for each individual pinned-memory
leg, 26.7 GB/s for an explicit GPU0-to-host-to-GPU1 path, and about 20.6 GB/s
per rank during simultaneous bidirectional exchange.

## D3D12 and RCCL 64 MiB comparison

These runs compare the native transports on the same two-GPU system. Effective
bandwidth divides the 64 MiB payload by end-to-end operation latency.

| Path | Average latency | Effective bandwidth |
| --- | ---: | ---: |
| D3D12 simultaneous bidirectional exchange, per GPU | 3.84 ms | 17.49 GB/s |
| D3D12 FP16 AllReduce | 4.63 ms | 14.50 GB/s |
| RCCL NET/Socket FP16 AllReduce | 36.73 ms | 1.83 GB/s |

For this payload the GPU-driven D3D12 AllReduce is about 7.9x faster than the
native RCCL socket transport. It uses D3D12 L0/system-memory cross-adapter heaps
and GPU copy engines; it does not copy tensor payloads on the CPU, but it is not
direct VRAM P2P.

## vLLM latency

The model benchmark used batch size 1, an 8-token input, one generated token,
and 30 measured iterations after warmup:

| Configuration | Average | p50 | p90 |
| --- | ---: | ---: | ---: |
| TP1, one R9700 | 77.75 ms | 77.49 ms | 78.97 ms |
| TP2, RCCL NET/Socket | 61.82 ms | 61.82 ms | 64.59 ms |
| TP2, D3D12 AllReduce + RCCL | 59.00 ms | 59.04 ms | 60.59 ms |

On this specific workload the hybrid reduced average latency by 24.1%, a 1.32x
speedup over TP1, and was about 4.6% faster than RCCL-only TP2. This is a small
model and one-token decode benchmark; larger models, batches, and contexts need
separate measurement.

## Current 24B AWQ W4A16 validation

The current reproducible checkpoint is
`stelterlab/Mistral-Small-24B-Instruct-2501-AWQ` at revision
`cbda099649a0188dd888d44f0e4964d8d982dc9a`. Its three weight shards total
14,234,370,648 bytes (13.26 GiB). The model metadata declares AWQ, 4-bit
weights, group size 128, asymmetric zero points, and GEMM layout.

Both one-GPU and two-GPU validation selected
`RDNAHybridW4A16LinearKernel for AutoAWQMarlinLinearMethod`. The TP2 run loaded
6.76 GiB of model memory per worker, retained about 21.5 GiB of KV cache per
GPU, and emitted D3D12 AllReduce plus RCCL AllGather routing traces. TP1 and TP2
returned identical token IDs and text for both deterministic prompts.

### Short decode and concurrency results

The warmed benchmark used 32 input tokens, 32 output tokens, three warmups, ten
measured iterations, BF16 activations, automatic `ROCM_ATTN`, and no prefix
cache. Approximate output tokens/s is `batch_size * 32 / average latency` and
includes prefill plus decode.

| Batch | Configuration | Average | p50 | p90 | Output tokens/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | TP1 eager/O0 | 0.9282 s | 0.9283 s | 0.9331 s | 34.47 |
| 1 | TP2 D3D12 + RCCL eager/O0 | 1.0773 s | 1.0697 s | 1.0994 s | 29.70 |
| 1 | TP2 RCCL eager/O0 | 0.8912 s | 0.8930 s | 0.8959 s | 35.91 |
| 1 | TP2 RCCL async + O1 | 0.8424 s | 0.8437 s | 0.8449 s | 37.99 |
| 8 | TP1 eager/O0 | 3.2915 s | 3.1502 s | 3.8286 s | 77.78 |
| 8 | TP2 RCCL eager/O0 | 2.4699 s | 2.4550 s | 2.5282 s | 103.65 |
| 8 | TP2 RCCL async + O1 | 2.3912 s | 2.3069 s | 2.4194 s | 107.06 |

At batch 8, tuned TP2 improved aggregate output throughput by 37.6% over the
TP1 eager baseline. Quantization changed the best transport for this workload:
RCCL-only beat the hybrid because the AWQ model's small reductions do not
amortize the D3D12 cross-adapter path.

The TP2 tuning A/B tests at batch 1 were:

| Change | Output tokens/s | Result versus tuned BF16/auto |
| --- | ---: | ---: |
| Async scheduling without Inductor | 36.43 | -4.1% |
| Async scheduling + O1, BF16, auto attention | 37.99 | reference |
| Change activations to FP16 | 36.15 | -4.8% |
| Force `TRITON_ATTN` | 35.67 | -6.1% |

The Windows platform keeps static graph mode disabled. To make Inductor O1
usable, collectives are presented as opaque `torch.ops.vllm` compiler
boundaries while their eager implementations continue to call the native RCCL
or D3D12 transport. The pinned vLLM nightly also requires
`--compilation-config '{"compile_sizes":[]}'` to enable dynamic-range lookup;
otherwise its piecewise backend rejects a valid 512-token shape.

## Qwen3.8-27B BF16 format trial

`Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` contains 18 BF16 weight shards
totaling 55,563,006,776 bytes (51.75 GiB). The model resolved to
`Qwen3_5ForConditionalGeneration` and ran with multimodal inputs disabled.
Each TP worker loaded 25.24 GiB of model memory, proving the checkpoint was
sharded because neither 31.86 GiB GPU can hold the full checkpoint.

All runs used 32 input and 32 output tokens, BF16, automatic `ROCM_ATTN`, no
prefix cache, a 512-token model limit, and 92% GPU-memory utilization.

| Batch | Configuration | Average | p50 | p90 | Output tokens/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | TP2 D3D12 + RCCL eager/O0 | 11.0946 s | 11.0987 s | 11.1249 s | 2.88 |
| 1 | TP2 RCCL eager/O0 | 10.5343 s | 10.4008 s | 10.8281 s | 3.04 |
| 1 | TP2 RCCL async + O1 | 10.2798 s | 10.2957 s | 10.3629 s | 3.11 |
| 1 | TP2 RCCL async eager + one-token MTP | 10.8185 s | 10.9276 s | 11.4470 s | 2.96 |
| 8 | TP2 RCCL async eager | 28.0533 s | 28.0294 s | 28.7700 s | 9.13 aggregate |

RCCL-only won for this workload. O1 plus async scheduling was the best tested
batch-1 profile, while the built-in MTP draft layer lost performance. The
ordinary O1 graph took 428.98 seconds to compile on its first launch. The
O1+MTP graph was not measured because its independent artifact exhausted the
remaining disk while being saved; eager MTP completed normally. The downloaded
BF16 checkpoint was removed after the trial, as required by the sequential
BF16, FP8, and 4-bit test plan.

## Historical BF16 large-model TP2 validation

The earlier large-model validation used
`mistralai/Mistral-Small-24B-Instruct-2501` at revision
`9527884be6e5616bdd54de542f9ae13384489724`, BF16, TP2, a 512-token model
limit, and 92% GPU-memory utilization. The ten selected safetensor shards are
47,144,848,872 bytes (43.91 GiB). That exceeds the 31.86 GiB physical capacity
of either individual R9700.

The run completed model initialization, deterministic generation, and clean
shutdown with these measurements:

| Observation | Result |
| --- | ---: |
| Checkpoint size reported by vLLM | 43.91 GiB |
| Model memory loaded per TP worker | 21.96 GiB |
| Weights plus non-torch memory per GPU | 22.11 GiB |
| Peak activation memory per GPU | 0.45 GiB |
| KV cache per GPU | 6.76 GiB |
| Full `LLM(...)` initialization | 61.07 s |
| Two prompts, 16 output tokens each | 5.97 s |

The first generation includes Triton JIT compilation and is not a steady-state
throughput benchmark. Rank 0 and rank 1 both emitted transport traces for
D3D12 AllReduce and RCCL AllGather. The generated continuations were:

- `The capital of France is` → `Paris. It is known for its iconic landmarks
  such as the Eiffel Tower`
- `Two plus two equals` → `four. This is a mathematical fact. It is not a
  matter of opinion.`

An initial uncapped vLLM profile exposed an 80 MiB AllReduce, larger than the
default 64 MiB D3D12 heap. The hybrid now routes such tensors to RCCL. The
targeted two-rank overflow probe passed exact FP16 values with 1 MiB on D3D12
and 80 MiB on RCCL.

### Historical 24B BF16 transport speed comparison

The end-to-end latency benchmark used the same model and memory settings,
batch size 1, 32 input tokens, 16 output tokens, 3 warmup iterations, and 10
measured iterations in eager/O0 mode:

| Transport | Average | p50 | p90 | Approx. output tokens/s |
| --- | ---: | ---: | ---: | ---: |
| D3D12 AllReduce + RCCL | 1.5303 s | 1.5381 s | 1.5439 s | 10.46 |
| RCCL NET/Socket only | 2.5938 s | 2.5840 s | 2.6656 s | 6.17 |

In this paired run the hybrid reduced average latency by 41.0%, a 1.69x
speedup over RCCL-only. Approximate output tokens/s divides the 16 generated
tokens by the end-to-end batch latency; it includes prefill and decode. The raw
benchmark command is reproduced in the README, and repeated runs are advised
before comparing hardware or driver versions.
