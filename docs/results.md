# Reference results

Reference machine: Windows 11, two AMD Radeon AI PRO R9700 (`gfx1201`) GPUs.

## Pinned stack

- Python 3.12
- PyTorch `2.11.0+rocm7.15.0a20260728`
- HIP `7.15.26290`
- AMD ROCm wheel train `7.15.0a20260728`
- vLLM `0.1.dev19487+gfb9fb8c5a.rocm715`, source commit
  `fb9fb8c5aeaed96c91eef5cb48743a96f8496907`

`torch.distributed` and Gloo are available; NCCL/RCCL is not. Two GPUs are
visible and peer access is false in both directions.

## Correctness

Two-rank probes pass CPU Gloo all-reduce; native FP32, FP16, and BF16
shared-memory all-reduce; host-staged all-reduce, all-gather, gather,
reduce-scatter, broadcast, and point-to-point. Direct GPU Gloo is intentionally
excluded because it causes a native Windows access violation.

The vLLM reference test used `Qwen/Qwen3-0.6B`, two prompts, deterministic
sampling, and eight output tokens. TP1 and TP2 returned identical token IDs:

| Prompt | Token IDs from both TP1 and TP2 |
| --- | --- |
| `The capital of France is` | `12095, 13, 576, 6722, 315, 9625, 374, 1083` |
| `Two plus two equals` | `1378, 11, 323, 1378, 5519, 1378, 16819, 1378` |

Each TP2 worker loaded about 0.57 GiB of model weights versus about 1.12 GiB
for TP1, confirming that the model was sharded rather than replicated.

## All-reduce transport latency

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

## vLLM latency

The model benchmark used batch size 1, an 8-token input, one generated token,
and 30 measured iterations after warmup:

| Configuration | Average | p50 | p90 |
| --- | ---: | ---: | ---: |
| TP1 | 77.44 ms | 77.54 ms | 78.77 ms |
| TP2 | 60.68 ms | 60.66 ms | 62.51 ms |

On this specific workload TP2 reduced average latency by 21.6%, a 1.28x
speedup. Larger models and contexts need separate measurement.
