# Reference results

Reference machine: Windows 11, two AMD Radeon AI PRO R9700 (`gfx1201`) GPUs.
Results below use PyTorch `2.11.0+rocm7.15.0a20260728` and HIP `7.15.26290`.

## Capability result

- `torch.distributed.is_available()`: `true`
- `torch.distributed.is_gloo_available()`: `true`
- `torch.distributed.is_nccl_available()`: `false`
- GPU count: `2`
- GPU peer access: `false` in both directions

The wheel reports `USE_GLOO=ON` and `USE_NCCL=OFF` in its build configuration.

## Two-rank correctness

The following operations passed on both ranks:

| Operation | Expected/result |
| --- | --- |
| CPU Gloo all-reduce | `1 + 2 = 3` |
| Host-staged GPU all-reduce | `1 + 2 = 3` |
| Host-staged GPU all-gather | `[1, 2, 3, 4]` |
| Host-staged GPU reduce-scatter | `[3, 3]` per rank |
| Host-staged GPU broadcast | `7` per rank |
| Host-staged GPU point-to-point | rank 1 received `11` |

Directly passing a HIP tensor to Gloo is excluded from normal probes because it
terminates the process with Windows exception `0xC0000005`.

## Baseline all-reduce performance

This measures a synchronous pinned-host copy, Gloo all-reduce, and copy back to
each GPU. "Logical MiB/s" is payload size divided by wall time; it is not a
claim about PCIe wire bandwidth.

| Payload | Latency | Logical throughput |
| ---: | ---: | ---: |
| 4 KiB | 0.752 ms | 5.20 MiB/s |
| 64 KiB | 0.659 ms | 94.9 MiB/s |
| 1 MiB | 9.33 ms | 107 MiB/s |
| 8 MiB | 23.57 ms | 339 MiB/s |

These are correctness-first baseline numbers, not the performance target.

