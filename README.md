# Windows AMD vLLM Multi-GPU

Experimental native-Windows tensor parallelism for AMD GPUs and vLLM. This is
an independent repository: it does not modify or build inside the Windows AMD
vLLM port at `C:\AI\vllm`.

## Milestone reached

The reference machine now runs a real vLLM tensor-parallel model across two
Radeon AI PRO R9700 GPUs and two worker processes. Qwen3-0.6B produced exactly
the same token IDs with tensor parallel size 1 and 2. For an 8-token prompt and
one generated token, the measured average latency was:

| Configuration | Average latency | Relative result |
| --- | ---: | ---: |
| TP1, one R9700 | 77.44 ms | baseline |
| TP2, two R9700s | 60.68 ms | 1.28x faster |

This is true model sharding and concurrent GPU execution, but it is not GPU
peer-to-peer or RCCL. The GPUs report no HIP peer access on this Windows
machine, and AMD's Windows ROCm stack does not currently provide RCCL. The
implemented fast path uses HIP-registered Windows shared memory plus
stream-ordered GPU waits, copies, and fused reduction kernels. Gloo remains the
control plane and the fallback for non-all-reduce collectives.

## Reproduce the transport

From a PowerShell prompt:

```powershell
.\scripts\bootstrap-nightly.ps1
.\scripts\build-native.cmd
.\scripts\run-probes.ps1
```

This creates only `.venv` and ignored build output inside this repository.

## Reproduce vLLM TP2

The vLLM bootstrap creates an isolated `.venv-vllm`, clones the pinned Windows
AMD vLLM fork into ignored `sandbox\vllm`, applies the version-pinned Windows
compatibility patch, builds it, and installs this plugin:

```powershell
.\scripts\bootstrap-vllm.ps1
.\scripts\run-vllm-tp2.ps1
```

Compare one and two GPUs with:

```powershell
.\scripts\run-vllm-tp2.ps1 -TensorParallelSize 1
.\scripts\run-vllm-tp2.ps1 -TensorParallelSize 2
```

The validated vLLM source commit is recorded in
`pins/nightly-2026-07-28.json`. The adapter deliberately rejects unsupported
layouts: TP is currently limited to 1 or 2, while PP, DP, DCP, PCP, and EP must
remain 1 or disabled. The shared mapping defaults to 64 MiB per rank and can be
changed with `WAVMG_SHM_MAX_BYTES` before launch.

The fork's Windows installer intentionally omits ROCm's Linux-only and optional
feature packages. The bootstrap installs the dependency set exercised by the
validated text-generation path; quantization, model streaming, speculative
decoding, multimodal models, and other optional vLLM features are not yet part
of this compatibility claim.

## Repository layout

- `native/` contains the HIP shared-memory reduction DLL.
- `src/` contains the transport and vLLM platform/communicator plugin.
- `probes/` contains correctness, capability, performance, and model tests.
- `scripts/` contains repeatable setup, build, patch, and launch commands.
- `patches/` contains only explicit, version-pinned vLLM compatibility patches.
- `pins/` records the exact successful PyTorch, ROCm, and vLLM versions.

See [docs/architecture.md](docs/architecture.md),
[docs/results.md](docs/results.md), and
[docs/upstream-status.md](docs/upstream-status.md) for design details, full
measurements, and the remaining RCCL gap.
