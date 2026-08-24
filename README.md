# Windows AMD vLLM Multi-GPU

Experimental native-Windows multi-GPU support for AMD GPUs and vLLM. This is
intentionally separate from the Windows AMD vLLM port: this repository must
prove the distributed substrate independently before it exports any vLLM
integration patch.

## Current status

The pinned post-fix AMD wheel provides real c10d/Gloo. Two Windows ranks now
pass all-reduce, all-gather, reduce-scatter, broadcast, and point-to-point tests
with tensors explicitly staged through pinned CPU memory. The first transport
benchmark is recorded in [docs/results.md](docs/results.md).

The next milestone is to replace Gloo's bulk data movement with a faster
Windows shared-memory/pinned-memory path while retaining Gloo for coordination.

RCCL is currently unsupported by AMD's Windows ROCm build. The two local
Radeon AI PRO R9700 GPUs also report no HIP peer access in either direction.
The first implementation therefore targets correctness using pinned CPU
buffers and Gloo. It does not claim RCCL-class performance.

## Quick start

From a PowerShell prompt:

```powershell
.\scripts\bootstrap-nightly.ps1
.\scripts\run-probes.ps1
```

The bootstrap installs only into this repository's `.venv`; it does not alter
the vLLM environment in `C:\AI\vllm`.

## Repository boundaries

- `pins/` records exact upstream versions used by successful experiments.
- `probes/` contains independently runnable correctness and capability tests.
- `scripts/` contains reproducible Windows setup/build entry points.
- `src/` will contain the transport implementation after the probes pass.
- `patches/` will eventually contain generated integration patches; it will
  never edit a vLLM checkout implicitly.

See [docs/architecture.md](docs/architecture.md) for the staged design.
