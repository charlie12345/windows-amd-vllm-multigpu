# Architecture and decision record

## Established facts

- Windows sees two AMD Radeon AI PRO R9700 GPUs (`gfx1201`).
- PyTorch `2.11.0+rocm7.15.0a20260728` provides c10d and Gloo, but not NCCL/RCCL.
- HIP reports no peer access in either GPU direction on the reference machine.
- Passing a HIP tensor directly to Windows Gloo terminates the worker with
  exception `0xC0000005`; PyTorch's Windows build omits the Gloo CUDA-like
  implementation and leaves no registered GPU-tensor collective creator.
- A Windows named mapping can be registered and mapped into both HIP processes.
- HIP stream memory write/wait operations observe flags in that shared mapping.

The design therefore uses the fastest capability actually present instead of
claiming a GPU-direct path.

## Process and memory layout

vLLM launches one worker per GPU. Rank 0 creates a named Windows file mapping;
rank 1 opens it. Each process calls `hipHostRegister` and
`hipHostGetDevicePointer` on the mapping, so its GPU can access the same pinned
host pages.

The mapping contains one fixed payload slot per rank and a separately aligned
control area. Ready and consumed epochs are placed at least 64 bytes apart to
avoid sharing a cache line. Both ranks agree on transport capability through a
small CPU Gloo collective before enabling the fast path.

## Stream-ordered two-rank all-reduce

For epoch `e`, each rank enqueues this sequence on its current PyTorch/HIP
stream:

1. wait until the peer has consumed this rank's previous slot contents;
2. copy the local GPU tensor into the rank's mapped-host slot;
3. publish this rank's ready epoch with `hipStreamWriteValue32`;
4. wait for the peer's ready epoch with `hipStreamWaitValue32`;
5. launch a fused FP32, FP16, or BF16 kernel that adds the peer slot directly
   into the local GPU tensor;
6. publish that the peer slot has been consumed.

The native DLL enqueues the whole sequence in one call. There is no host
synchronization in the hot path. Epoch comparisons use greater-than-or-equal
semantics and the Python layer prevents unsafe 32-bit wraparound. Its filename
is ABI-versioned so Windows can link an updated build beside an older DLL that
is still mapped by a worker process.

This crosses PCIe/system memory, so it is not RCCL or peer-to-peer. It does,
however, preserve correct GPU stream ordering and lets both GPUs reduce
concurrently. Unsupported dtypes and operations fall back to pinned-host Gloo.

## vLLM integration boundary

The package exposes a vLLM platform plugin and external communicator. Launch
scripts opt it in explicitly with:

```text
WAVMG_ENABLE=1
VLLM_PLUGINS=windows_amd_multigpu
```

The plugin subclasses vLLM's ROCm platform, selects Gloo for c10d control,
disables incompatible custom collectives/static graphs, and supplies the
shared-memory all-reduce. It currently validates only TP1/TP2 with PP1 and no
additional data, decode-context, prefill-context, or expert parallel groups.

The vLLM clone lives under ignored `sandbox\vllm`. A pinned patch makes three
general Windows compatibility fixes: accepts `PipeConnection`, guards the
missing `os.sched_yield`, and skips Linux `/proc/self/maps` cleanup. The patch
does not add this transport to the user's vLLM tree.

## Safety limits and next work

The working prototype synchronizes streams before unregistering mappings and
has CPU-side process timeouts in its probes. Before calling it production
ready, it still needs a GPU-wait watchdog/peer-death escape, size-based
autotuning, more native non-all-reduce collectives, long-duration fault and
epoch stress, and broader model validation.

A custom PyTorch ProcessGroup could eventually make this usable outside vLLM.
A genuine Windows RCCL port remains a separate, substantially larger effort
involving RCCL, HIP/HSA build assumptions, process bootstrap, topology, and
upstream maintenance.
