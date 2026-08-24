# Architecture and decision record

## Established facts

- Windows sees two AMD Radeon AI PRO R9700 GPUs (`gfx1201`).
- PyTorch `2.11.0+rocm7.15.0a20260728` provides c10d and Gloo, but not NCCL/RCCL.
- HIP reports no peer access in either GPU direction on the reference machine.
- Cross-device `hipIpcOpenMemHandle` fails, so RCCL cannot use its P2P/IPC
  transport on this driver.
- Passing a HIP tensor directly to Windows Gloo terminates the worker with
  exception `0xC0000005`; PyTorch's Windows build omits the Gloo CUDA-like
  implementation and leaves no registered GPU-tensor collective creator.
- A Windows named mapping can be registered and mapped into both HIP processes.
- HIP stream memory write/wait operations observe flags in that shared mapping.

The repository now builds RCCL 2.30.7 as a native PE DLL. It uses GPU
collective kernels while selecting NET/Socket and pinned system-memory buffers
for the inter-GPU data path. This is deliberately distinguished from
GPU-direct peer VRAM access.

## Hybrid D3D12 fast path

The Windows driver exposes D3D12 cross-adapter heaps even though HIP peer
access and HIP IPC fail. Each rank creates a named cross-adapter heap and
timeline fence on its matching DXGI adapter. The peer process opens both by
name, re-exports adapter-owned handles, and imports the heap and fence through
HIP external-memory/semaphore APIs.

For each two-rank AllReduce, both ranks enqueue the following on the current
PyTorch HIP stream:

1. copy the local tensor from VRAM to that rank's D3D12 heap;
2. signal the rank's cross-adapter timeline fence;
3. wait for the peer's fence;
4. copy the peer heap into a temporary local-VRAM tensor;
5. launch the FP16, FP32, or BF16 local SUM kernel;
6. signal consumption and wait until the peer consumed the local heap.

No CPU thread copies tensor payloads or blocks in the hot sequence. D3D12
cross-adapter heaps are nevertheless L0/system-memory allocations on discrete
adapters, so the project calls this a GPU-driven cross-adapter path rather than
VRAM P2P or GPU-direct. Direct HIP-kernel loads from the peer external pointer
did not provide coherent values on this driver; the validated implementation
uses the HIP copy engine into local VRAM before reduction.

## Native Windows RCCL port

The pinned patch replaces Linux-only build, topology, socket, dynamic-loading,
threading, and optional-plugin assumptions with Win32 equivalents. The Windows
build omits Linux RDMA/HSA components and the standalone RAS client, exports
the public C ABI from `rccl.dll`, and targets `gfx1201` using the ROCm SDK
wheels. Gloo broadcasts RCCL's 128-byte unique ID; RCCL owns the data-plane
collectives after communicator initialization.

Windows uses the LLP64 C data model: `unsigned long` and HIP's `ulong2` are
32-bit-lane/8-byte objects, unlike Linux's 64-bit `unsigned long`. RCCL assumed
a 16-byte `ulong2` while copying its device kernel work descriptor. That
skipped alternate words and corrupted ring-peer indexes. The port uses fixed
width 128-bit storage (`uint4`/`ulonglong2`) for those structures and includes a
native ABI probe so this regression is reproducible.

The Python wrapper loads `rccl.dll` with `ctypes`, initializes one communicator
per vLLM tensor-parallel group, and launches AllReduce, AllGather,
ReduceScatter, and Broadcast directly on PyTorch's current HIP stream. PyTorch
c10d remains Gloo because the pinned Windows PyTorch build has no RCCL backend;
the adapter bypasses c10d only for the GPU data plane. When both fast paths are
enabled, D3D12 handles AllReduce while RCCL handles AllGather, ReduceScatter,
and Broadcast.

## Mapped-host fallback process and memory layout

vLLM launches one worker per GPU. Rank 0 creates a named Windows file mapping;
rank 1 opens it. Each process calls `hipHostRegister` and
`hipHostGetDevicePointer` on the mapping, so its GPU can access the same pinned
host pages.

The mapping contains one fixed payload slot per rank and a separately aligned
control area. Ready and consumed epochs are placed at least 64 bytes apart to
avoid sharing a cache line. Both ranks agree on transport capability through a
small CPU Gloo collective before enabling the fast path.

## Stream-ordered fallback all-reduce

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

This fallback also crosses PCIe/system memory and is not peer-to-peer. It
preserves correct GPU stream ordering and lets both GPUs reduce concurrently.
Unsupported dtypes and operations fall back to pinned-host Gloo.

## vLLM integration boundary

The package exposes a vLLM platform plugin and external communicator. Launch
scripts opt it in explicitly with:

```text
WAVMG_ENABLE=1
VLLM_PLUGINS=windows_amd_multigpu
```

The plugin subclasses vLLM's ROCm platform, selects Gloo for c10d control,
disables incompatible built-in custom collectives/static graphs, and supplies
the native RCCL communicator when `WAVMG_USE_RCCL=1`. With
`WAVMG_USE_D3D12=1`, the D3D12 communicator takes priority for two-rank
AllReduce. Without either variable it uses the mapped-host all-reduce. It
currently validates only TP1/TP2 with PP1 and no additional data,
decode-context, prefill-context, or expert parallel groups.

The vLLM clone lives under ignored `sandbox\vllm`. A pinned patch makes three
general Windows compatibility fixes: accepts `PipeConnection`, guards the
missing `os.sched_yield`, and skips Linux `/proc/self/maps` cleanup. The patch
does not add this transport to the user's vLLM tree.

## Limits and next work

The working prototype synchronizes streams before unregistering mappings and
has CPU-side process timeouts in its probes. Before calling it production
ready, it still needs a GPU-wait watchdog/peer-death escape, size-based
autotuning, more native non-all-reduce collectives, long-duration fault and
epoch stress, and broader model validation.

A custom PyTorch ProcessGroup could make the native communicator transparent
to software outside vLLM. True VRAM P2P still requires a driver-supported peer
allocation/import mechanism. D3D12 external sharing works but is specified as
system memory; Vulkan external memory imports only on the owning GPU on this
stack. The repository keeps each capability probe explicit rather than
inferring that ReBAR alone provides access.
