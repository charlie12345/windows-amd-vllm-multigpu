# Architecture and decision record

## Facts established on the reference machine

- Windows sees two AMD Radeon AI PRO R9700 GPUs (`gfx1201`).
- The stable PyTorch `2.11.0+rocm7.13.0` wheel was compiled with
  `USE_GLOO=OFF`, `USE_NCCL=OFF`, and no usable c10d extension.
- AMD TheRock enabled Gloo for its PyTorch production-wheel environment on
  2026-06-16.
- TheRock's Windows support matrix still marks RCCL unsupported.
- HIP/PyTorch reports `can_device_access_peer(0, 1) == False` and the reverse
  direction is also false on the reference machine.
- Passing a HIP tensor directly to Gloo on Windows terminates both ranks with
  native exception `0xC0000005` in `ProcessGroupGloo::enqueue`. This matches
  [pytorch/pytorch#186535](https://github.com/pytorch/pytorch/issues/186535),
  where the Windows Gloo all-reduce registry has no GPU-tensor creator. Direct
  GPU/Gloo calls are therefore never made by the safe probes.

## Staged design

### Stage 1: control plane

Use PyTorch c10d with Gloo for rendezvous, stores, process groups, object
broadcasts, barriers, and CPU tensor collectives. This is already an upstream
Windows-capable PyTorch path and avoids inventing a process-group API.

### Stage 2: correctness data plane

Each rank owns one GPU. A collective copies the GPU tensor into a reusable
pinned CPU buffer, performs the Gloo collective on that buffer, then copies the
result back to the rank's GPU. This is true multi-process/multi-GPU execution,
but the data plane crosses system memory.

### Stage 3: optimized Windows data plane

Benchmark and pursue these options in order:

1. overlapped device-to-host, Gloo, and host-to-device transfers;
2. shared-memory ring buffers that reduce Gloo copying overhead;
3. HIP IPC and peer mappings only if the runtime and topology prove they work;
4. a native Windows RCCL port as a separate upstream-scale effort.

### Stage 4: vLLM adapter

Export an explicit patch or plugin that selects the Windows AMD communicator.
The adapter must not silently modify another checkout. Tensor-parallel
correctness comes first; pipeline parallelism is evaluated separately because
it communicates fewer, larger activation tensors and may suit a host-staged
transport better.

## Success criteria

- Two ranks initialize and shut down repeatedly without hanging.
- CPU and host-staged GPU collectives produce bitwise-correct small results and
  tolerance-correct floating-point stress results.
- Failures have bounded timeouts and terminate all child processes.
- A two-GPU model produces the same output as a one-GPU reference.
- Benchmarks report latency/bandwidth honestly; no "RCCL" or "peer-to-peer"
  claim is made unless those paths are actually active.
