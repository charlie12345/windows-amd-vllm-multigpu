# Upstream status and sources

## Windows PyTorch and Gloo

- AMD's TheRock enabled Gloo in its Windows PyTorch production-wheel pipeline:
  <https://github.com/ROCm/TheRock/pull/5694>
- TheRock's PyTorch build configuration enables Gloo:
  <https://github.com/ROCm/TheRock/blob/main/external-builds/pytorch/build_prod_wheels.py>
- PyTorch tracks the fatal Windows direct-Gloo/GPU-tensor registry failure:
  <https://github.com/pytorch/pytorch/issues/186535>

Source inspection shows that `ProcessGroupGlooCuda.cpp` is omitted on Windows.
The collective creator lookup can consequently return null for a GPU tensor,
and the enqueue path dereferences it. Compiling that source and adding a null
check would close the crash, but Gloo would still be a host-staged transport,
not an AMD GPU collective library.

## RCCL gap

- TheRock's Windows support matrix marks RCCL unsupported:
  <https://github.com/ROCm/TheRock/blob/main/docs/development/windows_support.md>
- RCCL upstream source and issue tracker:
  <https://github.com/ROCm/rccl>
- The Radeon AI PRO R9700 also has a currently tracked RCCL issue on Linux:
  <https://github.com/ROCm/rccl/issues/5480>

A Windows RCCL effort would need more than a compiler switch. It must replace
or port Linux/HSA process, shared-memory, topology, device-discovery, and build
assumptions; establish Windows multi-process bootstrap; validate kernels on
`gfx1201`; and then integrate with PyTorch c10d. This remains worthwhile, but
it is an upstream-scale project rather than a prerequisite for proving useful
vLLM tensor parallelism.

## HIP primitives used by this prototype

- HIP host-memory registration and mapped-host APIs:
  <https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory.html>
- HIP stream memory operations:
  <https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/stream_management.html>

The stream write/wait APIs are marked Beta. The repository therefore treats
this implementation as experimental and documents the need for watchdog and
fault testing before production use.
