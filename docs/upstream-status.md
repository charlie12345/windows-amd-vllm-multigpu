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

## Native Windows RCCL status

- TheRock's Windows support matrix marks RCCL unsupported:
  <https://github.com/ROCm/TheRock/blob/main/docs/development/windows_support.md>
- RCCL upstream source and issue tracker:
  <https://github.com/ROCm/rccl>
- The Radeon AI PRO R9700 also has a currently tracked RCCL issue on Linux:
  <https://github.com/ROCm/rccl/issues/5480>

The pinned AMD Windows wheel train does not ship RCCL. This repository now
ports the pinned RCCL 2.30.7 source to a native Windows DLL, establishes
multi-process bootstrap through Gloo, validates collective kernels on
`gfx1201`, and integrates the data plane directly into vLLM. It does not patch
the user's vLLM checkout or claim that PyTorch c10d gained an RCCL backend.

The current driver exposes neither HIP peer access nor cross-device HIP IPC,
so the validated RCCL topology uses NET/Socket through system memory. The next
upstream boundary is a Windows driver/runtime mechanism for importing
cross-adapter device-backed memory. D3D12 and Vulkan external-memory probes are
included to test that boundary explicitly.

## D3D12 and Vulkan external-memory status

D3D12 cross-adapter heaps and fences work on the reference GPUs in one process
and across the two vLLM worker processes. HIP can import each D3D12 heap and
fence on both GPUs. That enables the validated GPU-driven AllReduce fast path,
but Microsoft specifies cross-adapter heaps as system-memory allocations on
discrete adapters rather than VRAM P2P:

- D3D12 shared heaps and cross-adapter restrictions:
  <https://learn.microsoft.com/en-us/windows/win32/direct3d12/shared-heaps>
- D3D12 `D3D12_MEMORY_POOL_L0` definition:
  <https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_memory_pool>

Vulkan external-memory import succeeds on the owning GPU but not the peer GPU
with the pinned driver. Re-Size BAR and Above-4G decoding remain useful platform
settings, but they do not override the driver/runtime peer-access capability.

## HIP primitives used by this prototype

- HIP host-memory registration and mapped-host APIs:
  <https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory.html>
- HIP stream memory operations:
  <https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/stream_management.html>

The stream write/wait APIs are marked Beta. The repository therefore treats
this implementation as experimental and documents the need for watchdog and
fault testing before production use.

## Ling 3.0 and MLA prefill

vLLM v0.28 includes `BailingMoeV3ForCausalLM`, Ling reasoning/tool parsing,
FP8, routed-expert MXFP4, KDA fixes, and DFlash2 support. The pinned Windows
fork is rebased on that release. This plugin no longer applies the former Ling
or DFlash patch queue.

The Windows fork adds `TRITON_MLA` after AITER FlashAttention and
FlashAttention. On the reference RDNA 4 Windows stack, AITER rejects the GPU
family and ROCm FlashAttention is unavailable, so Triton is selected third. It
was numerically checked on `gfx1201`, including the v0.28 chunked-context API,
and used by successful TP1 and TP2 Ling dummy-weight generation. Tuned
AITER/FlashAttention remain preferable where available.

The real-weight O1 benchmark also exposed a compile dispatcher bug: dynamic
compile ranges were never searched when `compile_sizes` was unset. The pinned
Windows fork removes that early return and includes a focused regression test.
Both generally useful changes landed through
[`vLLM_for_AMD` PR #26](https://github.com/charlie12345/vLLM_for_AMD/pull/26),
not through the transport plugin.
