# llama.cpp / ROCmFPX RCCL shim

This adapter exposes the small NCCL/RCCL ABI used by the tested llama.cpp HIP
backend and routes two-GPU AllReduce calls between the native Windows RCCL DLL
and the D3D12 cross-adapter path. It is an external CMake package; it does not
patch or overlay llama.cpp source.

The shim exports exactly these symbols:

- `ncclCommInitAll`
- `ncclCommDestroy`
- `ncclGetErrorString`
- `ncclGroupStart`
- `ncclGroupEnd`
- `ncclAllReduce`

This is not a general RCCL replacement. In particular, do not load it as the
RCCL backend for PyTorch or vLLM. Those multi-process clients require the full
Windows RCCL build and additional API entry points.

> [!CAUTION]
> This is experimental development software. A collective, HIP, D3D12, or
> driver failure can hang the process or desktop and may require a restart.
> Save other work and run the small validation probes before loading a model.

The installed `bin\rccl.dll` is the shim. The real Windows RCCL DLL is copied
beside it as `bin\rccl-real.dll` and loaded explicitly by the shim.

The D3D12 route uses a custom HIP publish kernel, D3D12 timeline fences, and a
fused reduction kernel that reads the peer external heap. This avoids ROCm
10's failing `hipMemcpyAsync` classification of imported D3D12 pointers.

## Runtime modes

`WAC_MODE` selects the route:

- `hybrid`: use the safe D3D12 route by default and allow an explicitly enabled
  experimental Direct-RCCL overflow route;
- `d3d12`: use D3D12 for supported two-rank FP16, FP32, and BF16 SUM calls;
- `rccl`: request the experimental same-process `rccl-real.dll` route.

The safe hybrid defaults route payloads from 0 through 64 MiB to D3D12. Override
the allocation and routing bounds with
`WAC_D3D12_MIN_BYTES`, `WAC_D3D12_MAX_BYTES`, and
`WAC_D3D12_BUFFER_BYTES` after measuring the target machine.

Direct-RCCL is intentionally disabled by default for llama.cpp. ROCm 10's
same-process `ncclCommInitAll` path produced corrupt tokens with concurrent rank
threads and stalled with native grouping on the tested dual-R9700 machine.
Set `WAC_LLAMA_RCCL_EXPERIMENTAL=1` only for development A/B tests. This does
not affect vLLM, which uses the full RCCL DLL with one process per GPU.

Build ROCmFPX/llama.cpp with `GGML_CUDA_NO_PEER_COPY=ON`, then run tensor
parallelism with `--split-mode tensor --tensor-split 1,1 --no-mmap`. These are
correctness requirements for the tested non-P2P Windows R9700 pair.

The source can be compiled for other supported ROCm GPU targets by selecting
the matching architecture. Only the dual-R9700 `gfx1201` configuration has been
runtime-qualified; `gfx1100` (including RX 7900 XTX) and other selectable
targets require their own coherency and performance validation.

Build and launch instructions are in
[`docs/install-llama-rocmfpx.md`](../../docs/install-llama-rocmfpx.md).
