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

The installed `bin\rccl.dll` is the shim. The real Windows RCCL DLL is copied
beside it as `bin\rccl-real.dll` and loaded explicitly by the shim.

## Runtime modes

`WAC_MODE` selects the route:

- `hybrid`: initialize both transports and route by payload size;
- `d3d12`: use D3D12 for supported two-rank FP16, FP32, and BF16 SUM calls;
- `rccl`: delegate every call to `rccl-real.dll`.

The provisional hybrid defaults route payloads from 0 through 32767 bytes to
D3D12 and larger payloads to Direct-RCCL. Override them with
`WAC_D3D12_MIN_BYTES`, `WAC_D3D12_MAX_BYTES`, and
`WAC_D3D12_BUFFER_BYTES` after measuring the target machine.

`WAC_RCCL_DISPATCH=threaded` is the default for the single-process/two-rank
llama call pattern. It launches both RCCL rank calls concurrently. Set it to
`native-group` only as an A/B comparator.

Build and launch instructions are in
[`docs/install-llama-rocmfpx.md`](../../docs/install-llama-rocmfpx.md).
