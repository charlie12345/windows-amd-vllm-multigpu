# Install the bridge in ROCmFPX / llama.cpp on Windows

The llama adapter is an external `roc::rccl` CMake package. A clean upstream
llama.cpp or ROCmFPX checkout consumes it through `find_package(rccl)`; no
llama source patch or overlay is required.

This clean integration is possible because current upstream ggml already
provides `GGML_HIP_RCCL`, `find_package(rccl)`, the six-symbol NCCL ABI, and
the `GGML_CUDA_ALLREDUCE=nccl` runtime selector. The adapter supplies that ABI
as an external package. If `git status --short` in the llama/ROCmFPX source
tree is non-empty, do not describe that checkout as the clean integration.

The current implementation is for llama's single-process, two-rank
`ncclCommInitAll` path. It must not be used as the vLLM/PyTorch RCCL DLL.

> [!CAUTION]
> This is experimental development software. A bad collective or driver state
> can hang the process or desktop, strand VRAM, or require a restart. Save other
> work, close unrelated GPU applications, and pass the small probes before
> loading a large model. The launcher refuses to start unless Windows reports
> two healthy AMD display adapters; do not bypass that gate for benchmarks.

## Build the adapter from this repository

First build full Windows RCCL with this repository's vLLM instructions, or
provide another compatible staged Windows RCCL package. With the repository's
own build output:

```powershell
.\scripts\build-llama-rccl-shim.ps1 `
    -RocmRoot G:\ROCm\10.0.0-gfx120x `
    -RcclDll .\build\rccl-windows\rccl.dll `
    -RcclIncludeDirectory .\build\rccl-windows\include `
    -InstallDirectory G:\AI-plugins\wavmg-llama-rocm10-gfx1201 `
    -Jobs 16
```

For an already staged RCCL package, pass its `bin\rccl.dll` and `include`
directory instead. Keep the RCCL and HIP runtime versions together as a tested
set; a successful link is not proof that mixed runtime generations are safe.

The install contains:

```text
bin\rccl.dll          # six-symbol llama shim
bin\rccl-real.dll     # full Windows RCCL used by the shim
lib\rccl.lib
lib\cmake\rccl\...
include\nccl.h
run-with-llama-plugin.ps1
```

## Build an unchanged llama.cpp / ROCmFPX checkout

```powershell
$rocm = 'G:\ROCm\10.0.0-gfx120x'
$plugin = 'G:\AI-plugins\wavmg-llama-rocm10-gfx1201'
$source = 'C:\AI\rocmfpx-upstream-merge-20260827'
$build = 'G:\AI-builds\rocmfpx-rocm10-plugin'

$env:ROCM_PATH = $rocm
$env:HIP_PATH = $rocm
$env:HIP_PLATFORM = 'amd'
$env:HIP_DEVICE_LIB_PATH = Join-Path $rocm 'lib\llvm\amdgcn\bitcode'
$env:PATH = "$(Join-Path $rocm 'bin');$(Join-Path $rocm 'lib\llvm\bin');$env:PATH"

cmake -S $source -B $build -G Ninja `
    -DGGML_HIP=ON `
    -DGGML_HIP_RCCL=ON `
    -DGGML_CUDA_NO_PEER_COPY=ON `
    -DGPU_TARGETS=gfx1201 `
    "-DCMAKE_PREFIX_PATH=$rocm;$plugin" `
    "-Drccl_DIR=$plugin\lib\cmake\rccl"
cmake --build $build --config Release --parallel 16
```

Inspect `CMakeCache.txt` and the `ggml-hip` link command to verify that
`rccl_DIR` resolves to the plugin, not to a stale ROCm or build-tree package.

The example targets the validated R9700 architecture, `gfx1201`. For a different
GPU, select its actual ROCm architecture consistently in the adapter and llama
build. The scripts expose RDNA 2/3/3.5/4 targets, including RX 7900 XTX as
`gfx1100`, but only dual-R9700 `gfx1201` has prior end-to-end validation.

## Launch

Use the installed launcher so the shim precedes the ROCm directory on `PATH`:

```powershell
G:\AI-plugins\wavmg-llama-rocm10-gfx1201\run-with-llama-plugin.ps1 `
    -Executable G:\AI-builds\rocmfpx-rocm10-plugin\bin\llama-server.exe `
    -RocmRoot G:\ROCm\10.0.0-gfx120x `
    -Mode hybrid `
    --model G:\models\model.gguf `
    --split-mode tensor `
    --tensor-split 1,1 `
    --no-mmap
```

The launcher sets `GGML_CUDA_ALLREDUCE=nccl`; llama uses that historical
environment-variable name for its HIP NCCL/RCCL selection too.

`GGML_CUDA_NO_PEER_COPY=ON` is required on the tested Windows R9700 pair.
Without it, upstream attempts `hipMemcpyPeerAsync` even though HIP reports no
peer access; both ordinary layer splitting and tensor splitting can then
silently produce corrupt output. Upstream falls back to correctness-first host
staging for non-collective cross-device copies when this option is enabled.

Use `--split-mode tensor`, not `row`: the six-symbol communicator hook belongs
to the current meta-backend tensor-parallel path. `row` uses the older split
buffer interface. Keep `--no-mmap` on Windows ROCm tensor splits because
mmap-backed device slices have produced silent weight corruption.

Startup should print `wavmg-llama-rccl-shim: initialized ranks=2`. The safe
default prints `Direct-RCCL=off` and routes supported FP16, FP32, and BF16 SUM
payloads through D3D12 up to the 64 MiB buffer limit.

Direct-RCCL remains validated for vLLM's process-per-GPU design. ROCm 10's
same-process `ncclCommInitAll` route used by llama.cpp either produced corrupt
tokens (`threaded`) or stalled (`native-group`) in runtime tests. It is disabled
by default here. For debugging only, pass `-EnableExperimentalLlamaRccl` to the
launcher; do not use that result as a coherency or release measurement.

### Qwen3.8 Flash-Next on 2 x R9700

The tested three-shard
`Qwen3.8-Flash-Next-ROCmFP4-STRIX_LEAN` GGUF is approximately 98.5 GiB. On a
128 GiB Windows host, tensor splitting with `--no-mmap` exhausted practical
RAM headroom while the checkpoint was loading. Tensor splitting with mmap
reached the first 20,480-byte D3D12 collective and then faulted in
`ggml-base.dll`. Do not present either configuration as validated.

The currently coherent full-model fallback is mmap-backed layer splitting:

```powershell
& G:\AI-plugins\wavmg-llama-rocm10-gfx1201\run-with-llama-plugin.ps1 `
    -Executable G:\AI-builds\rocmfpx-rocm10\bin\llama-cli.exe `
    -Mode hybrid `
    --model G:\AI-models\Qwen3.8-Flash-Next-ROCmFP4-STRIX_LEAN-GGUF\Qwen3.8-Flash-Next-ROCmFP4-STRIX_LEAN-00001-of-00003.gguf `
    --n-gpu-layers 40 `
    --split-mode layer `
    --tensor-split 1,1
```

With a deterministic four-token probe, that placement matched the one-GPU
continuation and measured 16.3 token/s generation versus 7.2 token/s on one
GPU. Layer splitting does not invoke llama's collective hook, so this result
validates the upstream host-staged multi-GPU fallback, not D3D12 or
same-process Direct-RCCL. Keep the small deterministic tensor-split parity
probe as the adapter's transport gate until the full-model mmap tensor fault is
fixed.

The D3D12 cross-adapter heap on these discrete adapters is a GPU-driven,
stream-ordered system-memory path. It is not proven direct VRAM-to-VRAM P2P.
The adapter publishes local data with a HIP kernel and performs the reduction
with a fused kernel that reads the fenced peer heap. It deliberately does not
call `hipMemcpyAsync` on the imported external pointer because that operation
faulted with ROCm 10 at a real 235,520-byte speculative payload.

## Updating ROCmFPX / llama.cpp

Update the clean engine checkout normally, then configure a new build
directory with the same `GGML_HIP_RCCL=ON`, `GGML_CUDA_NO_PEER_COPY=ON`, and
`rccl_DIR` arguments. No adapter patch needs to be rebased. Rebuild the adapter
only when its ABI, ROCm runtime, GPU architecture, or Windows RCCL build
changes. After either side changes, compare deterministic D3D12 and safe-hybrid
output against a one-GPU reference.
