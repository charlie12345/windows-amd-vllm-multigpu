# Install the bridge in ROCmFPX / llama.cpp on Windows

The llama adapter is an external `roc::rccl` CMake package. A clean upstream
llama.cpp or ROCmFPX checkout consumes it through `find_package(rccl)`; no
llama source patch or overlay is required.

The current implementation is for llama's single-process, two-rank
`ncclCommInitAll` path. It must not be used as the vLLM/PyTorch RCCL DLL.

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
    -DGPU_TARGETS=gfx1201 `
    "-DCMAKE_PREFIX_PATH=$rocm;$plugin" `
    "-Drccl_DIR=$plugin\lib\cmake\rccl"
cmake --build $build --config Release --parallel 16
```

Inspect `CMakeCache.txt` and the `ggml-hip` link command to verify that
`rccl_DIR` resolves to the plugin, not to a stale ROCm or build-tree package.

## Launch

Use the installed launcher so the shim precedes the ROCm directory on `PATH`:

```powershell
G:\AI-plugins\wavmg-llama-rocm10-gfx1201\run-with-llama-plugin.ps1 `
    -Executable G:\AI-builds\rocmfpx-rocm10-plugin\bin\llama-server.exe `
    -RocmRoot G:\ROCm\10.0.0-gfx120x `
    -Mode hybrid `
    --model G:\models\model.gguf `
    --split-mode row `
    --tensor-split 1,1
```

The launcher sets `GGML_CUDA_ALLREDUCE=nccl`; llama uses that historical
environment-variable name for its HIP NCCL/RCCL selection too.

Startup should print `wavmg-llama-rccl-shim: initialized ranks=2`. The first
operation on each route prints either `route=D3D12` or `route=Direct-RCCL`.
Test `-Mode rccl`, `-Mode d3d12`, and `-Mode hybrid` independently before
benchmarking.

The D3D12 cross-adapter heap on these discrete adapters is a GPU-driven,
stream-ordered system-memory path. It is not proven direct VRAM-to-VRAM P2P.
