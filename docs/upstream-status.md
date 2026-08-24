# Upstream status and sources

- AMD's TheRock Windows support matrix marks RCCL unsupported:
  <https://github.com/ROCm/TheRock/blob/main/docs/development/windows_support.md>
- AMD enabled Gloo in its Windows PyTorch wheel pipeline on 2026-06-16:
  <https://github.com/ROCm/TheRock/pull/5694>
- The current TheRock PyTorch build script sets `USE_GLOO=ON`:
  <https://github.com/ROCm/TheRock/blob/main/external-builds/pytorch/build_prod_wheels.py>
- PyTorch tracks the fatal Windows direct-Gloo/GPU-tensor registry failure:
  <https://github.com/pytorch/pytorch/issues/186535>
- PyTorch's root build supports `USE_DISTRIBUTED` on Windows and exposes
  `USE_C10D_GLOO` when distributed and Gloo are enabled:
  <https://github.com/pytorch/pytorch/blob/main/CMakeLists.txt>

The current direction is therefore Gloo as a CPU control plane plus an explicit
Windows AMD data plane. A native RCCL port remains a much larger upstream-scale
alternative.

