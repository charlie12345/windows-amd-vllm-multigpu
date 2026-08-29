# Windows AMD multi-GPU bridge v0.2.0-rc1

Initial source prerelease for the dual-R9700 Windows test stack.

## Added

- External llama.cpp/ROCmFPX `roc::rccl` package; no llama source patch or
  overlay is required.
- Six-symbol llama RCCL shim that delegates to `rccl-real.dll` and can route
  supported two-rank AllReduce calls through the D3D12 cross-adapter path.
- Reproducible build, install, packaging, checksum, and provenance scripts.
- Separate vLLM and llama/ROCmFPX integration guides.
- Packaging guards that reject dirty or runtime-unvalidated binary inputs by
  default.

## Validation completed

- The shim compiles and links with portable ROCm 10.0.0 for `gfx1201`.
- Its export table contains only the six intended llama symbols.
- An unchanged ROCmFPX/llama source tree configured with the installed
  `rccl_DIR` and built `ggml-hip.dll` successfully.
- At the time of this historical RC1 upload, GitHub reported the repository as
  private. The repository became public in the later source-release phase.

## Not yet a stable binary release

The compile-only local package currently combines a ROCm 10.0-built shim with
an RCCL 2.30.7 DLL previously built using the pinned ROCm 7.15 toolchain. GPU
runtime compatibility has not yet been tested because another GPU workload was
active. Do not publish that archive as a stable ROCm 10 binary. Rebuild RCCL
against the chosen release runtime, then run the three transport modes and the
exact-value/model gates in `docs/publishing.md`.

The D3D12 path is GPU-driven and stream ordered, but its cross-adapter heap is
system-memory backed on the validated discrete adapters. This release does not
claim direct VRAM-to-VRAM P2P.
