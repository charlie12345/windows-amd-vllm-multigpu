# Windows AMD multi-GPU bridge v0.2.0-rc2

Source update for the dual-R9700 Windows ROCm 10 test stack. This PR
publishes no Docker image, prebuilt DLL, wheel containing native code, model
weight, tag, or GitHub Release asset; users compile the components locally.

## Fixed

- Replaced `hipMemcpyAsync` publishing to imported D3D12 memory with a custom
  HIP byte-copy kernel. ROCm 10 could reject or fault while classifying that
  external pointer.
- Removed the second external-pointer copy and now reduce directly from the
  fenced peer D3D12 heap into local VRAM.
- Added an explicit current-device guard for vLLM worker/Inductor threads.
- Added the exact 235,520-byte Qwen speculative collective to the D3D12
  FP16/FP32/BF16 regression matrix.
- Run Direct-RCCL validation on a real non-default PyTorch HIP stream.
- Allow native and RCCL builds to use an explicit portable `ROCM_ROOT`,
  including ROCm 10, without accidentally loading unrelated DLLs from PATH.
- Added a fail-closed Windows adapter-health check before multi-GPU probes and
  launchers. A missing or unhealthy adapter stops before a HIP context is
  created instead of risking another driver hang.
- Document and validate upstream `GGML_CUDA_NO_PEER_COPY=ON` for non-P2P
  Windows HIP adapters. This removed silent TP2 corruption without patching
  ROCmFPX/llama.cpp.
- Correct the external llama launch mode to `--split-mode tensor --no-mmap`.
- Default the llama shim to D3D12 for supported payloads through 64 MiB and
  quarantine same-process Direct-RCCL behind
  `WAC_LLAMA_RCCL_EXPERIMENTAL=1` after coherency failures.

## Integration

- vLLM remains an external Python platform and communicator plugin. The exact
  host is `charlie12345/vLLM_for_AMD` commit
  `264ac2cf2b44bb71af4618630f9cccef6f616f49`, based on upstream vLLM v0.28.0.
  The bootstrap verifies a clean tracked host at that commit and applies zero
  vLLM patches.
- llama.cpp/ROCmFPX remains an external six-symbol `roc::rccl` package built
  through upstream's existing `GGML_HIP_RCCL` hook; no engine source overlay is
  required.
- Updated the pinned build stack to PyTorch `2.13.0+rocm10.0.0`, torchvision
  `0.28.0+rocm10.0.0`, Triton Windows `3.7.1.post27`, and ROCm `10.0.0`.
- Build scripts expose RDNA 2, RDNA 3, RDNA 3.5, and RDNA 4 targets. Only the
  dual-R9700 `gfx1201` configuration has prior end-to-end runtime validation;
  the RX 7900 XTX is build-selectable as `gfx1100`, not runtime-qualified.
- Added a fail-closed `WAVMG_ALLOW_TP_EXPERT_PARALLEL=1` gate for TP=2 expert
  ownership only when vLLM does not require All-to-All. True All-to-All remains
  rejected. The machine-specific Flash-Next launch script and vLLM model-code
  experiments remain outside this release candidate.

## Known limitations

- The 98.5 GiB Qwen3.8 Flash-Next ROCmFP4 GGUF is coherent on the tested
  2 x R9700 machine with mmap-backed layer splitting. It measured 16.3 token/s
  generation versus 7.2 token/s on one GPU for the deterministic smoke prompt.
- That layer split uses upstream host staging and does not exercise the
  collective adapter. Full-model tensor splitting is not yet qualified:
  `--no-mmap` leaves insufficient host-RAM headroom on the 128 GiB test system,
  while mmap tensor splitting faulted after its first D3D12 collective.
- The vLLM W4A16/FP8-PLE checkpoint reached TP=2/EP=2 model construction and
  shard 17/25 with the plugin. Runtime instrumentation then found only
  0.95 GiB of Windows commit headroom before shard 1 on a separate guarded
  pass, because WDDM VRAM reservations and the 47.745 GiB CPU PLE pool share
  the system commit limit. A larger G-drive pagefile plus reboot is required
  before generation testing. No Flash-Next vLLM throughput or coherency claim
  is made yet.

## Local source validation

- Public Windows AMD vLLM PRs 24 through 26 were merged; the final host pin is
  `264ac2cf2b44bb71af4618630f9cccef6f616f49`.
- The exact-host verifier accepted a clean detached checkout at that pin and
  rejected a different commit. The retired patch command performed verification
  only and applied zero vLLM patches.
- The RC2 plugin installed into the ROCm 10 vLLM environment, registered its
  platform entry point, selected `WindowsAmdMultiGpuPlatform`, and passed its
  parallel-configuration gate against the exact v0.28 host source.
- The native D3D12/HIP transport compiled for `gfx1201`; Windows RCCL configured
  from the exact pinned sources as ROCm 10.0.0 and completed all 646 build
  targets; the external llama shim then compiled and linked against that DLL.
- Changed Python passed Ruff's fatal-error/undefined-name rules, Ruff's formatter
  check, and bytecode compilation. All 33 PowerShell scripts parsed
  successfully. Both JSON pins, local Markdown links, license copies, credential
  scan, `git diff --check`, and the source-distribution content audit passed.
- The CPU/single-process Gloo control-plane probe passed. No post-update TP=2
  GPU test was attempted after the health gate confirmed the second adapter is
  unavailable.

## Release gate

The source PR may be merged after the static, exact-host, package-content, and
compile/link checks in `docs/publishing.md` pass. After the power interruption,
Windows currently reports the second R9700 as unhealthy, so no new TP=2 runtime
claim is made for the final vLLM v0.28/ROCm 10 commit. The fail-closed health
gate is expected to reject that state. Do not publish stable binary assets or a
new performance claim until Direct-RCCL and hybrid vLLM TP=2 plus D3D12 and
safe-hybrid llama/ROCmFPX TP=2 are revalidated with both adapters healthy.
Same-process llama Direct-RCCL remains explicitly experimental.
