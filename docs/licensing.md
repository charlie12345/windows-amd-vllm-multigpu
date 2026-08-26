# Licensing and attribution

This document records the provenance of source, patches, generated binaries,
and optional model weights used by this project. It is an engineering
distribution checklist, not legal advice.

## Repository license map

| Material | Origin and pinned source | Applicable terms | Files shipped here |
| --- | --- | --- | --- |
| Original Python, C++, HIP, PowerShell, CMake, probes, and documentation | Carlo Pasquale (`@charlie12345`) and Windows AMD vLLM Multi-GPU contributors | Apache License 2.0 | `LICENSE`, `NOTICE` |
| vLLM compatibility patch | `charlie12345/vLLM_for_AMD` at `fb9fb8c5aeaed96c91eef5cb48743a96f8496907`, derived from vLLM | Apache License 2.0 and the pinned fork's NOTICE | `patches/vllm/`, `LICENSES/VLLM-UPSTREAM-LICENSE.txt`, `LICENSES/VLLM-UPSTREAM-NOTICE.txt` |
| Native Windows RCCL patch and a resulting `rccl.dll` | ROCm `rocm-systems/projects/rccl` at `ee3bae9a931561506c49dcf82fca52ec4711c34f`, derived in part from NVIDIA NCCL | Upstream BSD-3-Clause terms plus retained third-party terms | `patches/rccl/`, `LICENSES/RCCL-UPSTREAM-LICENSE.txt`, `LICENSES/RCCL-UPSTREAM-NOTICES.txt`, `LICENSES/RCCL-UPSTREAM-ThirdPartyNotices.txt` |
| RCCL files added by this Windows port | Windows AMD vLLM Multi-GPU contributors, added inside the RCCL-derived tree | BSD-3-Clause, matching the surrounding RCCL project; each added source carries an SPDX identifier | Included inside the version-pinned RCCL patch |
| HIPIFY | ROCm HIPIFY at `f1af27c6e0c43e1a9663dc3650dcff54f980e6a6` | Its upstream license | Fetched into ignored `sandbox/hipify`; no HIPIFY source is redistributed |
| ROCm, HIP, PyTorch, Triton, Vulkan SDK, Windows SDK, and build tools | Their respective publishers | Their respective upstream terms | Installed/fetched dependencies; not committed or relicensed here |
| Current Mistral Small 24B AWQ test weights | `stelterlab/Mistral-Small-24B-Instruct-2501-AWQ` at `cbda099649a0188dd888d44f0e4964d8d982dc9a`, derived from Mistral Small 24B | Quantizer-publisher-declared Apache-2.0; retain base-model attribution | Downloaded to the user's Hugging Face cache; never committed or bundled |
| Historical Mistral Small 24B BF16 baseline | `mistralai/Mistral-Small-24B-Instruct-2501` at `9527884be6e5616bdd54de542f9ae13384489724` | Model-publisher-declared Apache-2.0 | Results and pin retained; weights are not bundled |
| Qwen3.8-27B BF16 format trial | `Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | Model-publisher-declared Apache-2.0 | Results and pin retained; weights were deleted and are not bundled |
| Qwen3.8-27B native FP8 format trial | `Qwen/Qwen3.8-27B-FP8` at `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` | Model-publisher-declared Apache-2.0 | Results and pin retained; weights were deleted and are not bundled |
| Qwen3.8-27B native NVFP4 compatibility trial | `unsloth/Qwen3.8-27B-NVFP4` at `9e3d73c76eddb75f795cc24ccfbc5affe41c66bd`, derived from `Qwen/Qwen3.8-27B` | Quantizer-publisher-declared Apache-2.0; retain base-model attribution | Results and pin retained; weights were deleted and are not bundled |
| Qwen3.8-27B standard W4A16 INT4 trial | `abihsoro/Qwen3.8-27B-AWQ-INT4` at `f2e0cac39907e7b1ed7fdb210363dd33cc18f993`, derived from `Qwen/Qwen3.8-27B` | Quantizer-publisher-declared Apache-2.0; retain base-model attribution | Downloaded to `G:` for local testing; never committed or bundled |
| Qwen3.8-27B DFlash2 drafter trial | `z-lab/Qwen3.8-27B-DFlash2` at `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`, for the Qwen3.8 target | Model-publisher-declared Apache-2.0; retain target-model attribution | Downloaded to `G:` for local testing; never committed or bundled |

The two shorter `LICENSES/RCCL-LICENSE.txt` and
`LICENSES/RCCL-NOTICES.txt` files are convenient distribution copies and a
summary of material terms. The `RCCL-UPSTREAM-*` files are verbatim texts from
the pinned upstream checkout, subject only to line-ending normalization by Git.
When the summary and upstream text differ, the upstream text controls.

## Patch attribution and modification notice

The RCCL and vLLM patch filenames contain abbreviated base commits, and the
complete commits are recorded in `pins/nightly-2026-07-28.json`. The patches
are clearly identified as Windows modifications rather than unmodified
upstream releases. New RCCL-tree files carry `SPDX-License-Identifier:
BSD-3-Clause` and a Windows AMD vLLM Multi-GPU contributor notice. Existing
upstream copyright and SPDX headers are retained in patch context and source.

The vLLM patch changes only the files named by its diff. The repository NOTICE,
the adjacent version-locked patch, and the pinned fork NOTICE identify the
modified distribution and its upstream source.

## Source-distribution checklist

A source archive must contain:

1. the top-level `LICENSE` and `NOTICE`;
2. the complete `LICENSES` directory;
3. the exact `pins/nightly-2026-07-28.json` file; and
4. both version-pinned patch directories.

`MANIFEST.in` includes all four categories. The package build is checked so
Python bytecode, upstream sandboxes, model weights, compiled objects, and local
logs are not included.

## Binary-distribution checklist

If `rccl.dll`, `wavmg_d3d12_v1.dll`, `wavmg_hip_v1.dll`, or another compiled
artifact is redistributed, place the following beside the binary archive or
inside an obvious `licenses` directory:

1. `LICENSE`;
2. `NOTICE`;
3. every file under `LICENSES`; and
4. a build manifest identifying the exact Git commit and dependency pins.

Do not bundle AMD drivers, ROCm wheels, Visual Studio components, the Vulkan
SDK, vLLM source, or model weights unless their separate redistribution terms
have also been reviewed and satisfied.
