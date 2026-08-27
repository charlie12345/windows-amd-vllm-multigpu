# Private publishing and release checklist

The GitHub repository is private and must remain private. Releases and their
assets are uploaded only to that private repository and are available only to
authorized GitHub users. Do not run a visibility-change command as part of the
build or release workflow.

## Repository structure

Publish one source repository with two explicit adapters:

- `src/windows_amd_vllm_multigpu`: multi-process vLLM platform plugin;
- `adapters/llama-rccl-shim`: single-process llama/ROCmFPX RCCL ABI shim.

Do not market the llama DLL as a universal RCCL replacement. Do not put the
shim in the vLLM RCCL path.

## Recommended release assets

Use separate, versioned assets because their ABI and runtime matrices differ:

- `wavmg-vllm-<version>-rocm<version>-gfx1201.zip`;
- `wavmg-llama-shim-<version>-rocm<version>-gfx1201.zip`;
- `SHA256SUMS.txt` and a provenance JSON for each binary archive.

GitHub automatically supplies source archives. Do not commit DLLs, model
weights, ROCm wheels, AMD drivers, Visual Studio files, or downloaded upstream
source trees to Git.

Each binary ZIP must include the top-level `LICENSE`, `NOTICE`, the complete
`LICENSES` directory, and a manifest recording:

- this repository's Git commit;
- ROCm distribution/version and GPU target;
- RCCL repository commit, RCCL version, and patch SHA-256;
- every shipped DLL's SHA-256;
- the tested GPU, driver, Windows build, and test result.

The RCCL binary remains under upstream RCCL/NCCL terms; the surrounding shim,
D3D12 adapter, Python integration, and scripts are Apache-2.0. See
[`licensing.md`](licensing.md). This is an engineering checklist, not legal
advice.

## Gate before a private binary release

1. Build from a fresh clone at the intended tag.
2. Run the RCCL ABI probe and exact-value native validation.
3. Run D3D12 exact-value tests for FP16, FP32, and BF16.
4. Run vLLM TP2 RCCL-only, then hybrid, and compare deterministic token IDs.
5. Run llama TP2 in `rccl`, `d3d12`, and `hybrid` modes and compare output to
   a one-GPU reference.
6. Stress repeated startup/shutdown and at least one long generation.
7. Confirm no DLL is loaded from an unintended directory with Process Explorer
   or loader diagnostics.
8. Generate the release archives and independently verify their hashes.

If GPU runtime validation has not happened, publish source or a GitHub
prerelease only. Label compile/link-only artifacts accordingly. The packaging
script refuses dirty and runtime-unvalidated provenance by default;
`-AllowUnvalidated` is only for an explicitly labeled prerelease.

## Maintainer commands

After reviewing and committing the release branch:

```powershell
gh auth switch --user charlie12345
gh repo view charlie12345/windows-amd-vllm-multigpu `
    --json nameWithOwner,isPrivate,url
# Stop unless isPrivate is true.

git push origin main

git tag -a v0.2.0-rc1 -m 'Windows AMD multi-GPU bridge v0.2.0-rc1'
git push origin v0.2.0-rc1
gh release create v0.2.0-rc1 .\dist\*.zip .\dist\SHA256SUMS.txt `
    --prerelease `
    --verify-tag `
    --title 'Windows AMD multi-GPU bridge v0.2.0-rc1' `
    --notes-file .\docs\release-notes-v0.2.0-rc1.md

gh repo view charlie12345/windows-amd-vllm-multigpu `
    --json nameWithOwner,isPrivate,url
# Confirm isPrivate is still true.
```

Never paste tokens into scripts or release notes. Use the GitHub CLI credential
store and verify the active account with `gh auth status` first.
