# Private source-publishing checklist

The multi-GPU plugin repository is private and must remain private. Source is
pushed only to that repository and is available only to authorized GitHub
users. Do not run a visibility-change command as part of the build or publishing
workflow.

## Two-repository boundary

The inference engine and multi-GPU transport are intentionally separate:

- `charlie12345/vLLM_for_AMD` is the public Windows AMD vLLM host; and
- `charlie12345/windows-amd-vllm-multigpu` is this private plugin repository.

This private repository has two engine-specific adapters:

- `src/windows_amd_vllm_multigpu`: multi-process vLLM platform plugin;
- `adapters/llama-rccl-shim`: single-process llama/ROCmFPX RCCL ABI shim.

Do not market the llama DLL as a universal RCCL replacement. Do not put the
shim in the vLLM RCCL path. Do not copy this repository's Python source into
the vLLM checkout. Install the Python package into the same virtual environment
as the exact pinned vLLM host instead.

## Current publication scope

Publish source only. Users compile the Windows AMD vLLM host, Windows RCCL,
native D3D12/HIP transport, and llama adapter themselves. This phase does not
publish Docker images, wheels containing native DLLs, ZIP binaries, or GitHub
Release assets.

Do not commit DLLs, model weights, ROCm wheels, AMD drivers, Visual Studio
files, build output, or downloaded upstream source trees to Git. Keep exact
source and dependency revisions in `pins/`. See
[`licensing.md`](licensing.md); it is an engineering checklist, not legal
advice.

## Source PR gate

1. Verify `gh auth status` reports `charlie12345` as the active account.
2. Verify the destination repository reports `"isPrivate": true` before push.
3. Review the complete diff and run `git diff --check`.
4. Validate both pin manifests and parse every changed PowerShell script.
5. Verify the exact clean vLLM host with `scripts/verify-vllm-host.ps1` and
   confirm the plugin applies zero vLLM patches.
6. Run Ruff's fatal-error/undefined-name rules and formatter check on changed
   Python, bytecode compilation, package-content checks, and available
   compile/link tests. Ensure the source artifact contains no DLLs, models,
   credentials, logs, or build output.
7. Run GPU probes only when Windows reports every required adapter healthy.
   Record any missing runtime validation in the PR; never bypass the health
   gate to turn an unhealthy adapter into a benchmark claim.
8. Push a feature branch, open a PR against `main`, inspect its checks and
   complete diff, and merge only after the source gate passes.
9. Verify the repository is still private and `charlie12345` is still active.

The current second R9700 health failure prevents a new ROCm 10 TP=2 runtime
revalidation. It does not prevent a clearly labeled source-only PR whose static,
host-integration, and compile checks pass. It does prevent a stable binary or
runtime-performance claim for this exact post-update stack.

## Maintainer commands

After reviewing and committing the source branch:

```powershell
gh auth switch --user charlie12345
gh repo view charlie12345/windows-amd-vllm-multigpu `
    --json nameWithOwner,isPrivate,url
# Stop unless isPrivate is true.

git push private-github feature/private-v0.2.0-rc2

gh pr create `
    --repo charlie12345/windows-amd-vllm-multigpu `
    --base main `
    --head feature/private-v0.2.0-rc2 `
    --title 'ROCm 10 and clean vLLM v0.28 plugin integration' `
    --body-file .\docs\release-notes-v0.2.0-rc2.md

gh pr checks --repo charlie12345/windows-amd-vllm-multigpu <PR-NUMBER>
# Review the Files changed tab and merge only after every source gate passes.
gh pr merge --repo charlie12345/windows-amd-vllm-multigpu `
    <PR-NUMBER> --merge --delete-branch=false

gh repo view charlie12345/windows-amd-vllm-multigpu `
    --json nameWithOwner,isPrivate,url
# Confirm isPrivate is still true.
```

Never paste tokens into scripts or release notes. Use the GitHub CLI credential
store and verify the active account with `gh auth status` first.

## Future binary releases

Binary distribution is out of scope for this phase. Before enabling it, restore
a separate binary-release gate that requires fresh-clone builds, exact-value
RCCL and D3D12 tests, deterministic vLLM and llama TP=2 parity, repeated clean
shutdown, loader-path verification, per-file hashes, full license bundles, and
an exact build manifest. Do not infer binary redistribution permission merely
from the source licenses.
