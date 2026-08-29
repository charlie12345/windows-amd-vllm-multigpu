# Public source-publishing checklist

This is a public, source-only repository. Users compile the Windows AMD vLLM
host, Windows RCCL, native D3D12/HIP transport, and llama adapter themselves.
Do not publish Docker images, wheels containing native DLLs, ZIP binaries, or
GitHub Release assets until the separate binary-release gate is complete.

Changing a repository from private to public exposes its reachable history,
branches, and tags, not only the current default branch. The full-history audit
below is therefore a release blocker. If it finds a real credential, stop,
revoke and rotate that credential first, then clean the affected history before
publication. Merely deleting a key from the latest commit is insufficient.

## Two-repository boundary

The inference engine and multi-GPU transport are intentionally separate:

- [`charlie12345/vLLM_for_AMD`](https://github.com/charlie12345/vLLM_for_AMD)
  is the public native-Windows AMD vLLM host; and
- [`charlie12345/windows-amd-vllm-multigpu`](https://github.com/charlie12345/windows-amd-vllm-multigpu)
  is this public external transport plugin.

This repository has two engine-specific adapters:

- `src/windows_amd_vllm_multigpu`: multi-process vLLM platform plugin;
- `adapters/llama-rccl-shim`: single-process llama/ROCmFPX RCCL ABI shim.

Do not market the llama DLL as a universal RCCL replacement. Do not put the
shim in the vLLM RCCL path. Do not copy this repository's Python source into
the vLLM checkout. Install the Python package into the same virtual environment
as the exact pinned vLLM host instead.

## Current publication scope

Publish source only. Do not commit DLLs, model weights, ROCm wheels, AMD
drivers, Visual Studio files, build output, downloaded upstream source trees,
credentials, or logs to Git. Keep exact source and dependency revisions in
`pins/`. See [`licensing.md`](licensing.md); it is an engineering checklist,
not legal advice.

## Source PR gate

1. Verify `gh api user --jq .login` reports `charlie12345`.
2. Fetch all branches and tags, then run the history audit:

   ```powershell
   git fetch --all --prune --tags
   .\scripts\audit-public-source.ps1
   ```

   The scanner checks every blob reachable through local refs and prints only
   finding type, object ID, and path—not the matched value. A candidate is a
   release blocker until manually resolved. Keep GitHub secret scanning enabled
   as an independent server-side check.
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
9. Verify the repository is public and `charlie12345` is still active.

The current second R9700 health failure prevents a new ROCm 10 TP=2 runtime
revalidation. It does not prevent a clearly labeled source-only PR whose
static, host-integration, and compile checks pass. It does prevent a stable
binary or runtime-performance claim for this exact post-update stack.

## Maintainer commands

After reviewing and committing the source branch:

```powershell
gh auth switch --user charlie12345
gh api user --jq .login

git fetch --all --prune --tags
.\scripts\audit-public-source.ps1

git push origin <feature-branch>

gh pr create `
    --repo charlie12345/windows-amd-vllm-multigpu `
    --base main `
    --head <feature-branch> `
    --title '<concise source change>' `
    --body-file <reviewed-release-notes.md>

gh pr checks --repo charlie12345/windows-amd-vllm-multigpu <PR-NUMBER>
# Review the Files changed tab and merge only after every source gate passes.
gh pr merge --repo charlie12345/windows-amd-vllm-multigpu `
    <PR-NUMBER> --merge --delete-branch=false

gh repo view charlie12345/windows-amd-vllm-multigpu `
    --json nameWithOwner,isPrivate,visibility,url
# Confirm isPrivate is false and visibility is PUBLIC.
```

Never paste tokens into scripts, commands, issue text, or release notes. Use
the GitHub CLI credential store and verify the active account before every push.

## Future binary releases

Binary distribution is out of scope for this phase. Before enabling it,
restore a separate binary-release gate that requires fresh-clone builds,
exact-value RCCL and D3D12 tests, deterministic vLLM and llama TP=2 parity,
repeated clean shutdown, loader-path verification, per-file hashes, full
license bundles, and an exact build manifest. Do not infer binary
redistribution permission merely from the source licenses.
