$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Pins = Get-Content (Join-Path $ProjectRoot "pins\nightly-2026-07-28.json") |
    ConvertFrom-Json
$VllmRoot = Join-Path $ProjectRoot "sandbox\vllm"
$Patches = @(
    "patches\vllm\fb9fb8c5-windows-pipeconnection.patch",
    "patches\vllm\fb9fb8c5-dflash2-upstream.patch",
    "patches\vllm\fb9fb8c5-windows-cooperative-engine-shutdown.patch"
)

$ActualCommit = (git -C $VllmRoot rev-parse HEAD).Trim()
if ($ActualCommit -ne $Pins.vllm_commit) {
    throw "Refusing to patch vLLM $ActualCommit; expected $($Pins.vllm_commit)"
}

foreach ($RelativePatch in $Patches) {
    $Patch = Join-Path $ProjectRoot $RelativePatch
    git -C $VllmRoot apply --check $Patch 2>$null
    if ($LASTEXITCODE -eq 0) {
        git -C $VllmRoot apply $Patch
        if ($LASTEXITCODE -ne 0) {
            throw "vLLM patch application failed: $RelativePatch"
        }
        Write-Host "Applied pinned vLLM patch: $RelativePatch"
        continue
    }

    git -C $VllmRoot apply --reverse --check $Patch 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Pinned vLLM patch is already applied: $RelativePatch"
        continue
    }

    throw "Pinned vLLM patch neither applies nor appears applied: $RelativePatch"
}
