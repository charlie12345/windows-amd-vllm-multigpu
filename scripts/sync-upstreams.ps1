#Requires -Version 5.1

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Pins = Get-Content -Raw -LiteralPath (
    Join-Path $ProjectRoot 'pins\rocm10-vllm-v0.28.0.json'
) | ConvertFrom-Json

function Sync-PinnedRepository {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Commit,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    if (-not (Test-Path -LiteralPath (Join-Path $Destination '.git'))) {
        New-Item -ItemType Directory -Force -Path (Split-Path $Destination) | Out-Null
        & git clone --filter=blob:none --no-checkout $Url $Destination
        if ($LASTEXITCODE -ne 0) {
            throw "Cloning $Url failed."
        }

        # A --no-checkout clone reports every tracked path as deleted until its
        # first checkout. Complete that exact pinned checkout before applying
        # the existing-worktree dirty guard below.
        & git -C $Destination fetch --depth 1 origin $Commit
        if ($LASTEXITCODE -ne 0) {
            throw "Fetching pinned commit $Commit from $Url failed."
        }
        & git -C $Destination checkout --detach $Commit
        if ($LASTEXITCODE -ne 0) {
            throw "Checking out pinned commit $Commit failed."
        }
        return
    }

    $Current = (& git -C $Destination rev-parse HEAD 2>$null).Trim()
    $Dirty = @(& git -C $Destination status --porcelain)
    if ($Dirty.Count -gt 0) {
        if ($Current -eq $Commit) {
            Write-Host "Leaving patched pinned checkout in place: $Destination"
            return
        }
        throw "Refusing to change a dirty checkout at $Destination."
    }

    & git -C $Destination fetch --depth 1 origin $Commit
    if ($LASTEXITCODE -ne 0) {
        throw "Fetching pinned commit $Commit from $Url failed."
    }
    & git -C $Destination checkout --detach $Commit
    if ($LASTEXITCODE -ne 0) {
        throw "Checking out pinned commit $Commit failed."
    }
}

Sync-PinnedRepository `
    -Url $Pins.rocm_systems_repo `
    -Commit $Pins.rocm_systems_commit `
    -Destination (Join-Path $ProjectRoot 'sandbox\rocm-systems')
Sync-PinnedRepository `
    -Url $Pins.hipify_repo `
    -Commit $Pins.hipify_commit `
    -Destination (Join-Path $ProjectRoot 'sandbox\hipify')

Write-Host 'Pinned RCCL and HIPIFY sources are ready.' -ForegroundColor Green
