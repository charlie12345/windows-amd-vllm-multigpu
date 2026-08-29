<#
.SYNOPSIS
    Fail closed unless the intended AMD display adapters are healthy.

.DESCRIPTION
    Performs a read-only Windows PnP check before any multi-GPU process or HIP
    context is started. It prevents a known hang-prone case where Device
    Manager still lists both adapters but one reports CM_PROB_FAILED_ADD.
#>
#Requires -Version 5.1

[CmdletBinding()]
param(
    [ValidateRange(1, 16)]
    [int]$RequiredCount = 2
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($env:OS -ne 'Windows_NT') {
    throw 'The Windows AMD GPU health gate supports native Windows only.'
}

try {
    $Adapters = @(Get-PnpDevice -Class Display -PresentOnly -ErrorAction Stop |
        Where-Object { $_.FriendlyName -match 'AMD|Radeon' } |
        ForEach-Object {
            [pscustomobject]@{
                Name = $_.FriendlyName
                Status = [string]$_.Status
                Problem = [string]$_.Problem
            }
        })
}
catch {
    throw "Could not query AMD display-adapter health: $($_.Exception.Message)"
}

if ($Adapters.Count -lt $RequiredCount) {
    throw "Detected $($Adapters.Count) AMD adapter(s); $RequiredCount required."
}

$Unhealthy = @($Adapters | Where-Object { $_.Status -ne 'OK' })
if ($Unhealthy.Count -gt 0) {
    $Details = ($Unhealthy | ForEach-Object {
        '{0}: status={1}, problem={2}' -f $_.Name, $_.Status, $_.Problem
    }) -join '; '
    throw @"
Windows reports an unhealthy AMD adapter: $Details
Do not start TP=2, Direct-RCCL, D3D12, or hybrid mode. Restart Windows and
verify both adapters show Device Manager status OK. If the error remains,
cold-power-cycle and repair/reinstall the matching AMD driver.
"@
}

Write-Host "AMD GPU health gate passed ($($Adapters.Count) healthy adapter(s))."
