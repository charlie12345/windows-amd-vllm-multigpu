#Requires -Version 5.1

[CmdletBinding()]
param(
    [ValidateRange(0, 1048576)]
    [double]$MinimumMiB = 1
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Samples = (Get-Counter '\GPU Process Memory(*)\Local Usage').CounterSamples |
    Where-Object { $_.CookedValue -ge ($MinimumMiB * 1MB) }

$Rows = foreach ($Sample in $Samples) {
    if ($Sample.InstanceName -notmatch 'pid_(\d+)_') {
        continue
    }

    $ProcessId = [int]$Matches[1]
    $Process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    [pscustomobject]@{
        Pid = $ProcessId
        Alive = $null -ne $Process
        ProcessName = if ($Process) { $Process.ProcessName } else { '<exited>' }
        LocalMiB = [math]::Round($Sample.CookedValue / 1MB, 1)
        Instance = $Sample.InstanceName
    }
}

if (-not $Rows) {
    Write-Output 'PASS: no Windows GPU process owns local memory above threshold.'
    exit 0
}

$Rows |
    Sort-Object -Property LocalMiB -Descending |
    Format-Table -AutoSize

$Exited = @($Rows | Where-Object { -not $_.Alive })
if ($Exited.Count -eq 0) {
    Write-Output 'PASS: every reported GPU-memory owner is a live process.'
} else {
    Write-Warning (
        '{0} GPU-memory counter instance(s) belong to exited processes.' -f
        $Exited.Count
    )
}
