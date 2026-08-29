#Requires -Version 5.1

[CmdletBinding(SupportsShouldProcess)]
param(
    [ValidatePattern('^[D-Zd-z]$')]
    [string]$PagefileDrive = 'G',
    [ValidateRange(32768, 262144)]
    [int]$InitialSizeMiB = 98304,
    [ValidateRange(32768, 262144)]
    [int]$MaximumSizeMiB = 131072,
    [switch]$RestoreAutomatic
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$principal = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this script from an elevated PowerShell window.'
}
if ($MaximumSizeMiB -lt $InitialSizeMiB) {
    throw 'MaximumSizeMiB must be greater than or equal to InitialSizeMiB.'
}

$drive = $PagefileDrive.ToUpperInvariant()
$memoryKey = 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management'
$computerSystem = Get-CimInstance Win32_ComputerSystem

if ($RestoreAutomatic) {
    if ($PSCmdlet.ShouldProcess('Windows pagefile configuration', 'restore automatic management')) {
        Set-CimInstance -InputObject $computerSystem -Property @{
            AutomaticManagedPagefile = $true
        } | Out-Null
        Set-ItemProperty -LiteralPath $memoryKey -Name PagingFiles `
            -Type MultiString -Value @('?:\pagefile.sys')
    }
    Write-Host 'Automatic pagefile management restored. Reboot Windows to apply it.'
    return
}

$volume = Get-Volume -DriveLetter $drive
if ($volume.FileSystemType -ne 'NTFS' -or $volume.HealthStatus -ne 'Healthy') {
    throw "$drive`: must be a healthy NTFS volume for a Windows pagefile."
}
$requiredFreeBytes = ($MaximumSizeMiB + 65536L) * 1MB
if ($volume.SizeRemaining -lt $requiredFreeBytes) {
    throw "$drive`: needs the maximum pagefile size plus 64 GiB free."
}

$buildDirectory = Join-Path $PSScriptRoot '..\build'
New-Item -ItemType Directory -Path $buildDirectory -Force | Out-Null
$backupPath = Join-Path $buildDirectory 'pagefile-settings-before.json'
$pagingFiles = @(
    (Get-ItemProperty -LiteralPath $memoryKey -Name PagingFiles).PagingFiles
)
[ordered]@{
    captured_at = Get-Date -Format o
    automatic_managed_pagefile = [bool]$computerSystem.AutomaticManagedPagefile
    paging_files = $pagingFiles
} | ConvertTo-Json | Set-Content -LiteralPath $backupPath -Encoding UTF8

$newPagingFiles = @(
    'C:\pagefile.sys 0 0'
    "$drive`:\pagefile.sys $InitialSizeMiB $MaximumSizeMiB"
)
if ($PSCmdlet.ShouldProcess(
        ($newPagingFiles -join ', '),
        'configure C: system-managed plus the requested secondary pagefile'
    )) {
    Set-CimInstance -InputObject $computerSystem -Property @{
        AutomaticManagedPagefile = $false
    } | Out-Null
    Set-ItemProperty -LiteralPath $memoryKey -Name PagingFiles `
        -Type MultiString -Value $newPagingFiles
}

Write-Host "Saved the previous configuration to $backupPath"
Write-Host 'Configured pagefiles:'
$newPagingFiles | ForEach-Object { Write-Host "  $_" }
Write-Host 'Reboot Windows before retrying the large model.'
Write-Host 'Undo with: .\scripts\configure-windows-pagefile.ps1 -RestoreAutomatic'
