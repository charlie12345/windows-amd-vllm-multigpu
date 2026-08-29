[CmdletBinding()]
param(
    [ValidateRange(1, 1024)]
    [int]$MaximumBlobMiB = 10
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location $repoRoot
try {
    $insideWorkTree = (& git rev-parse --is-inside-work-tree 2>$null).Trim()
    if ($LASTEXITCODE -ne 0 -or $insideWorkTree -ne 'true') {
        throw "Not inside a Git worktree: $repoRoot"
    }

    $objectRows = @(& git rev-list --objects --all)
    if ($LASTEXITCODE -ne 0) {
        throw 'git rev-list failed.'
    }

    $pathByOid = @{}
    foreach ($row in $objectRows) {
        $parts = $row -split ' ', 2
        if ($parts.Count -eq 2) {
            $pathByOid[$parts[0]] = $parts[1]
        }
    }

    $objectIds = @(
        $objectRows |
            ForEach-Object { ($_ -split ' ', 2)[0] } |
            Sort-Object -Unique
    )
    $metadata = @(
        $objectIds |
            & git cat-file --batch-check='%(objectname) %(objecttype) %(objectsize)'
    )
    if ($LASTEXITCODE -ne 0) {
        throw 'git cat-file metadata scan failed.'
    }

    $patterns = [ordered]@{
        'GitHub token' = '(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})'
        'OpenAI-style key' = '(?<![A-Za-z0-9])sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}'
        'Hugging Face token' = '(?<![A-Za-z0-9])hf_[A-Za-z0-9]{20,}'
        'AWS access key' = '(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])'
        'Google API key' = '(?<![A-Za-z0-9_-])AIza[0-9A-Za-z_-]{30,}'
        'GitLab token' = '(?<![A-Za-z0-9_-])glpat-[A-Za-z0-9_-]{20,}'
        'Slack token' = '(?<![A-Za-z0-9-])xox[baprs]-[0-9A-Za-z-]{10,}'
        'Stripe live key' = '(?<![A-Za-z0-9_])(?:sk|rk)_live_[0-9A-Za-z]{16,}'
        'npm token' = '(?<![A-Za-z0-9_])npm_[A-Za-z0-9]{30,}'
        'PyPI token' = '(?<![A-Za-z0-9_-])pypi-AgEIcH[A-Za-z0-9_-]{30,}'
        'Private key block' = '-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----'
        'Credential-bearing URL' = 'https?://[^/\s:@]+:[^/\s@]+@'
        'JWT-like secret' = '(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}'
        'Hardcoded credential assignment' = '(?im)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret|password|passwd)\b\s*[:=]\s*["'']?[A-Za-z0-9_./+=-]{16,}'
    }

    $findings = New-Object System.Collections.Generic.List[object]
    $maximumBlobBytes = [int64]$MaximumBlobMiB * 1MB
    $blobCount = 0

    foreach ($line in $metadata) {
        $parts = $line -split ' '
        if ($parts.Count -lt 3 -or $parts[1] -ne 'blob') {
            continue
        }

        $blobCount++
        $oid = $parts[0]
        $size = [int64]$parts[2]
        $path = '(historical path unavailable)'
        if ($pathByOid.ContainsKey($oid)) {
            $path = $pathByOid[$oid]
        }

        if ($size -gt $maximumBlobBytes) {
            $findings.Add([pscustomobject]@{
                    Type = "Blob exceeds $MaximumBlobMiB MiB review limit"
                    Oid  = $oid
                    Path = $path
                })
            continue
        }

        $contentLines = @(& git cat-file blob $oid 2>$null)
        if ($LASTEXITCODE -ne 0) {
            throw "Could not read Git blob $oid."
        }
        $content = [string]::Join("`n", [string[]]$contentLines)

        foreach ($entry in $patterns.GetEnumerator()) {
            if ([regex]::IsMatch(
                    $content,
                    $entry.Value,
                    [System.Text.RegularExpressions.RegexOptions]::CultureInvariant)) {
                $findings.Add([pscustomobject]@{
                        Type = $entry.Key
                        Oid  = $oid
                        Path = $path
                    })
            }
        }
    }

    $sensitivePathPattern = '(^|/)(\.env($|\.)|id_(rsa|dsa|ecdsa|ed25519)$|.*\.(pem|p12|pfx|key|jks|keystore|kdbx)$|credentials($|\.)|secrets?($|\.)|auth\.json$|netrc$|\.npmrc$|\.pypirc$)'
    $binaryPathPattern = '\.(dll|exe|lib|obj|pdb|ptx|hsaco|onnx|safetensors|gguf|bin|zip|7z|tar|gz|whl)$'
    foreach ($row in $objectRows) {
        $parts = $row -split ' ', 2
        if ($parts.Count -ne 2) {
            continue
        }

        $oid = $parts[0]
        $path = $parts[1]
        if ($path -match $sensitivePathPattern) {
            $findings.Add([pscustomobject]@{
                    Type = 'Sensitive filename requires review'
                    Oid  = $oid
                    Path = $path
                })
        }
        if ($path -match $binaryPathPattern) {
            $findings.Add([pscustomobject]@{
                    Type = 'Binary or release artifact in history'
                    Oid  = $oid
                    Path = $path
                })
        }
    }

    $uniqueFindings = @($findings | Sort-Object Type, Path, Oid -Unique)
    Write-Host "Scanned $blobCount reachable Git blobs across every local ref."
    if ($uniqueFindings.Count -gt 0) {
        Write-Host 'Release-blocking candidates were found. No matched value is printed.' -ForegroundColor Red
        $uniqueFindings | Format-Table -AutoSize Type, Oid, Path
        throw 'Public-source audit failed. Review every candidate and rotate any real credential before publication.'
    }

    Write-Host 'Public-source history audit passed: no credential, sensitive-path, binary-artifact, or oversized-blob candidates.' -ForegroundColor Green
}
finally {
    Pop-Location
}
