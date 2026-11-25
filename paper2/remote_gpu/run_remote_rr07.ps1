param(
    [Parameter(Mandatory=$true)][string]$SshHost,
    [Parameter(Mandatory=$true)][string]$SshKey,
    [Parameter(Mandatory=$true)][string]$RemoteRoot,
    [Parameter(Mandatory=$true)][string]$BundleName,
    [int]$SshPort = 22,
    [int]$Cap = 160,
    [string]$StudyName = "restart_rr07_outer_gpu_v1"
)

$ErrorActionPreference = "Stop"
if ($SshHost -notmatch '^[A-Za-z0-9._-]+@[A-Za-z0-9.-]+$') { throw "SSH host must use the safe user@host form." }
$hostName = ($SshHost -split '@')[-1].ToLowerInvariant()
if ($hostName -in @('localhost', '127.0.0.1', '::1') -or $hostName -match '\.(example|invalid|test)$') { throw "A real remote SSH host is required." }
if ($RemoteRoot -notmatch '^/[A-Za-z0-9._/-]+$' -or ($RemoteRoot -split '/') -contains '..') { throw "RemoteRoot must be a safe absolute POSIX path." }
if ($BundleName -notmatch '^[A-Za-z0-9._-]+$' -or $StudyName -notmatch '^[A-Za-z0-9._-]+$') { throw "BundleName and StudyName must be safe names." }
if ($SshPort -lt 1 -or $SshPort -gt 65535 -or $Cap -ne 160) { throw "RR07 requires a valid SSH port and all 160 planned runs." }
if (-not (Get-Command ssh -ErrorAction SilentlyContinue) -or -not (Get-Command scp -ErrorAction SilentlyContinue)) { throw "OpenSSH ssh.exe and scp.exe are required." }
$keyPath = (Resolve-Path -LiteralPath $SshKey -ErrorAction Stop).Path
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$bundleRoot = Join-Path $root "paper2/results/$BundleName"; $source = Join-Path $bundleRoot "paper2"
if (-not (Test-Path (Join-Path $bundleRoot "bundle_manifest.json"))) { throw "Verified RR07 remote bundle is missing." }
$manifest = Get-Content (Join-Path $bundleRoot "bundle_manifest.json") -Raw | ConvertFrom-Json
$packageName = [string]$manifest.execution_package
if ([string]::IsNullOrWhiteSpace($packageName) -or $packageName -notmatch '^[A-Za-z0-9._-]+$') { throw "RR07 bundle execution package name is invalid." }
$localStudy = Join-Path $root "paper2/results/$StudyName"
if (Test-Path $localStudy) { throw "Local RR07 output already exists; immutable outputs cannot be overwritten." }
$python = Join-Path $root ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Local verification Python is missing." }
$sshOptions = @('-i', $keyPath, '-p', "$SshPort", '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes')
$scpOptions = @('-i', $keyPath, '-P', "$SshPort", '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes')
Push-Location $root
try { & $python -B -m paper2.restart.verify_remote_bundle --manifest (Join-Path $bundleRoot "bundle_manifest.json"); if ($LASTEXITCODE -ne 0) { throw "Local RR07 bundle verification failed." } } finally { Pop-Location }
$remoteProject = "$RemoteRoot/paper2_rr07"
& ssh @sshOptions $SshHost "test ! -e '$remoteProject' && mkdir -p '$remoteProject'"
if ($LASTEXITCODE -ne 0) { throw "Remote project path exists or could not be created." }
& scp @scpOptions -r $source "$SshHost`:$remoteProject/"
if ($LASTEXITCODE -ne 0) { throw "Remote RR07 copy failed." }
& scp @scpOptions (Join-Path $bundleRoot "bundle_manifest.json") "$SshHost`:$remoteProject/bundle_manifest.json"
if ($LASTEXITCODE -ne 0) { throw "Remote RR07 manifest copy failed." }
$command = "cd '$remoteProject' && test ! -e 'paper2/results/$StudyName' && python3 -B -m paper2.restart.verify_remote_bundle --manifest bundle_manifest.json && bash paper2/remote_gpu/remote_preflight.sh && PAPER2_DEVICE=cuda python3 -B -m paper2.restart.rr07_runner --package paper2/results/$packageName --output paper2/results/$StudyName --execute --allow-outer-inference --cap $Cap"
& ssh @sshOptions $SshHost $command
if ($LASTEXITCODE -ne 0) { throw "Remote RR07 study failed; inspect immutable attempt journals remotely." }
& scp @scpOptions -r "$SshHost`:$remoteProject/paper2/results/$StudyName" (Join-Path $root "paper2/results")
if ($LASTEXITCODE -ne 0) { throw "Remote RR07 result retrieval failed." }
