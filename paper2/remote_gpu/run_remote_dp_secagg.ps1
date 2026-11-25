param(
    [Parameter(Mandatory=$true)][string]$SshHost,
    [Parameter(Mandatory=$true)][string]$SshKey,
    [Parameter(Mandatory=$true)][string]$RemoteRoot,
    [int]$SshPort = 22,
    [string]$OutputName = "restart_dp_secagg_pilot_execution_remote_v1"
)

$ErrorActionPreference = "Stop"
if ($SshHost -notmatch '^[A-Za-z0-9._-]+@[A-Za-z0-9.-]+$') { throw "SSH host must use the safe user@host form." }
$hostName = ($SshHost -split '@')[-1].ToLowerInvariant()
if ($hostName -in @('localhost', '127.0.0.1', '::1') -or $hostName -match '\.(example|invalid|test)$') {
    throw "A real remote SSH host is required; local execution is intentionally disabled."
}
if ($RemoteRoot -notmatch '^/[A-Za-z0-9._/-]+$' -or ($RemoteRoot -split '/') -contains '..') { throw "RemoteRoot must be a safe absolute POSIX path." }
if ($OutputName -notmatch '^[A-Za-z0-9._-]+$') { throw "OutputName contains unsafe characters." }
if ($SshPort -lt 1 -or $SshPort -gt 65535) { throw "SshPort must be between 1 and 65535." }
if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) { throw "OpenSSH ssh.exe is required." }
if (-not (Get-Command scp -ErrorAction SilentlyContinue)) { throw "OpenSSH scp.exe is required." }
$keyPath = (Resolve-Path -LiteralPath $SshKey -ErrorAction Stop).Path
if (-not (Test-Path -LiteralPath $keyPath -PathType Leaf)) { throw "SSH private key file is missing." }

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$bundleRoots = @(Get-ChildItem (Join-Path $root "paper2/results") -Directory | Where-Object {
    $_.Name -match '^restart_dp_secagg_remote_bundle_v\d+$'
} | Sort-Object { [int]($_.Name -replace '^.*_v', '') })
if ($bundleRoots.Count -eq 0) { throw "No verified DP+SecAgg bundle is available." }
$bundleRoot = $bundleRoots[-1].FullName
$source = Join-Path $bundleRoot "paper2"
if (-not (Test-Path (Join-Path $bundleRoot "bundle_manifest.json"))) { throw "Verified DP+SecAgg bundle is missing." }
$bundleManifest = Get-Content -Raw (Join-Path $bundleRoot "bundle_manifest.json") | ConvertFrom-Json
$pilotPackages = @($bundleManifest.packages | Where-Object { $_ -match '^restart_dp_secagg_pilot_protocol_v\d+$' })
if ($pilotPackages.Count -ne 1) { throw "Bundle must contain exactly one frozen DP+SecAgg pilot protocol." }
$pilotPackage = $pilotPackages[0]
$localExecution = Join-Path $root "paper2/results/$OutputName"
$auditName = "${OutputName}_independent_audit_v1"
$localAudit = Join-Path $root "paper2/results/$auditName"
if ((Test-Path $localExecution) -or (Test-Path $localAudit)) { throw "Local execution or audit output already exists." }
$python = Join-Path $root ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Local verification Python is missing." }
$sshOptions = @('-i', $keyPath, '-p', "$SshPort", '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes')
$scpOptions = @('-i', $keyPath, '-P', "$SshPort", '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes')
Push-Location $root
try {
    & $python -B -m paper2.restart.verify_remote_bundle --manifest (Join-Path $bundleRoot "bundle_manifest.json")
    if ($LASTEXITCODE -ne 0) { throw "Local DP+SecAgg bundle verification failed." }
} finally { Pop-Location }
$remoteProject = "$RemoteRoot/paper2_privacy_pilot"
& ssh @sshOptions $SshHost "test ! -e '$remoteProject' && mkdir -p '$remoteProject'"
if ($LASTEXITCODE -ne 0) { throw "Remote project path exists or could not be created." }

& scp @scpOptions -r $source "$SshHost`:$remoteProject/"
if ($LASTEXITCODE -ne 0) { throw "Remote copy failed." }
& scp @scpOptions (Join-Path $bundleRoot "bundle_manifest.json") "$SshHost`:$remoteProject/bundle_manifest.json"
if ($LASTEXITCODE -ne 0) { throw "Remote bundle manifest copy failed." }

$command = "cd '$remoteProject' && python3 -B -m paper2.restart.verify_remote_bundle --manifest bundle_manifest.json && bash paper2/remote_gpu/remote_privacy_preflight.sh && python3 -B -m paper2.restart.run_dp_secagg_pilot --package paper2/results/$pilotPackage --data-root paper2/results/restart_public_data_v1 --output paper2/results/$OutputName --execute"
& ssh @sshOptions $SshHost $command
if ($LASTEXITCODE -ne 0) { throw "Remote DP+SecAgg pilot failed; inspect its frozen output logs." }

& scp @scpOptions -r "$SshHost`:$remoteProject/paper2/results/$OutputName" (Join-Path $root "paper2/results")
if ($LASTEXITCODE -ne 0) { throw "Remote result retrieval failed." }
Push-Location $root
try {
    & $python -B -m paper2.restart.verify_dp_secagg_execution --execution "paper2/results/$OutputName" --protocol "paper2/results/$pilotPackage" --output "paper2/results/$auditName"
    if ($LASTEXITCODE -ne 0) { throw "Retrieved DP+SecAgg pilot audit failed." }
} finally { Pop-Location }
