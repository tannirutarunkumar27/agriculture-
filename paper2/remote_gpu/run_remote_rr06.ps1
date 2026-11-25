param(
    [Parameter(Mandatory=$true)][string]$SshHost,
    [Parameter(Mandatory=$true)][string]$SshKey,
    [Parameter(Mandatory=$true)][string]$RemoteRoot,
    [int]$SshPort = 22,
    [int]$Workers = 1,
    [int]$Cap = 504,
    [string]$StudyName = "restart_matched_neural_study_gpu_v1"
)

$ErrorActionPreference = "Stop"
if ($SshHost -notmatch '^[A-Za-z0-9._-]+@[A-Za-z0-9.-]+$') {
    throw "SSH host must use the safe user@host form."
}
$hostName = ($SshHost -split '@')[-1].ToLowerInvariant()
if ($hostName -in @('localhost', '127.0.0.1', '::1') -or $hostName -match '\.(example|invalid|test)$') {
    throw "A real remote SSH host is required; local execution is intentionally disabled."
}
if ($RemoteRoot -notmatch '^/[A-Za-z0-9._/-]+$' -or ($RemoteRoot -split '/') -contains '..') {
    throw "RemoteRoot must be a safe absolute POSIX path without parent traversal."
}
if ($StudyName -notmatch '^[A-Za-z0-9._-]+$') { throw "StudyName contains unsafe characters." }
if ($SshPort -lt 1 -or $SshPort -gt 65535) { throw "SshPort must be between 1 and 65535." }
if ($Workers -lt 1 -or $Workers -gt 8) { throw "Workers must be between 1 and 8." }
if ($Cap -ne 504) { throw "The clean RR06 study requires exactly 504 planned candidates." }
if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) { throw "OpenSSH ssh.exe is required." }
if (-not (Get-Command scp -ErrorAction SilentlyContinue)) { throw "OpenSSH scp.exe is required." }
$keyPath = (Resolve-Path -LiteralPath $SshKey -ErrorAction Stop).Path
if (-not (Test-Path -LiteralPath $keyPath -PathType Leaf)) { throw "SSH private key file is missing." }

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$remoteProject = "$RemoteRoot/paper2_project"
$bundleRoot = Join-Path $root "paper2/results/restart_rr06_remote_bundle_v2"
$source = Join-Path $bundleRoot "paper2"
if (-not (Test-Path (Join-Path (Split-Path $source -Parent) "bundle_manifest.json"))) {
    throw "Verified RR06 remote bundle is missing."
}
$localStudy = Join-Path $root "paper2/results/$StudyName"
$auditName = "${StudyName}_independent_audit_v1"
$localAudit = Join-Path $root "paper2/results/$auditName"
if ((Test-Path $localStudy) -or (Test-Path $localAudit)) {
    throw "Local study or audit output already exists; immutable outputs cannot be overwritten."
}
$python = Join-Path $root ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Local verification Python is missing." }
$sshOptions = @('-i', $keyPath, '-p', "$SshPort", '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes')
$scpOptions = @('-i', $keyPath, '-P', "$SshPort", '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes')

Push-Location $root
try {
    & $python -B -m paper2.restart.verify_remote_bundle --manifest (Join-Path $bundleRoot "bundle_manifest.json")
    if ($LASTEXITCODE -ne 0) { throw "Local RR06 bundle verification failed." }
} finally {
    Pop-Location
}

Write-Host "Copying frozen Paper 2 package to remote host $SshHost ..."
& ssh @sshOptions $SshHost "test ! -e '$remoteProject' && mkdir -p '$remoteProject'"
if ($LASTEXITCODE -ne 0) { throw "Remote project path exists or could not be created." }
& scp @scpOptions -r $source "$SshHost`:$remoteProject/"
if ($LASTEXITCODE -ne 0) { throw "Remote copy failed." }
& scp @scpOptions (Join-Path $bundleRoot "bundle_manifest.json") "$SshHost`:$remoteProject/bundle_manifest.json"
if ($LASTEXITCODE -ne 0) { throw "Remote bundle manifest copy failed." }
$gpuCheck = 'import torch; assert torch.cuda.device_count() >= ' + $Workers + ', "not enough GPUs"; print("selected_gpu_count=", torch.cuda.device_count())'
$command = "cd '$remoteProject' && test ! -e 'paper2/results/$StudyName' && python3 -B -m paper2.restart.verify_remote_bundle --manifest bundle_manifest.json && bash paper2/remote_gpu/remote_preflight.sh && python3 -c '$gpuCheck' && PAPER2_DEVICE=cuda python3 -B -m paper2.restart.matched_neural_runner --package paper2/results/restart_matched_neural_execution_v10 --protocol-override paper2/results/restart_matched_neural_protocol_v7 --data-override paper2/results/restart_public_data_v1 --output paper2/results/$StudyName --execute --workers $Workers --cap $Cap"
Write-Host "Starting remote GPU execution; no local training will be launched."
& ssh @sshOptions $SshHost $command
if ($LASTEXITCODE -ne 0) { throw "Remote GPU execution failed; inspect the remote attempt journals." }

Write-Host "Retrieving immutable RR06 results for local independent audit ..."
& scp @scpOptions -r "$SshHost`:$remoteProject/paper2/results/$StudyName" (Join-Path $root "paper2/results")
if ($LASTEXITCODE -ne 0) { throw "Remote result retrieval failed." }
Push-Location $root
try {
    & $python -B -m paper2.restart.verify_matched_neural --package paper2/results/restart_matched_neural_execution_v10 --study "paper2/results/$StudyName" --output "paper2/results/$auditName"
    if ($LASTEXITCODE -ne 0) { throw "Retrieved RR06 result audit failed." }
} finally {
    Pop-Location
}
