# Paper 2 remote execution checklist

This checklist is the required order for the external Linux/GPU work. The
PowerShell launchers reject blank, localhost, and loopback targets; no local
training is allowed.

## Host acceptance

1. Obtain an SSH host, non-root working directory, key-based authentication,
   and an NVIDIA GPU allocation.
2. Confirm Python 3.10+, CUDA-enabled PyTorch, 10 GiB free disk, and one
   worker per physical GPU.
3. Install the candidate environment from
   `paper2/requirements-linux-lock-proposed.txt`, then record the actual
   package index, wheel hashes, CUDA version, driver version, GPU model, and
   `pip freeze` output.
4. Create `paper2/requirements-linux-lock.txt` only after those checks pass.
   It must include exact `# STATUS=REMOTE_VERIFIED` and `# PLATFORM=Linux`
   lines, exact versions, and a wheel SHA-256 hash for every requirement. A
   proposed lock or marker-only file is not release evidence.
5. Copy `paper2/remote_gpu/clean_linux_receipt.example.json` into the retrieved
   result package, replace every placeholder with observed evidence, and bind
   `linux_lock_sha256` to the verified lock. The release gate requires all five
   checks and an independent audit to pass; file existence alone is rejected.

## Execution order

After SSH access is available:

```powershell
pwsh -File paper2/remote_gpu/run_remote_rr06.ps1 `
  -SshHost <user>@<host> -SshKey <private-key-path> -SshPort 22 `
  -RemoteRoot /srv/paper2 -Workers <gpu-count> -Cap 504

pwsh -File paper2/remote_gpu/run_remote_dp_secagg.ps1 `
  -SshHost <user>@<host> -SshKey <private-key-path> -SshPort 22 `
  -RemoteRoot /srv/paper2

pwsh -File paper2/remote_gpu/run_remote_dp_secagg_study.ps1 `
  -SshHost <user>@<host> -SshKey <private-key-path> -SshPort 22 `
  -RemoteRoot /srv/paper2
```

RR07 must not start until the RR06 receipt passes independent audit and the
RR06 model-selection record is frozen.

## Required receipts

- authenticated host identity and remote preflight output;
- bundle and protocol hashes;
- CUDA/driver/GPU/package environment;
- per-run attempt journal and result hashes;
- interruption/recovery test;
- secure result-transfer receipt;
- SecAgg+ and central-DP receipts for every round and release;
- independent audit sign-off.

For authenticated-network validation, copy
`paper2/remote_gpu/network_deployment_evidence.example.json`, preserve seven
separate process identities, replace every placeholder with observed values,
and run `paper2.restart.network_deployment_gate` against that evidence. Logical
clients sharing one process, duplicate endpoints, non-Linux nodes, missing TLS
peer certificates, or unhashed receipts are rejected.
