# Remote GPU execution plan

The local execution was stopped at 441/504 candidates and independently audited
under the prior v7 execution package (`restart_matched_neural_audit_v15`).
The remaining 63 CPU candidates are preserved as provenance; they are not
silently merged into a GPU run because the CUDA-enabled runner has a different
execution source hash.

The final GPU-capable source is frozen in
`results/restart_matched_neural_execution_v9/`. It supports:

- explicit `PAPER2_DEVICE=cuda` selection;
- fail-closed CUDA availability and GPU-count checks;
- one process per physical GPU;
- CPU aggregation with GPU local model updates;
- a separate v9 GPU study; v7 CPU receipts are not rebound or silently merged
  because the execution source identities differ;
- no outer-test inference or selection before all 504 candidates complete.

The current environment has no NVIDIA runtime, no Azure CLI/subscription, and
no SSH host configuration. Therefore no remote execution has been claimed.
The operator must provide a Linux GPU SSH host and run:

```powershell
.\paper2\remote_gpu\run_remote_rr06.ps1 -SshHost user@host -RemoteRoot /work/paper2 -Workers <gpu_count> -Cap 504
```

The remote host must have Python 3.12, compatible PyTorch/CUDA, and enough GPUs
for the requested worker count. The launcher starts a new v9 GPU study, checks
CUDA remotely, and fails if the target is local, already occupied, or lacks
GPUs. A full GPU study requires all 504 candidates. The archived v7 CPU study
remains independently valid at 441/504, with 63 CPU candidates pending.
