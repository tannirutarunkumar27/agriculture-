# Paper 2 GPU transition status

The local Windows batch was intentionally stopped after reaching 441/504
candidates. Its receipts remain preserved; no local workers are running now.

CUDA support was added to the local-update primitive and runner, including
explicit device selection and one-worker-per-GPU checks. The final no-training
execution package is `results/restart_matched_neural_execution_v9/`; its dry
run and 14 targeted tests pass.

No remote training has been claimed because this environment has no configured
GPU server, SSH target, NVIDIA runtime, Azure CLI/subscription, or cloud
credentials. The fail-closed launcher is
`paper2/remote_gpu/run_remote_rr06.ps1`. It will reject localhost targets,
check remote CUDA, copy the frozen package, and launch a separate GPU study.
