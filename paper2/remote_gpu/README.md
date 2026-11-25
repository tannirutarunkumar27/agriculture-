# Paper 2 remote-GPU execution

The local Windows host has no NVIDIA runtime, Azure CLI/subscription, or SSH
host configured. Do not execute the RR06 study locally when following the
remote-GPU plan.

Required remote host:

- Linux GPU machine with NVIDIA drivers, CUDA, Python 3.12, and SSH access.
- The repository and frozen public-data package copied to the host.
- `torch.cuda.is_available()` must be true on the remote host.
- One worker process per physical GPU; never start more workers than
  `torch.cuda.device_count()`.

The current verified local study has 441/504 candidates complete and 63
pending. Its completed receipts remain valid CPU provenance. Before claiming a
GPU execution, the remote runner must use a new execution package whose
manifest records the CUDA device, GPU name, driver/runtime, package versions,
worker count, and protocol hash. GPU results must not be merged into the CPU
study unless the exact protocol and device policy are explicitly reconciled.

Use `run_remote_rr06.ps1` only after supplying a real SSH host and private-key
path. It requires strict host-key verification, verifies the bundle before
upload, rejects reserved/loopback targets and unsafe paths, retrieves the
completed study, and runs a separate local audit without model inference.

The GPU launcher creates a separate `restart_matched_neural_study_gpu_v1`
because changing CPU/GPU execution code changes the frozen execution hash. It
does not silently merge the 441 CPU receipts into the GPU study.
