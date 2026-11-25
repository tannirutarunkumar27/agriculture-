#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import importlib.metadata
import shutil
import sys
import torch

assert sys.version_info >= (3, 10), sys.version
assert torch.cuda.is_available(), "CUDA is unavailable"
count = torch.cuda.device_count()
assert count >= 1, "No CUDA device detected"
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"gpu_count={count}")
for index in range(count):
    print(f"gpu_{index}={torch.cuda.get_device_name(index)}")
free = shutil.disk_usage('.').free
print(f"free_disk_bytes={free}")
assert free >= 10 * 1024**3, "At least 10 GiB free disk is required"
for name in ("numpy", "pandas", "scikit-learn", "joblib", "xgboost", "pypdf", "ray", "flwr", "cryptography"):
    print(f"{name}={importlib.metadata.version(name)}")
PY

command -v nvidia-smi >/dev/null
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
