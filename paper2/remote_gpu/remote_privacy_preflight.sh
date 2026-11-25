#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import importlib.metadata
import importlib.util
import sys

assert sys.version_info >= (3, 10), sys.version
required = ("flwr", "torch", "numpy", "pandas", "scikit-learn", "cryptography")
versions = {name: importlib.metadata.version(name) for name in required}
assert versions["flwr"] == "1.42.0", versions["flwr"]
from flwr.server.workflow import SecAggPlusWorkflow
from flwr.client.mod import fixedclipping_mod, secaggplus_mod
assert SecAggPlusWorkflow and fixedclipping_mod and secaggplus_mod
for name, version in versions.items():
    print(f"{name}={version}")
print("privacy_runtime_preflight=PASS")
PY
