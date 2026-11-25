"""Fail-closed preflight for the local WSL/Linux Paper 2 execution profile.

This command performs no training and installs nothing.  It records whether the
current Linux runtime can freeze and execute the CPU-only Flower study without
silently falling back to the unsupported native-Windows Ray path.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import sys

from paper2.restart.validation_search import ROOT, atomic_json, read, sha


RESULTS = ROOT / "paper2/results"
DATA_MANIFEST = RESULTS / "restart_public_data_v1/manifest.json"
SECURITY_REGISTRY = ROOT / "paper2/configs/upstream_security_registry_v1.json"
MIN_CPUS = 10
MIN_MEMORY_BYTES = 6 * 1024**3
MIN_DISK_BYTES = 10 * 1024**3
REQUIRED_DEPENDENCIES = {
    "numpy": "2.5.1",
    "pandas": "3.0.3",
    "scikit-learn": "1.9.0",
    "joblib": "1.5.3",
    "xgboost": "3.3.0",
    "ray": "2.55.1",
    "flwr": "1.42.0",
    # The local-version suffix is security/reproducibility relevant here: it
    # proves that the CPU wheel was installed instead of a CUDA-enabled build.
    "torch": "2.13.0+cpu",
    "cryptography": "50.0.2",
}


def _memory_bytes() -> int:
    values = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0]) * 1024
    return values["MemTotal"]


def collect_snapshot() -> dict:
    dependencies = {}
    for name in REQUIRED_DEPENDENCIES:
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    torch_cuda_available = None
    if dependencies["torch"] is not None:
        import torch
        torch_cuda_available = bool(torch.cuda.is_available())
    disk_target = ROOT if ROOT.exists() else Path.cwd()
    disk = shutil.disk_usage(disk_target)
    return {
        "platform_system": platform.system(),
        "kernel_release": platform.release(),
        "python": platform.python_version(),
        "python_tuple": list(sys.version_info[:3]),
        "cpu_count": os.cpu_count() or 0,
        "memory_bytes": _memory_bytes() if platform.system() == "Linux" else 0,
        "disk_free_bytes": disk.free,
        "dependencies": dependencies,
        "torch_cuda_available": torch_cuda_available,
        "data_manifest_exists": DATA_MANIFEST.is_file(),
        "security_registry_exists": SECURITY_REGISTRY.is_file(),
    }


def evaluate(snapshot: dict) -> dict:
    python_tuple = tuple(snapshot.get("python_tuple", (0, 0, 0)))
    dependencies = snapshot.get("dependencies", {})
    dependency_checks = {
        name: dependencies.get(name) == expected
        for name, expected in REQUIRED_DEPENDENCIES.items()
    }
    registry_safe = False
    if snapshot.get("security_registry_exists") and SECURITY_REGISTRY.is_file():
        registry = read(SECURITY_REGISTRY)
        registry_safe = (
            registry.get("production_security_claim_allowed") is False
            and registry.get("all_known_security_issues_resolved") is False
        )
    checks = {
        "linux_runtime": snapshot.get("platform_system") == "Linux",
        "python_supported": (3, 11) <= python_tuple[:2] < (3, 15),
        "cpu_capacity": snapshot.get("cpu_count", 0) >= MIN_CPUS,
        "memory_capacity": snapshot.get("memory_bytes", 0) >= MIN_MEMORY_BYTES,
        "disk_capacity": snapshot.get("disk_free_bytes", 0) >= MIN_DISK_BYTES,
        "dependencies_exact": all(dependency_checks.values()),
        "prepared_data_present": snapshot.get("data_manifest_exists") is True,
        "known_security_limit_recorded": registry_safe,
        "cpu_only_declared": snapshot.get("torch_cuda_available") is False,
    }
    passed = all(checks.values())
    return {
        "status": "PASS_LOCAL_LINUX_CPU_PREFLIGHT" if passed else "BLOCKED_LOCAL_LINUX_CPU_PREFLIGHT",
        "checks": checks,
        "dependency_checks": dependency_checks,
        "snapshot": snapshot,
        "environment_kind": (
            "WSL2" if snapshot.get("platform_system") == "Linux" and "microsoft" in str(snapshot.get("kernel_release", "")).lower()
            else "native_linux" if snapshot.get("platform_system") == "Linux" else "unsupported"
        ),
        "training_performed": False,
        "production_security_claim_allowed": False,
    }


def audit(output: Path) -> dict:
    output = Path(output).resolve()
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    result = evaluate(collect_snapshot())
    output.mkdir(parents=True)
    atomic_json(output / "preflight.json", result)
    atomic_json(output / "manifest.json", {
        "status": result["status"],
        "preflight_sha256": sha(output / "preflight.json"),
        "source_sha256": sha(Path(__file__)),
        "data_manifest_sha256": sha(DATA_MANIFEST) if DATA_MANIFEST.is_file() else None,
        "security_registry_sha256": sha(SECURITY_REGISTRY) if SECURITY_REGISTRY.is_file() else None,
        "training_performed": False,
    })
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.output)
    print({"status": report["status"], "checks": report["checks"]})
