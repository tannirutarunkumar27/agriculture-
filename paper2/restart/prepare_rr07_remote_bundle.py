"""Create a hash-manifested RR07 CUDA bundle from a frozen execution package."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from paper2.restart.rr07_execution_gate import verify_execution
from paper2.restart.validation_search import ROOT, atomic_json, sha


RESULTS = ROOT / "paper2/results"


def prepare(output, package):
    output, package = Path(output).resolve(), Path(package).resolve()
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    manifest, _, data = verify_execution(package)
    project = output / "paper2"
    shutil.copytree(ROOT / "paper2/restart", project / "restart")
    for name in ("__init__.py", "federated.py"):
        shutil.copy2(ROOT / "paper2" / name, project / name)
    shutil.copytree(ROOT / "paper2/remote_gpu", project / "remote_gpu")
    (project / "results").mkdir()
    shutil.copytree(data, project / "results" / data.name)
    plan_package = Path(manifest["plan_package"])
    shutil.copytree(plan_package, project / "results" / plan_package.name)
    shutil.copytree(package, project / "results" / package.name)
    inventory = {path.relative_to(output).as_posix(): sha(path) for path in sorted(project.rglob("*")) if path.is_file()}
    total_bytes = sum(path.stat().st_size for path in project.rglob("*") if path.is_file())
    atomic_json(output / "bundle_manifest.json", {
        "status": "RR07_REMOTE_GPU_BUNDLE_READY_NO_OUTER_INFERENCE", "execution_package": package.name,
        "execution_manifest_sha256": sha(package / "manifest.json"), "plan_package": plan_package.name,
        "data": data.name, "files": inventory, "file_count": len(inventory), "bytes": total_bytes,
    })
    return {"status": "RR07_REMOTE_GPU_BUNDLE_READY_NO_OUTER_INFERENCE", "file_count": len(inventory), "bytes": total_bytes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args(); print(prepare(args.output, args.package))
