"""Create a minimal, hash-manifested Flower DP+SecAgg remote bundle."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from paper2.restart.validation_search import ROOT, atomic_json, sha
from paper2.restart.paper2_completion_audit import latest_result_file

RESULTS = ROOT / "paper2/results"
DATA = RESULTS / "restart_public_data_v1"

def packages() -> tuple[Path, Path]:
    """Bundle the latest verified immutable pilot and fixed-round protocol."""
    return (
        latest_result_file("restart_dp_secagg_pilot_protocol", "manifest.json").parent,
        latest_result_file("restart_dp_secagg_study_protocol", "manifest.json").parent,
    )


def prepare(output: Path) -> dict:
    output = Path(output).resolve()
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    project = output / "paper2"; (project / "restart").mkdir(parents=True)
    for source in (ROOT / "paper2/restart").glob("*.py"): shutil.copy2(source, project / "restart" / source.name)
    for name in ("__init__.py", "federated.py"): shutil.copy2(ROOT / "paper2" / name, project / name)
    for folder in ("secagg_pilot", "secagg_dp_pilot"):
        shutil.copytree(ROOT / "paper2" / folder, project / folder)
    # Carry both the observed runtime manifest and the verified Linux lock so
    # the target-host receipt can bind its environment to exact reviewed inputs.
    for name in ("requirements-runtime-observed.txt", "requirements-linux-lock-proposed.txt",
                 "requirements-linux-lock.txt"):
        shutil.copy2(ROOT / "paper2" / name, project / name)
    (project / "remote_gpu").mkdir()
    for name in ("remote_privacy_preflight.sh", "remote_authenticated_network_collect.sh"):
        shutil.copy2(ROOT / "paper2/remote_gpu" / name, project / "remote_gpu" / name)
    (project / "results").mkdir()
    shutil.copytree(DATA, project / "results" / DATA.name)
    selected_packages = packages()
    for package in selected_packages: shutil.copytree(package, project / "results" / package.name)
    files = sorted(path for path in project.rglob("*") if path.is_file())
    inventory = {path.relative_to(output).as_posix(): sha(path) for path in files}
    atomic_json(output / "bundle_manifest.json", {"status": "DP_SECAGG_REMOTE_BUNDLE_READY_NO_TRAINING",
                "files": inventory, "file_count": len(files), "bytes": sum(p.stat().st_size for p in files),
                "data": DATA.name, "packages": [p.name for p in selected_packages]})
    return {"status": "DP_SECAGG_REMOTE_BUNDLE_READY_NO_TRAINING", "file_count": len(files),
            "bytes": sum(p.stat().st_size for p in files)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); print(prepare(args.output))
