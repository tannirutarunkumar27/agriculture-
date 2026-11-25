"""Create a minimal, hash-manifested RR06 remote execution bundle."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from paper2.restart.validation_search import ROOT, atomic_json, sha

RESULTS = ROOT / "paper2/results"
DATA = RESULTS / "restart_public_data_v1"
PROTOCOL = RESULTS / "restart_matched_neural_protocol_v7"
EXECUTION = RESULTS / "restart_matched_neural_execution_v10"


def prepare(output: Path) -> dict:
    output = Path(output).resolve()
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    project = output / "paper2"
    (project / "restart").mkdir(parents=True)
    for source in (ROOT / "paper2/restart").glob("*.py"):
        shutil.copy2(source, project / "restart" / source.name)
    for name in ("__init__.py", "federated.py"):
        shutil.copy2(ROOT / "paper2" / name, project / name)
    (project / "remote_gpu").mkdir()
    shutil.copy2(ROOT / "paper2/remote_gpu/remote_preflight.sh", project / "remote_gpu/remote_preflight.sh")
    (project / "results").mkdir()
    for source in (DATA, PROTOCOL, EXECUTION):
        shutil.copytree(source, project / "results" / source.name)
    files = sorted(path for path in project.rglob("*") if path.is_file())
    inventory = {path.relative_to(output).as_posix(): sha(path) for path in files}
    atomic_json(output / "bundle_manifest.json", {
        "status": "RR06_REMOTE_BUNDLE_READY_NO_TRAINING", "files": inventory,
        "file_count": len(files), "bytes": sum(path.stat().st_size for path in files),
        "protocol": PROTOCOL.name, "execution": EXECUTION.name, "data": DATA.name,
    })
    return {"status": "RR06_REMOTE_BUNDLE_READY_NO_TRAINING", "file_count": len(files),
            "bytes": sum(path.stat().st_size for path in files)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); print(prepare(args.output))
