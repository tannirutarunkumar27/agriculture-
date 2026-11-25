"""Freeze a one-round, no-evaluation Flower SecAgg+ pilot; never runs it."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from paper2.restart.flower_secagg import SecAggPilotConfig, protocol_record, validate
from paper2.restart.validation_search import ROOT, atomic_json, read, runtime, sha


RESULTS = ROOT / "paper2/results"
DATA = RESULTS / "restart_public_data_v1"
SOURCES = (
    ROOT / "paper2/secagg_pilot/task.py",
    ROOT / "paper2/secagg_pilot/client_app.py",
    ROOT / "paper2/secagg_pilot/server_app.py",
    ROOT / "paper2/restart/flower_secagg.py",
    Path(__file__),
)


def payload(data=DATA, fold_id="outer_0"):
    data = Path(data).resolve()
    manifest = read(data / "manifest.json")
    if manifest.get("status") != "PREPARED_NO_TRAINING" or fold_id not in manifest.get("assets", {}):
        raise ValueError("Prepared public data and a known outer fold are required")
    partition = read(data / "folds" / fold_id / "partition.json")
    if partition.get("kind") != "outer_refit":
        raise ValueError("Pilot must use outer-refit training clients")
    roster = sorted(partition.get("clients", {}))
    config = SecAggPilotConfig()
    validate(config)
    if len(roster) != config.participants or any(not partition["clients"][name] for name in roster):
        raise ValueError("Frozen roster must match secure-aggregation participant count")
    return {
        "run_config.json": {
            "num-server-rounds": 1, "learning-rate": 0.01, "features": 54,
            "fold-id": fold_id, "data-root": str(data), "client-roster": ",".join(roster),
            "outer_test_evaluation": False, "differential_privacy_enabled": False,
        },
        "roster.json": {"fold": fold_id, "clients": roster,
                        "training_row_counts": {name: len(partition["clients"][name]) for name in roster}},
        "security_scope.json": {**protocol_record(config), "outer_test_evaluation": False,
            "raw_updates_exported": False,
            "limitations": ["single public dataset represented as logical clients", "no differential privacy", "no network deployment", "not a production security audit"]},
    }


def freeze(output, data=DATA):
    output = Path(output).resolve()
    if RESULTS not in output.parents or output.exists() or output == DATA or DATA in output.parents:
        raise ValueError("Output must be a new separate results directory")
    values = payload(data)
    output.mkdir(parents=True)
    (output / "source_snapshot").mkdir()
    for source in SOURCES:
        shutil.copyfile(source, output / "source_snapshot" / source.name)
    for name, value in values.items():
        atomic_json(output / name, value)
    atomic_json(output / "manifest.json", {
        "status": "FROZEN_SECAGG_PILOT_NO_EXECUTION", "runtime": runtime(),
        "data_manifest_sha256": sha(Path(data) / "manifest.json"),
        "source_sha256": {source.name: sha(source) for source in SOURCES},
        "artifacts": {name: sha(output / name) for name in values},
        "training_started": False, "outer_test_inference": False,
    })
    return {"status": "FROZEN_SECAGG_PILOT_NO_EXECUTION", "participants": 7, "rounds": 1}


def verify(package):
    package = Path(package).resolve()
    manifest = read(package / "manifest.json")
    if manifest.get("status") != "FROZEN_SECAGG_PILOT_NO_EXECUTION" or manifest.get("runtime") != runtime():
        raise ValueError("Package status/runtime differs")
    for source in SOURCES:
        if sha(source) != manifest["source_sha256"].get(source.name) or sha(package / "source_snapshot" / source.name) != sha(source):
            raise ValueError("Source binding changed")
    data = Path(read(package / "run_config.json")["data-root"])
    if sha(data / "manifest.json") != manifest["data_manifest_sha256"]:
        raise ValueError("Data binding changed")
    expected = payload(data)
    if any(sha(package / name) != digest for name, digest in manifest["artifacts"].items()) or any(read(package / name) != value for name, value in expected.items()):
        raise ValueError("Protocol artifact or semantic binding changed")
    return {"status": "PASS_FROZEN_SECAGG_PILOT", "training_started": False, "outer_test_inference": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify")); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(freeze(args.output) if args.action == "freeze" else verify(args.output))
