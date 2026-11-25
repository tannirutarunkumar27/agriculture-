"""Freeze a prospective 20-round central-DP plus SecAgg+ study; no execution."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from paper2.restart.freeze_dp_secagg_pilot import DATA, RESULTS, SOURCES as PILOT_SOURCES, dp_payload
from paper2.restart.validation_search import atomic_json, read, runtime, sha

ROUNDS = 20
SOURCES = tuple(dict.fromkeys((*PILOT_SOURCES, Path(__file__), Path(__file__).with_name("run_dp_secagg_study.py"))))


def study_payload(data=DATA):
    values = dp_payload(data, rounds=ROUNDS)
    values["release_policy.json"] = {
        "adaptive_model_releases": ROUNDS,
        "client_metrics_released": False, "private_validation_released": False,
        "intermediate_checkpoints_released": False, "noise_seeds_released": False,
        "raw_or_individual_updates_released": False,
        "preprocessing": "fixed public transform bound to prepared-data manifest",
        "selection": "none; fixed protocol only",
        "qualification": "Every adaptive round is accounted even when only the final model is retained.",
    }
    return values


def freeze(output, data=DATA):
    output, data = Path(output).resolve(), Path(data).resolve()
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    values = study_payload(data); output.mkdir(parents=True); (output / "source_snapshot").mkdir()
    for source in SOURCES: shutil.copy2(source, output / "source_snapshot" / source.name)
    for name, value in values.items(): atomic_json(output / name, value)
    atomic_json(output / "manifest.json", {
        "status": "FROZEN_MULTIROUND_DP_SECAGG_STUDY_NO_EXECUTION", "rounds": ROUNDS,
        "runtime": runtime(), "data_manifest_sha256": sha(data / "manifest.json"),
        "source_sha256": {p.name: sha(p) for p in SOURCES},
        "artifacts": {name: sha(output / name) for name in values},
        "training_started": False, "outer_test_inference": False,
    })
    return {"status": "FROZEN_MULTIROUND_DP_SECAGG_STUDY_NO_EXECUTION", "rounds": ROUNDS}


def verify(package, data_override=None):
    package = Path(package).resolve(); manifest = read(package / "manifest.json")
    if manifest.get("status") != "FROZEN_MULTIROUND_DP_SECAGG_STUDY_NO_EXECUTION" or manifest.get("runtime") != runtime() or manifest.get("rounds") != ROUNDS:
        raise ValueError("Study status/runtime/rounds changed")
    for source in SOURCES:
        expected = manifest["source_sha256"].get(source.name)
        if expected != sha(source) or sha(package / "source_snapshot" / source.name) != expected:
            raise ValueError("Study source changed")
    frozen_run = read(package / "run_config.json")
    data = Path(data_override).resolve() if data_override is not None else Path(frozen_run["data-root"])
    if sha(data / "manifest.json") != manifest["data_manifest_sha256"]:
        raise ValueError("Prepared data binding changed")
    if any(sha(package / name) != digest for name, digest in manifest["artifacts"].items()):
        raise ValueError("Study artifact changed")
    expected = study_payload(data); expected["run_config.json"]["data-root"] = frozen_run["data-root"]
    if any(read(package / name) != value for name, value in expected.items()):
        raise ValueError("Study semantics changed")
    return {"status": "PASS_FROZEN_MULTIROUND_DP_SECAGG_STUDY", "rounds": ROUNDS, "training_started": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify")); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-override", type=Path)
    args = parser.parse_args(); print(freeze(args.output) if args.action == "freeze" else verify(args.output, args.data_override))
