"""Freeze central-DP plus SecAgg+ pilot metadata; no execution or inference."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from paper2.restart.flower_secagg import SecAggPilotConfig, protocol_record
from paper2.restart.privacy_accounting import gaussian_budget, replacement_sensitivity
from paper2.restart.freeze_secagg_pilot import DATA, RESULTS, payload
from paper2.restart.validation_search import ROOT, atomic_json, read, runtime, sha


UPSTREAM_SECURITY = ROOT / "paper2/configs/upstream_security_registry_v1.json"
SOURCES = (
    ROOT / "paper2/secagg_dp_pilot/pyproject.toml", ROOT / "paper2/secagg_dp_pilot/LICENSE",
    ROOT / "paper2/secagg_pilot/dp_client_app.py", ROOT / "paper2/secagg_pilot/dp_server_app.py",
    ROOT / "paper2/secagg_pilot/client_app.py", ROOT / "paper2/secagg_pilot/server_app.py",
    ROOT / "paper2/secagg_pilot/task.py", ROOT / "paper2/restart/flower_secagg.py",
    UPSTREAM_SECURITY,
    ROOT / "paper2/restart/freeze_secagg_pilot.py", ROOT / "paper2/restart/privacy_accounting.py",
    ROOT / "paper2/restart/validation_search.py", ROOT / "paper2/restart/run_dp_secagg_pilot.py",
    ROOT / "paper2/restart/run_dp_secagg_local_simulation.py", Path(__file__),
)
CLIP_NORM, NOISE_MULTIPLIER, DELTA = 1.0, 8.0, 1e-5


def flower_noise_std(clip_norm, noise_multiplier, participants):
    """Match Flower 1.42 compute_stdv: z * C / sampled clients."""
    return noise_multiplier * clip_norm / participants


def dp_payload(data=DATA, rounds=1):
    if type(rounds) is not int or rounds < 1:
        raise ValueError("Positive integer release count required")
    base = payload(data)
    settings = SecAggPilotConfig()
    sensitivity = replacement_sensitivity(CLIP_NORM, settings.participants)
    noise_std = flower_noise_std(CLIP_NORM, NOISE_MULTIPLIER, settings.participants)
    event = {"sensitivity": sensitivity, "noise_std": noise_std, "count": rounds}
    upstream_security = read(UPSTREAM_SECURITY)
    run_config = base["run_config.json"]
    # Flower validates CLI overrides against the pyproject app schema.  Keep
    # executable run-config keys in Flower's canonical hyphenated form.
    run_config.pop("outer_test_evaluation")
    run_config.pop("differential_privacy_enabled")
    run_config |= {
        "num-server-rounds": rounds,
        "noise-multiplier": NOISE_MULTIPLIER,
        "clipping-norm": CLIP_NORM,
        "central-dp": True,
        "differential-privacy-enabled": True,
        "outer-test-evaluation": False,
    }
    base["security_scope.json"].update({
        "upstream_security_registry": upstream_security,
        "production_security_claim_allowed": False,
    })
    base["privacy_contract.json"] = {
        **protocol_record(settings), "differential_privacy_enabled": True,
        "mechanism": "Flower central DP with client-side fixed clipping plus SecAgg+",
        "clip_norm": CLIP_NORM, "noise_multiplier": NOISE_MULTIPLIER, "delta": DELTA,
        "adaptive_releases": rounds, "flower_noise_std": noise_std,
        "conditional_budget": gaussian_budget([event], DELTA),
        "trusted_server_required": True, "individual_updates_hidden_from_server": False,
        "application_privacy_certified": False,
        "known_security_issues_resolved": upstream_security["all_known_security_issues_resolved"],
        "independent_cryptographic_review_complete": upstream_security["independent_cryptographic_review_complete"],
        "qualification": "Prospective fixed-round simulation contract; accounting excludes hyperparameter selection, private preprocessing, metrics, and deployment transcript.",
    }
    base["execution_profile.json"] = {
        "runtime": "flower_managed_local_simulation",
        "num_supernodes": settings.participants,
        "client_resources_num_cpus": 1,
        "client_resources_num_gpus": 0.0,
        "init_args_num_cpus": 10,
        "init_args_num_gpus": 0,
        "managed_superlink_http_port": 39181,
        "runtime_dependency_installation": False,
        "reason": "Pinned dependencies are preinstalled; the dedicated port and terminal cleanup isolate each audited execution.",
    }
    return base


def freeze(output, data=DATA):
    output = Path(output).resolve()
    if RESULTS not in output.parents or output.exists():
        raise ValueError("Output must be a new Paper 2 results directory")
    values = dp_payload(data); output.mkdir(parents=True); (output / "source_snapshot").mkdir()
    for source in SOURCES: shutil.copyfile(source, output / "source_snapshot" / source.name)
    for name, value in values.items(): atomic_json(output / name, value)
    atomic_json(output / "manifest.json", {"status": "FROZEN_DP_SECAGG_PILOT_NO_EXECUTION", "runtime": runtime(),
        "data_manifest_sha256": sha(Path(data) / "manifest.json"), "source_sha256": {p.name: sha(p) for p in SOURCES},
        "artifacts": {name: sha(output / name) for name in values}, "training_started": False, "outer_test_inference": False})
    return {"status": "FROZEN_DP_SECAGG_PILOT_NO_EXECUTION", "rounds": 1, "participants": 7}


def verify(package, data_override=None):
    package = Path(package).resolve(); manifest = read(package / "manifest.json")
    if manifest.get("status") != "FROZEN_DP_SECAGG_PILOT_NO_EXECUTION" or manifest.get("runtime") != runtime():
        raise ValueError("Protocol status/runtime changed")
    for source in SOURCES:
        expected = manifest["source_sha256"].get(source.name)
        if expected != sha(source) or sha(package / "source_snapshot" / source.name) != expected:
            raise ValueError("Protocol source changed")
    frozen_run = read(package / "run_config.json")
    data = Path(data_override).resolve() if data_override is not None else Path(frozen_run["data-root"])
    if sha(data / "manifest.json") != manifest["data_manifest_sha256"]:
        raise ValueError("Data binding changed")
    expected = dp_payload(data); expected["run_config.json"]["data-root"] = frozen_run["data-root"]
    if any(sha(package / name) != digest for name, digest in manifest["artifacts"].items()) or any(read(package / name) != value for name, value in expected.items()):
        raise ValueError("Protocol artifact or semantic binding changed")
    return {"status": "PASS_FROZEN_DP_SECAGG_PILOT", "training_started": False, "outer_test_inference": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); print(freeze(args.output))
