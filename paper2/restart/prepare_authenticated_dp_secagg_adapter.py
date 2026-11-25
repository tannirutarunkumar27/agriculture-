"""Bind the self-contained deployment FAB source to the frozen DP/SecAgg study.

The original study snapshot remains immutable and simulation-oriented.  This
adapter records that a separate deployment app preserves the frozen scientific
parameters while changing operational details: each authenticated SuperNode
supplies its own local data location and identity.  It is not an execution,
approval, or replacement of the frozen study.
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

# Support both ``python -m paper2.restart...`` and the documented direct
# script invocation from the repository root.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper2.restart.paper2_completion_audit import ROOT, RESULTS, atomic_json, read, sha


APP = ROOT / "paper2/paper2-dp-secagg-deploy"
STUDY = RESULTS / "restart_dp_secagg_study_protocol_v12"
REQUIRED_FILES = (
    "pyproject.toml", "LICENSE", "README.md", "paper2_dp_secagg_deploy/__init__.py",
    "paper2_dp_secagg_deploy/protocol.py", "paper2_dp_secagg_deploy/task.py",
    "paper2_dp_secagg_deploy/client_app.py", "paper2_dp_secagg_deploy/server_app.py",
)


def validate(app: Path = APP, study: Path = STUDY) -> dict:
    app, study = Path(app), Path(study)
    missing = [name for name in REQUIRED_FILES if not (app / name).is_file()]
    if missing:
        raise ValueError(f"Deployment app files missing: {missing}")
    frozen = read(study / "run_config.json")
    config = tomllib.loads((app / "pyproject.toml").read_text(encoding="utf-8"))
    project = config.get("project", {})
    flwr_app = config.get("tool", {}).get("flwr", {}).get("app", {})
    components = flwr_app.get("components", {})
    run = flwr_app.get("config", {})
    if "flwr>=1.42.0,<1.43.0" not in project.get("dependencies", []):
        raise ValueError("Deployment app must pin Flower to the frozen 1.42 minor range")
    if flwr_app.get("flwr-version-target") != "1.42.0":
        raise ValueError("Deployment app Flower target differs from frozen study")
    if components != {
        "serverapp": "paper2_dp_secagg_deploy.server_app:app",
        "clientapp": "paper2_dp_secagg_deploy.client_app:app",
    }:
        raise ValueError("Deployment components are not self-contained app modules")
    for key in ("num-server-rounds", "learning-rate", "features", "noise-multiplier", "clipping-norm",
                "central-dp", "differential-privacy-enabled", "outer-test-evaluation"):
        if run.get(key) != frozen.get(key):
            raise ValueError(f"Deployment app changes frozen parameter: {key}")
    source_text = "\n".join((app / name).read_text(encoding="utf-8") for name in REQUIRED_FILES if name.endswith(".py"))
    if "from paper2." in source_text or "import paper2." in source_text:
        raise ValueError("Deployment FAB must not import modules outside itself")
    return {
        "status": "PASS_AUTHENTICATED_DEPLOYMENT_ADAPTER_SEMANTIC_BINDING",
        "frozen_study_manifest_sha256": sha(study / "manifest.json"),
        "frozen_run_config_sha256": sha(study / "run_config.json"),
        "app_sources_sha256": {name: sha(app / name) for name in REQUIRED_FILES},
        "node_local_required_config": ["client-id", "data-root", "fold-id"],
        "data_paths_embedded_in_fab": False,
        "frozen_study_modified": False,
        "training_started": False,
        "claim_allowed": False,
        "interpretation": "Source-bound deployment adapter only; execution requires an independently reviewed, prospectively frozen deployment protocol.",
    }


def prepare(output: Path, app: Path = APP, study: Path = STUDY) -> dict:
    output = Path(output).resolve()
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    report = validate(app, study)
    output.mkdir(parents=True)
    atomic_json(output / "adapter_binding.json", report)
    atomic_json(output / "manifest.json", {
        "status": report["status"], "adapter_binding_sha256": sha(output / "adapter_binding.json"),
        "frozen_study_modified": False, "training_started": False, "claim_allowed": False,
    })
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--app", type=Path, default=APP)
    parser.add_argument("--study", type=Path, default=STUDY)
    args = parser.parse_args()
    print(prepare(args.output, args.app, args.study))
