"""Run the frozen DP+SecAgg+ protocol as a CPU-only logical-client simulation.

This is a Windows-compatible research fallback for Flower's managed local
SuperLink.  It deliberately uses seven virtual clients in one process tree;
it is not a replacement for the authenticated multi-host deployment study.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from paper2.restart.freeze_dp_secagg_pilot import RESULTS, verify
from paper2.restart.validation_search import ROOT, atomic_json, read, sha

MODE = "local_logical_client_cpu_simulation"
TIMEOUT_SECONDS = 300


def worker(package: Path) -> None:
    """Execute the frozen app in Flower's in-process Simulation Engine."""
    verified = verify(package)
    if verified["status"] != "PASS_FROZEN_DP_SECAGG_PILOT":
        raise RuntimeError("Frozen DP+SecAgg+ pilot verification failed")
    import flwr
    from flwr.app import Context, RecordDict
    from flwr.app.user_config import UserConfig
    from flwr.simulation.app import _run_simulation
    from flwr.supercore.telemetry import EventType
    from paper2.secagg_pilot.dp_client_app import app as client_app
    from paper2.secagg_pilot.dp_server_app import app as server_app

    if flwr.__version__ != "1.42.0":
        raise RuntimeError(f"Expected Flower 1.42.0, found {flwr.__version__}")
    run_config = read(package / "run_config.json")
    roster = run_config["client-roster"].split(",")
    if len(roster) != 7 or len(set(roster)) != 7:
        raise ValueError("Frozen pilot must use seven distinct logical clients")
    context = Context(run_id=2026102701, node_id=0, node_config=UserConfig(),
                      state=RecordDict(), run_config=UserConfig(run_config))
    _run_simulation(
        num_supernodes=len(roster), server_app=server_app, client_app=client_app,
        backend_name="ray",
        backend_config={"init_args": {"include_dashboard": False},
                        "client_resources": {"num_cpus": 1, "num_gpus": 0}},
        server_app_context=context, exit_event=EventType.PYTHON_API_RUN_SIMULATION_LEAVE,
    )


def _write_completion(output: Path, completed: subprocess.CompletedProcess | None,
                      timed_out: subprocess.TimeoutExpired | None = None) -> dict:
    stdout = "" if completed is None else completed.stdout
    stderr = "" if completed is None else completed.stderr
    if timed_out is not None:
        stdout = timed_out.stdout or ""
        stderr = timed_out.stderr or ""
        if isinstance(stdout, bytes): stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes): stderr = stderr.decode(errors="replace")
    (output / "flower_stdout.log").write_text(stdout, encoding="utf-8")
    (output / "flower_stderr.log").write_text(stderr, encoding="utf-8")
    result = {
        "returncode": None if timed_out is not None else completed.returncode,
        "status": "TIMED_OUT_NO_AUTOMATIC_RETRY" if timed_out is not None else
                  ("COMPLETE" if completed.returncode == 0 else "FAILED_NO_AUTOMATIC_RETRY"),
        "execution_mode": MODE, "outer_test_inference": False,
        "raw_updates_exported": False, "authenticated_network": False,
        "timeout_seconds": TIMEOUT_SECONDS,
        "log_sha256": {name: sha(output / name) for name in ("flower_stdout.log", "flower_stderr.log")},
    }
    atomic_json(output / "completion.json", result)
    return result


def recover_interrupted(output: Path, reason: str) -> dict:
    """Close a persisted started-only receipt after an operator interruption."""
    output = Path(output).resolve()
    if not (output / "manifest.json").is_file() or (output / "completion.json").exists():
        raise ValueError("Only an incomplete local-simulation receipt can be recovered")
    (output / "flower_stdout.log").write_text("", encoding="utf-8")
    (output / "flower_stderr.log").write_text(reason, encoding="utf-8")
    result = {"returncode": None, "status": "ABORTED_BY_OPERATOR_NO_AUTOMATIC_RETRY",
              "execution_mode": MODE, "outer_test_inference": False,
              "raw_updates_exported": False, "authenticated_network": False,
              "log_sha256": {name: sha(output / name) for name in ("flower_stdout.log", "flower_stderr.log")}}
    atomic_json(output / "completion.json", result)
    return result


def launch(package: Path, output: Path, execute: bool = False) -> dict:
    package, output = Path(package).resolve(), Path(output).resolve()
    verified = verify(package)
    command = [sys.executable, "-m", "paper2.restart.run_dp_secagg_local_simulation",
               "--worker", "--package", str(package)]
    if not execute:
        return {"status": "DRY_RUN_NO_TRAINING", "verified": verified,
                "execution_mode": MODE, "command": command}
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Execution output must be a new Paper 2 results directory")
    output.mkdir(parents=True)
    atomic_json(output / "manifest.json", {
        "status": "EXECUTION_STARTED", "execution_mode": MODE,
        "protocol": str(package), "protocol_manifest_sha256": sha(package / "manifest.json"),
        "verified": verified, "outer_test_inference": False,
        "raw_updates_exported": False, "authenticated_network": False,
    })
    try:
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        return _write_completion(output, None, exc)
    return _write_completion(output, completed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--recover-interrupted", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.package.resolve())
    elif args.recover_interrupted:
        if args.output is None:
            parser.error("--output is required with --recover-interrupted")
        print(json.dumps(recover_interrupted(args.output, "Interrupted after Windows Ray startup hang; no training result."), indent=2))
    elif args.output is None:
        parser.error("--output is required unless --worker is used")
    else:
        print(json.dumps(launch(args.package, args.output, args.execute), indent=2))
