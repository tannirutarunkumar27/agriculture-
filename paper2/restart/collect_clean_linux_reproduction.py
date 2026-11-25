"""Collect a fail-closed candidate receipt from a clean Linux Paper 2 host.

The collector performs only local evidence collection.  It never creates the
``CLEAN_LINUX_REPRODUCTION_VERIFIED`` release receipt: a separately recorded
independent audit and a retrieved-result receipt remain required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

from paper2.restart.release_reproducibility_gate import validate_linux_lock
from paper2.restart.validation_search import ROOT, atomic_json, sha

RESULTS = ROOT / "paper2/results"


def _run(command: list[str], cwd: Path, transcript_path: Path) -> dict:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=1800)
    transcript = (completed.stdout or "") + ("\n" if completed.stdout and completed.stderr else "") + (completed.stderr or "")
    transcript_path.write_text(transcript, encoding="utf-8")
    return {"command": command, "returncode": completed.returncode,
            "transcript_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
            "transcript_file": transcript_path.name,
            "status": "PASS" if completed.returncode == 0 else "FAIL"}


def collect(bundle_manifest: Path, linux_lock: Path, output: Path, *, run_tests: bool = False) -> dict:
    """Run bounded Linux checks and persist a candidate, never a release approval."""
    bundle_manifest, linux_lock, output = map(lambda value: Path(value).resolve(), (bundle_manifest, linux_lock, output))
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    if not bundle_manifest.is_file():
        raise ValueError("Bundle manifest is required")
    lock = validate_linux_lock(linux_lock)
    output.mkdir(parents=True)
    checks = {
        "linux_platform": "PASS" if platform.system() == "Linux" else "FAIL",
        "linux_lock": lock["status"],
        "environment_install": "NOT_RUN",
        "unit_tests": "NOT_RUN",
        "bundle_integrity": "NOT_RUN",
        "recovery": "NOT_RUN",
        "result_retrieval": "NOT_RUN_REQUIRES_SEPARATE_SSH_RETRIEVAL_RECEIPT",
    }
    transcripts: dict[str, dict] = {}
    if checks["linux_platform"] == "PASS" and lock["verified"]:
        # The remote bundle is intentionally a minimal execution payload, not
        # a standalone copy of every historical test fixture.  Test the clean
        # source checkout and verify the bundle independently by its manifest.
        root = ROOT
        transcript_dir = output / "transcripts"
        transcript_dir.mkdir()
        commands = {
            "environment_install": [sys.executable, "-m", "pip", "check"],
            "bundle_integrity": [sys.executable, "-B", "-m", "paper2.restart.verify_remote_bundle", "--manifest", str(bundle_manifest)],
            "recovery": [sys.executable, "-m", "unittest", "paper2.restart.test_outer_training", "paper2.restart.test_outer_runner"],
        }
        if run_tests:
            commands["unit_tests"] = [sys.executable, "-m", "unittest", "discover", "-s", "paper2/restart", "-p", "test_*.py"]
        for name, command in commands.items():
            result = _run(command, root, transcript_dir / f"{name}.log")
            transcripts[name] = result
            checks[name] = result["status"]
    status = "CLEAN_LINUX_REPRODUCTION_CANDIDATE_PENDING_INDEPENDENT_AUDIT"
    if checks["linux_platform"] != "PASS":
        status = "CLEAN_LINUX_REPRODUCTION_CANDIDATE_BLOCKED_NOT_LINUX"
    elif not lock["verified"]:
        status = "CLEAN_LINUX_REPRODUCTION_CANDIDATE_BLOCKED_LOCK_UNVERIFIED"
    receipt = {
        "schema_version": 1, "status": status, "verified": False,
        "platform": {"system": platform.system(), "release": platform.release(), "python": sys.version.split()[0]},
        "linux_lock_sha256": lock.get("sha256"), "bundle_manifest_sha256": sha(bundle_manifest),
        "checks": checks, "command_receipts": transcripts,
        "independent_audit": {"status": "PENDING", "reviewer": None},
        "qualification": "Candidate evidence only. Result retrieval and independent audit are not performed by this collector.",
    }
    atomic_json(output / "receipt_candidate.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--linux-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    print(json.dumps(collect(args.bundle_manifest, args.linux_lock, args.output, run_tests=args.run_tests), indent=2))
