"""Independently audit a local authenticated Flower deployment receipt.

The runner's process exit status is insufficient evidence: ``flwr run`` can
return after the SuperLink has accepted a run even when the ServerApp exits
unsuccessfully.  This tool verifies recorded log hashes and searches the
immutable runtime transcript for Flower failure signals.  It never changes the
source receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "paper2/results"

FAILURE_SIGNATURES = {
    "serverapp_exception": r"ServerApp raised an exception",
    "clientapp_exception": r"ClientApp raised an exception",
    "runtime_dependency_installation": r"RuntimeDependencyInstallationError",
    "runtime_dependency_failure": r"Runtime installation of app dependencies failed",
    "flower_nonzero_exit": r"ERROR\s*:\s*Exit Code:\s*[1-9]",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_receipt(receipt: dict, logs: Path) -> dict:
    """Return reproducible semantic findings for a recorded deployment."""
    expected_hashes = receipt.get("log_sha256", {})
    hash_mismatches = {}
    missing_logs = []
    for name, expected in expected_hashes.items():
        path = logs / name
        if not path.is_file():
            missing_logs.append(name)
        elif sha256(path) != expected:
            hash_mismatches[name] = {"expected": expected, "actual": sha256(path)}
    transcript = "\n".join(
        (path.read_text(encoding="utf-8", errors="replace"))
        for path in (logs / "run_stdout.log", logs / "run_stderr.log")
        if path.is_file()
    )
    matched = [name for name, pattern in FAILURE_SIGNATURES.items() if re.search(pattern, transcript)]
    rounds_requested = int(receipt.get("rounds_requested", 0))
    expected_rounds = set(range(1, rounds_requested + 1))
    rounds_by_node = {}
    missing_rounds_by_node = {}
    for index in range(7):
        path = logs / f"supernode-{index}.log"
        if not path.is_file():
            missing_rounds_by_node[str(index)] = list(range(1, rounds_requested + 1))
            continue
        observed = {
            int(value)
            for value in re.findall(r"\[RUN\s+\d+,\s+ROUND\s+(\d+)\]", path.read_text(encoding="utf-8", errors="replace"))
        }
        rounds_by_node[str(index)] = sorted(observed)
        absent = sorted(expected_rounds - observed)
        if absent:
            missing_rounds_by_node[str(index)] = absent
    round_evidence_passed = rounds_requested > 0 and not missing_rounds_by_node
    semantic_passed = (
        receipt.get("run_returncode") == 0
        and not matched
        and not missing_logs
        and not hash_mismatches
        and receipt.get("online_nodes_before_run") == receipt.get("unique_node_ids_before_run") == 7
        and receipt.get("clientapp_isolation") == "subprocess"
        and round_evidence_passed
    )
    return {
        "semantic_passed": semantic_passed,
        "raw_receipt_status": receipt.get("status"),
        "run_returncode": receipt.get("run_returncode"),
        "matched_failure_signatures": matched,
        "missing_logs": missing_logs,
        "hash_mismatches": hash_mismatches,
        "protocol_evidence": {
            "rounds_requested": rounds_requested,
            "rounds_by_node": rounds_by_node,
            "missing_rounds_by_node": missing_rounds_by_node,
            "all_seven_nodes_reached_every_requested_round": round_evidence_passed,
            "clientapp_isolation": receipt.get("clientapp_isolation"),
        },
        "cli_return_code_is_sufficient": False,
    }


def run(source: Path, output: Path) -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new directory under paper2/results")
    receipt_path = source / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    findings = audit_receipt(receipt, source / "sanitized_logs")
    raw_success_claim = str(receipt.get("status", "")).startswith("PASS_")
    semantic_failure = not findings["semantic_passed"]
    result = {
        "status": (
            "FAIL_LOCAL_AUTHENTICATED_DP_SECAGG_DEPLOYMENT_SEMANTIC_AUDIT"
            if semantic_failure else "PASS_LOCAL_AUTHENTICATED_DP_SECAGG_DEPLOYMENT_SEMANTIC_AUDIT_NOT_CERTIFIED"
        ),
        "source_receipt": str(receipt_path.relative_to(ROOT)),
        "source_receipt_sha256": sha256(receipt_path),
        "raw_success_claim_overturned": raw_success_claim and semantic_failure,
        "findings": findings,
        "counts_as_external_linux_deployment": False,
        "privacy_claim_allowed": False,
        "interpretation": (
            "The raw CLI status is overturned: Flower recorded a ServerApp/runtime failure. "
            "No DP, SecAgg, or deployment-completion claim may use this execution."
            if semantic_failure else
            "This local loopback audit passed semantic checks but remains neither an external multi-host deployment nor a privacy certification."
        ),
    }
    output.mkdir(parents=True)
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        "status": result["status"],
        "code_sha256": sha256(Path(__file__)),
        "source_receipt_sha256": result["source_receipt_sha256"],
    }, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output), indent=2))
