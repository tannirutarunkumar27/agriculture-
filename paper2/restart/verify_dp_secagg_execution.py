"""Audit a completed Flower DP+SecAgg pilot without rerunning training."""
from __future__ import annotations

import argparse
from pathlib import Path

from paper2.restart.validation_search import ROOT, atomic_json, read, sha
from paper2.restart.run_dp_secagg_pilot import semantic_log_audit

RESULTS = ROOT / "paper2/results"


def inspect_execution(execution: Path, protocol: Path) -> dict:
    execution, protocol = Path(execution).resolve(), Path(protocol).resolve()
    manifest, completion = read(execution / "manifest.json"), read(execution / "completion.json")
    contract, run = read(protocol / "privacy_contract.json"), read(protocol / "run_config.json")
    errors = []
    if manifest.get("protocol_manifest_sha256") != sha(protocol / "manifest.json"): errors.append("protocol_binding")
    if completion.get("returncode") != 0 or completion.get("status") != "COMPLETE": errors.append("execution_failed")
    logs = completion.get("log_sha256", {})
    if set(logs) != {"flower_stdout.log", "flower_stderr.log"}: errors.append("required_logs")
    for name, digest in logs.items():
        if sha(execution / name) != digest: errors.append(f"log_hash:{name}")
    if run.get("differential-privacy-enabled") is not True or run.get("central-dp") is not True: errors.append("dp_disabled")
    if contract.get("secure_aggregation_requested") is not True or contract.get("differential_privacy_enabled") is not True:
        errors.append("secagg_or_contract_dp_disabled")
    rounds = run.get("num-server-rounds")
    stdout = (execution / "flower_stdout.log").read_text(encoding="utf-8") if (execution / "flower_stdout.log").is_file() else ""
    stderr = (execution / "flower_stderr.log").read_text(encoding="utf-8") if (execution / "flower_stderr.log").is_file() else ""
    semantic = semantic_log_audit(stdout, stderr, rounds if isinstance(rounds, int) else 0)
    if not semantic["passed"]: errors.append("semantic_log_failure")
    if completion.get("semantic_log_audit") != semantic: errors.append("semantic_receipt_mismatch")
    budget = contract.get("conditional_budget", contract.get("one_round_conditional_budget"))
    if not isinstance(rounds, int) or rounds < 1: errors.append("invalid_round_count")
    if contract.get("adaptive_releases") != rounds or not isinstance(budget, dict) or budget.get("releases") != rounds:
        errors.append("privacy_release_count")
    if run.get("outer-test-evaluation") is not False: errors.append("outer_test_policy")
    if manifest.get("raw_updates_exported") is not False or completion.get("raw_updates_exported") is not False: errors.append("raw_update_policy")
    mode = completion.get("execution_mode", manifest.get("execution_mode", "flower_cli"))
    if mode not in {"flower_cli", "local_logical_client_cpu_simulation"}: errors.append("execution_mode")
    kind = "PILOT" if rounds == 1 else "STUDY"
    status = f"PASS_EXECUTED_DP_SECAGG_{kind}_NOT_CERTIFIED" if not errors else "FAIL_DP_SECAGG_EXECUTION_AUDIT"
    report = {"status": status, "errors": errors, "rounds": rounds, "conditional_budget": budget,
              "semantic_log_audit": semantic,
              "application_privacy_certified": False, "production_deployment_verified": False,
              "execution_mode": mode,
              "qualification": f"Execution evidence for a {rounds}-round logical-client {kind.lower()}; not an end-to-end application privacy or remote-farm certification."}
    if errors: raise ValueError("DP+SecAgg audit failed: " + ",".join(errors))
    return report


def verify(execution: Path, protocol: Path, output: Path) -> dict:
    execution, protocol, output = map(lambda p: Path(p).resolve(), (execution, protocol, output))
    if output.exists() or RESULTS not in output.parents or output == execution:
        raise ValueError("Audit output must be a new separate Paper 2 results directory")
    report = inspect_execution(execution, protocol)
    output.mkdir(parents=True); atomic_json(output / "audit.json", report)
    atomic_json(output / "manifest.json", {"status": report["status"], "execution_sha256": sha(execution / "completion.json"), "source_sha256": sha(Path(__file__))})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, required=True); parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    print(verify(args.execution, args.protocol, args.output))
