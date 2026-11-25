"""Audit Paper 2 completion state without training or changing archived studies."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "paper2/results"


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def exists(relative: str) -> bool:
    return (ROOT / relative).exists()


def latest_result_file(prefix: str, filename: str, results: Path = RESULTS) -> Path:
    """Return the newest immutable versioned result that contains ``filename``."""
    expression = re.compile(rf"^{re.escape(prefix)}_v(\d+)$")
    candidates = []
    for directory in Path(results).iterdir():
        match = expression.fullmatch(directory.name)
        path = directory / filename
        if match and directory.is_dir() and path.is_file():
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No versioned result found for {prefix}/{filename}")
    return max(candidates, key=lambda item: item[0])[1]


def implementation_gaps(root: Path = ROOT) -> list[dict]:
    """Report code that is intentionally a gate/template rather than an executor."""
    checks = {
        "rr07_remote_gpu_runner": root / "paper2/restart/rr07_runner.py",
        "authenticated_network_orchestrator": root / "paper2/restart/authenticated_network_runner.py",
        "agronomic_guidance_study_runner": root / "paper2/restart/agronomic_guidance_runner.py",
        "restricted_ranking_executor": root / "paper2/restart/restricted_ranking_runner.py",
    }
    gaps = []
    for name, path in checks.items():
        if not path.is_file():
            gaps.append({"component": name, "status": "NOT_IMPLEMENTED", "path": str(path)})
            continue
        text = path.read_text(encoding="utf-8")
        if "PENDING_REMOTE_GPU_RUNNER" in text or "intentionally not enabled" in text:
            gaps.append({"component": name, "status": "SCAFFOLD_ONLY", "path": str(path)})
    return gaps


def build_report(pending: dict, privacy: dict, network: dict, agronomic: dict, ranking: dict,
                 release: dict | None = None, domain_review: dict | None = None,
                 reference_gate: dict | None = None, artifacts: dict[str, str] | None = None,
                 gaps: list[dict] | None = None, cpu_rr06: dict | None = None,
                 cpu_rr07: dict | None = None, dp_execution: dict | None = None,
                 dp_pilot_audit: dict | None = None,
                 dp_study_audit: dict | None = None,
                 local_wsl_smoke: dict | None = None,
                 crossref_evidence: dict | None = None,
                 deployment_adapter: dict | None = None,
                 local_authenticated_deployment_audit: dict | None = None,
                 authenticated_runtime_privacy_accounting: dict | None = None) -> dict:
    release = release or {"status": "BLOCKED_RELEASE_GATE_NOT_RUN", "release_allowed": False, "blocked": []}
    domain_review = domain_review or {"status": "AGRONOMIC_DOMAIN_REVIEW_NOT_RUN", "rules_pending": 0}
    reference_gate = reference_gate or {
        "status": "RANKING_REFERENCE_SUPPLEMENT_NOT_RUN",
        "source_missing_experiments": 0,
        "approved_usable_reference_supplements": 0,
        "unresolved_experiments": [],
    }
    artifacts = artifacts or {
        "matched_neural_protocol": "restart_matched_neural_protocol_v7",
        "matched_neural_execution": "restart_matched_neural_execution_v10",
        "dp_pilot_protocol": "restart_dp_secagg_pilot_protocol_v8",
        "dp_study_protocol": "restart_dp_secagg_study_protocol_v2",
        "dp_pilot_audit": "not_available",
        "dp_study_audit": "not_available",
        "dp_privacy_review_packet": "not_available",
        "ranking_review": "restart_ranking_experiment_review_v2",
        "ranking_combined_review_packet": "not_available",
        "agronomic_review_packet": "not_available",
        "agronomic_review_submission_kit": "not_available",
        "ranking_reference_gate": "restart_ranking_reference_gate_v2",
        "ranking_gate": "restart_ranking_approval_gate_v3",
        "release_gate": "restart_release_reproducibility_gate_v12",
        "dependency_license_review_submission_kit": "not_available",
        "dataset_rights_review_submission_kit": "not_available",
        "project_code_license_decision_kit": "not_available",
        "local_wsl_network_smoke": "not_available",
        "free_crossref_candidate_evidence": "not_available",
        "authenticated_dp_secagg_adapter": "not_available",
        "local_authenticated_dp_secagg_deployment_audit": "not_available",
        "authenticated_runtime_privacy_accounting": "not_available",
    }
    gaps = gaps or []
    cpu_rr06_complete = bool(cpu_rr06 and cpu_rr06.get("status") == "PASS_COMPLETE_MATCHED_NEURAL_STUDY"
                             and cpu_rr06.get("verified") == 504 and cpu_rr06.get("device") == "cpu")
    cpu_rr07_complete = bool(cpu_rr07 and cpu_rr07.get("status") == "PASS_COMPLETE_RR07_DEVELOPMENT_OUTER_STUDY"
                             and cpu_rr07.get("verified") == 160 and cpu_rr07.get("device") == "cpu")
    cpu_rr06_summary = ({key: cpu_rr06.get(key) for key in ("status", "verified", "pending", "device", "hardware_qualification")}
                         if cpu_rr06_complete else None)
    cpu_rr07_summary = ({key: cpu_rr07.get(key) for key in ("status", "verified", "planned", "device", "hardware_qualification")}
                         if cpu_rr07_complete else None)
    dp_execution_summary = ({key: dp_execution.get(key) for key in ("status", "returncode", "execution_mode", "timeout_seconds")}
                            if dp_execution else None)
    dp_execution_complete = bool(dp_execution and dp_execution.get("status") == "COMPLETE" and dp_execution.get("returncode") == 0)
    dp_pilot_complete = bool(
        dp_pilot_audit
        and dp_pilot_audit.get("status") == "PASS_EXECUTED_DP_SECAGG_PILOT_NOT_CERTIFIED"
        and dp_pilot_audit.get("rounds") == 1
        and dp_pilot_audit.get("semantic_log_audit", {}).get("passed") is True
    )
    dp_study_complete = bool(
        dp_study_audit
        and dp_study_audit.get("status") == "PASS_EXECUTED_DP_SECAGG_STUDY_NOT_CERTIFIED"
        and dp_study_audit.get("rounds") == 20
        and dp_study_audit.get("semantic_log_audit", {}).get("passed") is True
    )
    dp_mechanism_complete = dp_pilot_complete and dp_study_complete
    local_runtime_accounting_complete = bool(
        authenticated_runtime_privacy_accounting
        and authenticated_runtime_privacy_accounting.get("status")
        == "LOCAL_AUTHENTICATED_RUNTIME_PRIVACY_ACCOUNTING_READY_PENDING_INDEPENDENT_REVIEW"
        and authenticated_runtime_privacy_accounting.get("privacy_claim_enabled") is False
    )
    dp_audit_summary = {
        "pilot": ({key: dp_pilot_audit.get(key) for key in ("status", "rounds", "semantic_log_audit")}
                  if dp_pilot_audit else None),
        "study": ({key: dp_study_audit.get(key) for key in ("status", "rounds", "semantic_log_audit")}
                  if dp_study_audit else None),
    }
    secure_aggregation_status = (
        "COMPLETE_AUDITED_WSL_LOGICAL_CLIENTS_NOT_CERTIFIED" if dp_mechanism_complete
        else "LOCAL_CPU_EXECUTION_COMPLETE_AUDIT_REQUIRED" if dp_execution_complete
        else "LOCAL_CPU_EXECUTION_TERMINAL_FAILURE" if dp_execution
        else "IMPLEMENTED_FROZEN_NOT_EXECUTED"
    )
    end_to_end_dp_status = (
        "LOCAL_AUTHENTICATED_RUNTIME_ACCOUNTED_PENDING_INDEPENDENT_SECURITY_REVIEW_AND_EXTERNAL_DEPLOYMENT"
        if local_runtime_accounting_complete else
        "BLOCKED_APPLICATION_PRIVACY_ACCOUNTING_AND_INDEPENDENT_SECURITY_REVIEW"
        if dp_mechanism_complete
        else "BLOCKED_RELEASE_AUDIT_AND_REMOTE_DEPLOYMENT_REQUIRED" if dp_execution_complete
        else "BLOCKED_LOCAL_CPU_EXECUTION_FAILURE" if dp_execution
        else "BLOCKED_EXECUTION_AND_RELEASE_AUDIT_REQUIRED"
    )
    gap_names = [gap["component"] for gap in gaps]
    checks = {
        "data_and_split_validation": {"status": "COMPLETE", "evidence": "DC01"},
        "inner_search": {"status": "COMPLETE", "evidence": "R12H"},
        "unnoised_refit": {"status": "COMPLETE", "evidence": "R13"},
        "noise_simulation": {"status": "COMPLETE_WITH_LIMITS", "evidence": "R14"},
        "feature_timing": {"status": "COMPLETE_WITH_PRE_SOWING_DISABLED", "evidence": "restart_feature_timing_audit_v1"},
        "matched_neural_search": {"status": "COMPLETE_AUDITED_CPU" if cpu_rr06_complete else "PENDING_MATCHED_NEURAL_EXECUTION", "planned_runs": 504,
                                  "portable_protocol": artifacts["matched_neural_protocol"],
                                  "portable_execution": artifacts["matched_neural_execution"],
                                  "archived_cpu_v7_verified": pending["verified"],
                                  "archived_cpu_v7_pending": pending["pending"],
                                  "cpu_audit": cpu_rr06_summary,
                                  "gpu_replication_required": False,
                                  "gpu_replication_optional": cpu_rr06_complete,
                                  "reason": (
                                      "The frozen scientific protocol is compute-device agnostic. A complete, "
                                      "independently audited CPU run satisfies RR06; GPU replication is optional "
                                      "performance/portability evidence and is not a completion blocker."
                                  )},
        "rr07_outer_evaluation": {
            "status": "COMPLETE_AUDITED_CPU" if cpu_rr07_complete else (
                "IMPLEMENTATION_MISSING_RR07_RUNNER_AND_RR06_SELECTION"
                if "rr07_remote_gpu_runner" in gap_names else "BLOCKED_PENDING_RR06_SELECTION"),
            "planned_runs": 160, "claim_allowed": False,
            "gpu_replication_required": False,
            "gpu_replication_optional": cpu_rr07_complete,
            "cpu_audit": cpu_rr07_summary,
        },
        "privacy_claim": {"status": privacy["status"], "enabled": privacy["privacy_claim_enabled"]},
        "secure_aggregation": {
            "status": secure_aggregation_status,
            "claim_allowed": False, "execution_receipt": dp_execution_summary,
            "audited_execution": dp_audit_summary,
            "evidence": [artifacts["dp_pilot_protocol"], artifacts["dp_study_protocol"],
                         artifacts["dp_pilot_audit"], artifacts["dp_study_audit"],
                         artifacts["dp_privacy_review_packet"]],
        },
        "end_to_end_dp_execution": {
            "status": end_to_end_dp_status,
            "claim_allowed": False, "execution_receipt": dp_execution_summary,
            "mechanism_execution": dp_audit_summary,
            "local_authenticated_runtime_accounting": {
                "artifact": artifacts.get("authenticated_runtime_privacy_accounting", "not_available"),
                "status": authenticated_runtime_privacy_accounting.get("status") if authenticated_runtime_privacy_accounting else "not_available",
                "conditional_budget": authenticated_runtime_privacy_accounting.get("conditional_zcdp_budget") if authenticated_runtime_privacy_accounting else None,
                "counts_as_end_to_end_privacy_certificate": False,
            },
        },
        "external_linux_deployment": {"status": network["status"],
                                      "deployment_verified": network["deployment_verified"],
                                      "claim_allowed": False, "missing": network.get("missing", []),
                                      "local_wsl_smoke": {
                                          "artifact": artifacts.get("local_wsl_network_smoke", "not_available"),
                                          "status": local_wsl_smoke.get("status") if local_wsl_smoke else "not_available",
                                          "online_nodes_observed": local_wsl_smoke.get("online_nodes_observed_from_listing") if local_wsl_smoke else 0,
                                          "counts_as_external_deployment": False,
                                      },
                                      "deployment_adapter": {
                                          "artifact": artifacts.get("authenticated_dp_secagg_adapter", "not_available"),
                                          "status": deployment_adapter.get("status") if deployment_adapter else "not_available",
                                          "semantic_binding_passed": bool(deployment_adapter and deployment_adapter.get("status") == "PASS_AUTHENTICATED_DEPLOYMENT_ADAPTER_SEMANTIC_BINDING"),
                                          "counts_as_execution": False,
                                      },
                                      "local_authenticated_deployment_audit": {
                                          "artifact": artifacts.get("local_authenticated_dp_secagg_deployment_audit", "not_available"),
                                          "status": local_authenticated_deployment_audit.get("status") if local_authenticated_deployment_audit else "not_available",
                                          "raw_success_overturned": bool(local_authenticated_deployment_audit and local_authenticated_deployment_audit.get("raw_success_claim_overturned")),
                                          "counts_as_external_deployment": False,
                                      }},
        "independent_agronomic_rules": {"status": domain_review["status"],
                                        "candidate_registry_status": agronomic["status"],
                                        "schema_valid_candidate_rules": agronomic["rules_schema_valid"],
                                        "domain_approved_declared": agronomic.get("rules_domain_approved_declared", 0),
                                        "domain_review_pending": domain_review.get("rules_pending", 0),
                                        "review_packet": artifacts["agronomic_review_packet"],
                                        "review_submission_kit": artifacts["agronomic_review_submission_kit"],
                                        "claim_allowed": False},
        "crop_system_ranking": {"status": ranking["status"],
                                "reviewed_blocks": ranking["blocks_total"] - ranking["blocks_pending"],
                                "pending_blocks": ranking["blocks_pending"],
                                "reference_supplement_status": reference_gate["status"],
                                "source_missing_experiments": reference_gate.get("source_missing_experiments", 0),
                                "approved_reference_supplements": reference_gate.get(
                                    "approved_usable_reference_supplements", 0
                                ),
                                "unresolved_reference_experiments": reference_gate.get(
                                    "unresolved_experiments", []
                                ),
                                "free_crossref_metadata_retrieved": crossref_evidence.get("crossref_retrieved", 0)
                                if crossref_evidence else 0,
                                "claim_allowed": ranking["ranking_enabled"],
                                "implementation_status": "IMPLEMENTATION_MISSING_RESTRICTED_RANKING_EXECUTOR"
                                if "restricted_ranking_executor" in gap_names else "IMPLEMENTED",
                                "evidence": ["restart_ranking_covariates_v2", artifacts["ranking_review"],
                                             artifacts["ranking_combined_review_packet"],
                                             artifacts["ranking_reference_gate"], artifacts["ranking_gate"],
                                             artifacts.get("free_crossref_candidate_evidence", "not_available")]},
        "publication_and_clean_reproduction": {
            "status": release["status"],
            "claim_allowed": release.get("release_allowed", False),
            "blocked": release.get("blocked", []),
            "evidence": [artifacts["release_gate"], artifacts["dependency_license_review_submission_kit"],
                         artifacts["dataset_rights_review_submission_kit"], artifacts["project_code_license_decision_kit"]],
        },
    }
    blockers = list(dict.fromkeys(
        name for name, item in checks.items()
        if item["status"].startswith(("PENDING", "BLOCKED", "NOT_", "READY_", "IMPLEMENTATION_MISSING"))
        or (item.get("claim_allowed") is False and not item["status"].startswith("COMPLETE_AUDITED"))
        or (name == "privacy_claim" and not privacy.get("privacy_claim_enabled", False))
    ))
    blockers.extend(f"implementation:{gap['component']}" for gap in gaps)
    return {
        "status": "PAPER2_IMPLEMENTATION_AND_EXTERNAL_WORK_REMAIN" if blockers or gaps else "PAPER2_LOCAL_SCOPE_COMPLETE",
        "checks": checks,
        "blockers": blockers,
        "implementation_gaps": gaps,
        "training_performed": False,
        "outer_test_inference": False,
        "interpretation": "Current evidence supports retrospective federated/conditional regression only; it does not support crop recommendation or end-to-end privacy claims.",
    }


def audit(output: Path) -> dict:
    output = Path(output).resolve()
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    sources = {
        "matched_neural_pending": latest_result_file("restart_matched_neural_pending", "pending.json"),
        "privacy_gate": latest_result_file("restart_privacy_claim_gate", "privacy_gate.json"),
        "network_gate": latest_result_file("restart_network_deployment_gate", "deployment_gate.json"),
        "agronomic_rules": latest_result_file("restart_agronomic_rule_validation", "rule_validation.json"),
        "ranking_gate": latest_result_file("restart_ranking_approval_gate", "ranking_gate.json"),
        "release_gate": latest_result_file("restart_release_reproducibility_gate", "release_gate.json"),
        "domain_review": latest_result_file("restart_agronomic_domain_review_gate", "domain_review_gate.json"),
        "ranking_reference_gate": latest_result_file("restart_ranking_reference_gate", "reference_gate.json"),
    }
    artifacts = {
        "matched_neural_protocol": latest_result_file("restart_matched_neural_protocol", "manifest.json").parent.name,
        "matched_neural_execution": latest_result_file("restart_matched_neural_execution", "manifest.json").parent.name,
        "dp_pilot_protocol": latest_result_file("restart_dp_secagg_pilot_protocol", "manifest.json").parent.name,
        "dp_study_protocol": latest_result_file("restart_dp_secagg_study_protocol", "manifest.json").parent.name,
        "ranking_review": latest_result_file("restart_ranking_experiment_review", "manifest.json").parent.name,
        "ranking_reference_gate": sources["ranking_reference_gate"].parent.name,
        "ranking_gate": sources["ranking_gate"].parent.name,
        "release_gate": sources["release_gate"].parent.name,
    }
    try:
        agronomic_packet_path = latest_result_file(
            "restart_agronomic_review_packet", "review_packet.json"
        )
    except FileNotFoundError:
        agronomic_packet_path = None
    try:
        ranking_packet_path = latest_result_file(
            "restart_ranking_combined_review_packet", "review_packet.json"
        )
    except FileNotFoundError:
        ranking_packet_path = None
    artifacts["agronomic_review_packet"] = (
        agronomic_packet_path.parent.name if agronomic_packet_path else "not_available"
    )
    artifacts["ranking_combined_review_packet"] = (
        ranking_packet_path.parent.name if ranking_packet_path else "not_available"
    )
    try:
        agronomic_submission_path = latest_result_file(
            "restart_agronomic_review_submission", "submission_kit.json"
        )
    except FileNotFoundError:
        agronomic_submission_path = None
    artifacts["agronomic_review_submission_kit"] = (
        agronomic_submission_path.parent.name if agronomic_submission_path else "not_available"
    )
    try:
        license_submission_path = latest_result_file(
            "restart_dependency_license_review_submission", "license_review_packet.json"
        )
    except FileNotFoundError:
        license_submission_path = None
    artifacts["dependency_license_review_submission_kit"] = (
        license_submission_path.parent.name if license_submission_path else "not_available"
    )
    try:
        dataset_rights_path = latest_result_file(
            "restart_dataset_rights_review_submission", "dataset_rights_packet.json"
        )
    except FileNotFoundError:
        dataset_rights_path = None
    artifacts["dataset_rights_review_submission_kit"] = (
        dataset_rights_path.parent.name if dataset_rights_path else "not_available"
    )
    try:
        code_license_path = latest_result_file(
            "restart_project_code_license_decision", "project_code_license_packet.json"
        )
    except FileNotFoundError:
        code_license_path = None
    artifacts["project_code_license_decision_kit"] = (
        code_license_path.parent.name if code_license_path else "not_available"
    )
    try:
        dp_pilot_audit_path = latest_result_file("restart_dp_secagg_pilot_wsl_cpu_audit", "audit.json")
    except FileNotFoundError:
        dp_pilot_audit_path = None
    try:
        dp_study_audit_path = latest_result_file("restart_dp_secagg_study_wsl_cpu_audit", "audit.json")
    except FileNotFoundError:
        dp_study_audit_path = None
    artifacts["dp_pilot_audit"] = dp_pilot_audit_path.parent.name if dp_pilot_audit_path else "not_available"
    artifacts["dp_study_audit"] = dp_study_audit_path.parent.name if dp_study_audit_path else "not_available"
    try:
        dp_privacy_review_path = latest_result_file(
            "restart_dp_secagg_privacy_review_packet", "privacy_review_packet.json"
        )
    except FileNotFoundError:
        dp_privacy_review_path = None
    artifacts["dp_privacy_review_packet"] = (
        dp_privacy_review_path.parent.name if dp_privacy_review_path else "not_available"
    )
    try:
        local_wsl_smoke_path = latest_result_file("restart_local_wsl_authenticated_smoke", "receipt.json")
    except FileNotFoundError:
        local_wsl_smoke_path = None
    artifacts["local_wsl_network_smoke"] = (
        local_wsl_smoke_path.parent.name if local_wsl_smoke_path else "not_available"
    )
    try:
        crossref_evidence_path = latest_result_file(
            "restart_free_crossref_reference_evidence", "crossref_candidate_evidence.json"
        )
    except FileNotFoundError:
        crossref_evidence_path = None
    artifacts["free_crossref_candidate_evidence"] = (
        crossref_evidence_path.parent.name if crossref_evidence_path else "not_available"
    )
    try:
        deployment_adapter_path = latest_result_file(
            "restart_authenticated_dp_secagg_adapter", "adapter_binding.json"
        )
    except FileNotFoundError:
        deployment_adapter_path = None
    artifacts["authenticated_dp_secagg_adapter"] = (
        deployment_adapter_path.parent.name if deployment_adapter_path else "not_available"
    )
    try:
        local_authenticated_deployment_audit_path = latest_result_file(
            "restart_local_authenticated_dp_secagg_deployment_audit", "audit.json"
        )
    except FileNotFoundError:
        local_authenticated_deployment_audit_path = None
    artifacts["local_authenticated_dp_secagg_deployment_audit"] = (
        local_authenticated_deployment_audit_path.parent.name
        if local_authenticated_deployment_audit_path else "not_available"
    )
    try:
        authenticated_runtime_accounting_path = latest_result_file(
            "restart_authenticated_runtime_privacy_accounting", "runtime_privacy_accounting.json"
        )
    except FileNotFoundError:
        authenticated_runtime_accounting_path = None
    artifacts["authenticated_runtime_privacy_accounting"] = (
        authenticated_runtime_accounting_path.parent.name
        if authenticated_runtime_accounting_path else "not_available"
    )
    try:
        cpu_rr06_path = latest_result_file("restart_matched_neural_study_cpu_local_audit", "audit.json")
    except FileNotFoundError:
        cpu_rr06_path = None
    try:
        cpu_rr07_path = latest_result_file("restart_rr07_outer_cpu_local_audit", "audit.json")
    except FileNotFoundError:
        cpu_rr07_path = None
    try:
        dp_execution_path = latest_result_file("restart_dp_secagg_pilot_execution_cpu_local", "completion.json")
    except FileNotFoundError:
        dp_execution_path = None
    result = build_report(*(read(path) for path in sources.values()), artifacts=artifacts,
                          gaps=implementation_gaps(), cpu_rr06=read(cpu_rr06_path) if cpu_rr06_path else None,
                          cpu_rr07=read(cpu_rr07_path) if cpu_rr07_path else None,
                          dp_execution=read(dp_execution_path) if dp_execution_path else None,
                          dp_pilot_audit=read(dp_pilot_audit_path) if dp_pilot_audit_path else None,
                          dp_study_audit=read(dp_study_audit_path) if dp_study_audit_path else None,
                          local_wsl_smoke=read(local_wsl_smoke_path) if local_wsl_smoke_path else None,
                          crossref_evidence=read(crossref_evidence_path) if crossref_evidence_path else None,
                          deployment_adapter=read(deployment_adapter_path) if deployment_adapter_path else None,
                          local_authenticated_deployment_audit=(
                              read(local_authenticated_deployment_audit_path)
                              if local_authenticated_deployment_audit_path else None
                          ),
                          authenticated_runtime_privacy_accounting=(
                              read(authenticated_runtime_accounting_path)
                              if authenticated_runtime_accounting_path else None
                          ))
    output.mkdir(parents=True)
    atomic_json(output / "completion_audit.json", result)
    atomic_json(output / "manifest.json", {
        "status": result["status"], "code_sha256": sha(Path(__file__)),
        "evidence": {name: str(path.relative_to(ROOT)) for name, path in sources.items()},
        "evidence_sha256": {name: sha(path) for name, path in sources.items()},
        "training_performed": False,
    })
    return {"status": result["status"], "blockers": result["blockers"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(audit(args.output))
