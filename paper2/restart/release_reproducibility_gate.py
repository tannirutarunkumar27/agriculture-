"""Audit portable bundles, clean-environment evidence, and release licensing.

The gate is intentionally conservative. It never treats a DOI, an accessible
download, or a working Windows environment as redistribution permission or a
clean Linux reproduction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

# Support direct execution from the repository root as well as module mode.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper2.restart.paper2_completion_audit import ROOT, RESULTS, atomic_json, latest_result_file, read, sha

LICENSING = ROOT / "paper2/configs/release_licensing_registry_v1.json"
RUNTIME = ROOT / "paper2/requirements-runtime-observed.txt"
LINUX_LOCK = ROOT / "paper2/requirements-linux-lock.txt"
CLEAN_LINUX_RECEIPT = RESULTS / "restart_clean_linux_reproduction_v1/receipt.json"
PROPOSED_LINUX_LOCK = ROOT / "paper2/requirements-linux-lock-proposed.txt"


def _locked_names(text: str) -> set[str]:
    names = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "--")):
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)==[^\s]+", line)
        if match:
            names.add(match.group(1).lower().replace("_", "-"))
    return names


def _requirement_blocks(text: str) -> list[str]:
    """Return logical pip requirements, including backslash continuations."""
    blocks: list[str] = []
    current: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("--") and not current:
            continue
        current.append(stripped)
        if not stripped.endswith("\\"):
            blocks.append(" ".join(current))
            current = []
    if current:
        blocks.append(" ".join(current))
    return blocks


def validate_linux_lock(path: Path = LINUX_LOCK) -> dict:
    """Reject marker-only or incomplete files masquerading as a Linux lock."""
    path = Path(path)
    errors = []
    if not path.is_file():
        return {"status": "BLOCKED_NO_LINUX_LOCK", "verified": False, "errors": ["missing"]}
    text = path.read_text(encoding="utf-8")
    lowered = text.lower()
    status_match = re.search(r"(?mi)^#\s*status=(remote_verified|local_wsl_verified)\s*$", text)
    if not status_match:
        errors.append("missing_exact_verified_status")
    if not re.search(r"(?mi)^#\s*platform=linux\s*$", text):
        errors.append("missing_linux_platform")
    required = _locked_names(PROPOSED_LINUX_LOCK.read_text(encoding="utf-8"))
    found = _locked_names(text)
    missing = sorted(required - found)
    if missing:
        errors.append("missing_required_packages:" + ",".join(missing))
    requirement_blocks = _requirement_blocks(text)
    if not requirement_blocks or any("==" not in block for block in requirement_blocks):
        errors.append("unversioned_requirement")
    if any("--hash=sha256:" not in block.lower() for block in requirement_blocks):
        errors.append("missing_wheel_hash")
    return {
        "status": "PASS" if not errors else "BLOCKED_INVALID_LINUX_LOCK",
        "verified": not errors,
        "errors": errors,
        "sha256": sha(path),
        "package_count": len(found),
        "verification_scope": status_match.group(1).upper() if status_match else None,
    }


def validate_clean_linux_receipt(path: Path, expected_lock_sha256: str | None) -> dict:
    """Validate substantive Linux evidence rather than accepting file existence."""
    path = Path(path)
    if not path.is_file():
        return {"status": "BLOCKED_NO_CLEAN_LINUX_RECEIPT", "verified": False, "errors": ["missing"]}
    try:
        receipt = read(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {"status": "BLOCKED_INVALID_CLEAN_LINUX_RECEIPT", "verified": False,
                "errors": [f"invalid_json:{type(exc).__name__}"]}
    errors = []
    if receipt.get("schema_version") != 1:
        errors.append("schema_version")
    if receipt.get("status") != "CLEAN_LINUX_REPRODUCTION_VERIFIED" or receipt.get("verified") is not True:
        errors.append("verification_status")
    if receipt.get("platform", {}).get("system") != "Linux":
        errors.append("platform")
    if not expected_lock_sha256 or receipt.get("linux_lock_sha256") != expected_lock_sha256:
        errors.append("linux_lock_sha256")
    checks = receipt.get("checks", {})
    for name in ("environment_install", "unit_tests", "bundle_integrity", "recovery", "result_retrieval"):
        if checks.get(name) != "PASS":
            errors.append(f"check:{name}")
    if receipt.get("independent_audit", {}).get("status") != "APPROVED":
        errors.append("independent_audit")
    return {
        "status": "PASS" if not errors else "BLOCKED_INVALID_CLEAN_LINUX_RECEIPT",
        "verified": not errors,
        "errors": errors,
        "sha256": sha(path),
    }


def validate_clean_linux_candidate(
    path: Path,
    expected_lock_sha256: str | None,
    expected_bundle_manifest_sha256: str | None = None,
) -> dict:
    """Validate successful local evidence without confusing it with release approval."""
    path = Path(path)
    if not path.is_file():
        return {"status": "BLOCKED_NO_CLEAN_LINUX_CANDIDATE", "evidence_verified": False, "errors": ["missing"]}
    try:
        receipt = read(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {"status": "BLOCKED_INVALID_CLEAN_LINUX_CANDIDATE", "evidence_verified": False,
                "errors": [f"invalid_json:{type(exc).__name__}"]}
    errors = []
    if receipt.get("schema_version") != 1 or receipt.get("verified") is not False:
        errors.append("schema_or_candidate_status")
    if receipt.get("status") != "CLEAN_LINUX_REPRODUCTION_CANDIDATE_PENDING_INDEPENDENT_AUDIT":
        errors.append("candidate_status")
    if receipt.get("platform", {}).get("system") != "Linux":
        errors.append("platform")
    if not expected_lock_sha256 or receipt.get("linux_lock_sha256") != expected_lock_sha256:
        errors.append("linux_lock_sha256")
    if (expected_bundle_manifest_sha256 is not None
            and receipt.get("bundle_manifest_sha256") != expected_bundle_manifest_sha256):
        errors.append("bundle_manifest_sha256")
    checks = receipt.get("checks", {})
    for name in ("environment_install", "unit_tests", "bundle_integrity", "recovery"):
        if checks.get(name) != "PASS":
            errors.append(f"check:{name}")
    receipt_dir = path.parent
    for name, command in receipt.get("command_receipts", {}).items():
        transcript_name = command.get("transcript_file")
        transcript = receipt_dir / "transcripts" / str(transcript_name)
        if command.get("status") != "PASS" or not transcript_name or not transcript.is_file():
            errors.append(f"transcript:{name}")
        elif hashlib.sha256(transcript.read_bytes()).hexdigest() != command.get("transcript_sha256"):
            errors.append(f"transcript_hash:{name}")
    return {
        "status": "PASS_CLEAN_LINUX_CANDIDATE_EVIDENCE" if not errors else "BLOCKED_INVALID_CLEAN_LINUX_CANDIDATE",
        "evidence_verified": not errors,
        "errors": errors,
        "sha256": sha(path),
        "release_approval": False,
    }


def verify_bundle(manifest_path: Path) -> dict:
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest = read(manifest_path)
    allowed = {"RR06_REMOTE_BUNDLE_READY_NO_TRAINING", "DP_SECAGG_REMOTE_BUNDLE_READY_NO_TRAINING"}
    if manifest.get("status") not in allowed or not isinstance(manifest.get("files"), dict):
        raise ValueError(f"Unexpected bundle manifest: {manifest_path}")
    errors = []
    total = 0
    for relative, expected in manifest["files"].items():
        relative_path = Path(relative)
        path = (root / relative_path).resolve()
        if relative_path.is_absolute() or root not in path.parents:
            errors.append(f"unsafe_path:{relative}")
        elif not path.is_file():
            errors.append(f"missing:{relative}")
        elif sha(path) != expected:
            errors.append(f"hash:{relative}")
        else:
            total += path.stat().st_size
    if total != manifest.get("bytes"):
        errors.append("byte_count")
    return {
        "status": "PASS" if not errors else "FAIL",
        "bundle_status": manifest.get("status"),
        "verified_files": len(manifest.get("files", {})) - len(errors),
        "declared_files": len(manifest.get("files", {})),
        "errors": errors,
    }


def evaluate(
    rr06: dict,
    dp: dict,
    licensing: dict,
    runtime_text: str,
    completion: dict,
    network: dict,
    ranking: dict,
    agronomic: dict,
    domain_review: dict | None = None,
    reference_gate: dict | None = None,
    dependency_inventory: dict | None = None,
    *,
    project_license_present: bool,
    linux_lock_present: bool,
    clean_linux_receipt_present: bool,
) -> dict:
    domain_review = domain_review or {"status": "AGRONOMIC_DOMAIN_REVIEW_NOT_RUN", "guidance_enabled": False}
    reference_gate = reference_gate or {
        "status": "RANKING_REFERENCE_SUPPLEMENT_NOT_RUN",
        "ranking_reference_supplement_enabled": False,
    }
    dependency_inventory = dependency_inventory or {"license_inventory_complete": False}
    required_license_sections = {"dataset", "project_code", "derived_artifacts", "third_party_dependencies"}
    if not required_license_sections <= set(licensing):
        raise ValueError("Licensing registry is incomplete")
    observed_only = "not a cross-platform lock" in runtime_text.lower()
    dataset = licensing["dataset"]
    code = licensing["project_code"]
    third_party = licensing["third_party_dependencies"]
    matched_status = completion.get("checks", {}).get("matched_neural_search", {}).get("status")
    accepted_matched_statuses = {"COMPLETE", "COMPLETE_AUDITED_CPU", "COMPLETE_AUDITED_GPU"}
    checks = {
        "rr06_remote_bundle_integrity": rr06["status"],
        "dp_secagg_remote_bundle_integrity": dp["status"],
        "observed_runtime_labeled_nonportable": "PASS" if observed_only else "FAIL",
        "linux_dependency_lock": "PASS" if linux_lock_present else "BLOCKED_NO_LINUX_LOCK",
        "clean_linux_reproduction": "PASS" if clean_linux_receipt_present else "BLOCKED_NO_CLEAN_LINUX_RECEIPT",
        "dataset_redistribution": "PASS" if dataset.get("license_identifier") and dataset.get("redistribution_approved") is True
                                  else "BLOCKED_LICENSE_NOT_DOCUMENTED",
        "project_code_license": "PASS" if project_license_present and code.get("license_identifier") and code.get("redistribution_approved") is True
                                else "BLOCKED_NO_APPROVED_PROJECT_LICENSE",
        "third_party_license_inventory": "PASS" if (
            third_party.get("license_inventory_complete") is True
            and dependency_inventory.get("license_inventory_complete") is True
        )
                                         else "BLOCKED_NO_DEPENDENCY_LICENSE_AUDIT",
        "matched_neural_results": "PASS" if matched_status in accepted_matched_statuses
                                  else "BLOCKED_MATCHED_NEURAL_RESULTS_PENDING",
        "network_deployment": "PASS" if network.get("deployment_verified") is True else network.get("status", "BLOCKED"),
        "ranking_domain_review": (
            "PASS"
            if ranking.get("ranking_enabled") is True
            and reference_gate.get("ranking_reference_supplement_enabled") is True
            else reference_gate.get("status", "BLOCKED")
            if reference_gate.get("ranking_reference_supplement_enabled") is not True
            else ranking.get("status", "BLOCKED")
        ),
        "agronomic_rule_review": "PASS" if agronomic.get("guidance_enabled") is True and domain_review.get("guidance_enabled") is True
                                  else domain_review.get("status", "BLOCKED"),
    }
    blocked = [name for name, status in checks.items() if status != "PASS"]
    return {
        "status": "RELEASE_REPRODUCIBILITY_PASS" if not blocked else "RELEASE_REPRODUCIBILITY_BLOCKED",
        "checks": checks,
        "blocked": blocked,
        "release_allowed": not blocked,
        "raw_dataset_in_release_allowed": checks["dataset_redistribution"] == "PASS",
        "model_training": False,
        "model_inference": False,
        "interpretation": (
            "Bundle integrity is necessary but does not establish Linux reproducibility or redistribution rights."
        ),
    }


def audit(output: Path) -> dict:
    output = Path(output).resolve()
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    evidence = {
        "rr06_bundle": latest_result_file("restart_rr06_remote_bundle", "bundle_manifest.json"),
        "dp_bundle": latest_result_file("restart_dp_secagg_remote_bundle", "bundle_manifest.json"),
        "completion": latest_result_file("restart_completion_audit", "completion_audit.json"),
        "network": latest_result_file("restart_network_deployment_gate", "deployment_gate.json"),
        "ranking": latest_result_file("restart_ranking_approval_gate", "ranking_gate.json"),
        "agronomic": latest_result_file("restart_agronomic_rule_validation", "rule_validation.json"),
        "domain_review": latest_result_file("restart_agronomic_domain_review_gate", "domain_review_gate.json"),
        "ranking_reference_gate": latest_result_file("restart_ranking_reference_gate", "reference_gate.json"),
        "dependency_inventory": latest_result_file("restart_dependency_license_inventory", "license_inventory.json"),
        "licensing": LICENSING, "runtime": RUNTIME,
    }
    try:
        clean_linux_candidate_path = latest_result_file("restart_clean_linux_reproduction", "receipt_candidate.json")
    except FileNotFoundError:
        clean_linux_candidate_path = None
    rr06, dp = verify_bundle(evidence["rr06_bundle"]), verify_bundle(evidence["dp_bundle"])
    licensing = read(LICENSING)
    completion, network, ranking, agronomic, domain_review, reference_gate, dependency_inventory = map(
        read, (evidence["completion"], evidence["network"], evidence["ranking"], evidence["agronomic"],
               evidence["domain_review"], evidence["ranking_reference_gate"], evidence["dependency_inventory"])
    )
    project_license_present = any((ROOT / name).is_file() for name in ("LICENSE", "LICENSE.txt", "COPYING"))
    linux_lock = validate_linux_lock()
    clean_linux_receipt = validate_clean_linux_receipt(CLEAN_LINUX_RECEIPT, linux_lock.get("sha256"))
    clean_linux_candidate = validate_clean_linux_candidate(
        clean_linux_candidate_path, linux_lock.get("sha256"), sha(evidence["dp_bundle"])
    ) if clean_linux_candidate_path else {"status": "BLOCKED_NO_CLEAN_LINUX_CANDIDATE", "evidence_verified": False}
    result = evaluate(
        rr06, dp, licensing, RUNTIME.read_text(encoding="utf-8"), completion, network, ranking, agronomic,
        domain_review, reference_gate, dependency_inventory,
        project_license_present=project_license_present,
        linux_lock_present=linux_lock["verified"],
        clean_linux_receipt_present=clean_linux_receipt["verified"],
    )
    result["bundle_details"] = {"rr06": rr06, "dp_secagg": dp}
    result["linux_evidence_details"] = {
        "dependency_lock": linux_lock,
        "clean_reproduction_candidate": clean_linux_candidate,
        "clean_reproduction_receipt": clean_linux_receipt,
    }
    output.mkdir(parents=True)
    atomic_json(output / "release_gate.json", result)
    atomic_json(output / "manifest.json", {
        "status": result["status"], "release_allowed": result["release_allowed"],
        "code_sha256": sha(Path(__file__)),
        "evidence_sha256": {name: sha(path) for name, path in evidence.items()},
        "release_gate_sha256": sha(output / "release_gate.json"),
        "model_training": False, "model_inference": False,
    })
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.output)
    print({"status": report["status"], "blocked": report["blocked"]})
