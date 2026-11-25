"""Audit Paper 2 local evidence and report portable-reproduction blockers.

This is deliberately an inventory/validation step.  It does not train a model,
run inference, or claim that the current Windows checkout is a portable release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
WINDOWS_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\r\n\"']+")
CORE_RUNTIME_PACKAGES = {
    "torch",
    "numpy",
    "pandas",
    "scikit-learn",
    "joblib",
    "xgboost",
}


@dataclass(frozen=True)
class Asset:
    path: str
    role: str
    json_expectations: tuple[tuple[str, Any], ...] = ()


ASSETS = (
    Asset("paper2/manuscript/PAPER2_FULL_MANUSCRIPT_DRAFT.md", "assembled manuscript draft"),
    Asset("paper2/research/restart_regression_closest_work.json", "registered literature and algorithm sources"),
    Asset("paper2/research/evidence_claim_matrix.json", "claim-to-evidence registry"),
    Asset("paper2/research/restart_reproducibility_guide.md", "local verification and recovery guide"),
    Asset("paper2/requirements-runtime-observed.txt", "observed Windows runtime versions"),
    Asset(
        "paper2/results/restart_regression_data_v1/manifest.json",
        "prepared regression data manifest",
        (("status", "PREPARED_REGRESSION_DATA_NO_MODEL_TRAINING"), ("rows", 607), ("outer_folds", 8)),
    ),
    Asset(
        "paper2/results/restart_regression_verified_v1/verification.json",
        "prepared-data independent verification",
        (("status", "PASS"), ("verified_folds", 64), ("rows", 607), ("outer_test_row_coverage_once", True)),
    ),
    Asset(
        "paper2/results/restart_validation_verified_v2/verification.json",
        "inner-validation audit",
        (("status", "PASS_COMPLETE"), ("verified_completed_runs", 1120), ("planned_runs", 1120), ("pending_runs", 0)),
    ),
    Asset(
        "paper2/results/restart_outer_verified_v3/verification.json",
        "original outer-evaluation audit",
        (("status", "PASS_COMPLETE"), ("verified", 320), ("planned", 320), ("pending", 0)),
    ),
    Asset(
        "paper2/results/restart_guard_analysis_v1/analysis.json",
        "post-hoc guard sensitivity analysis",
        (("status", "PASS_COMPLETE_POST_HOC_SENSITIVITY"), ("verified", 320), ("training_performed", False)),
    ),
    Asset(
        "paper2/results/restart_control_search_integrity_v1/verification.json",
        "fixed-public control-search provenance",
        (("status", "PASS_COMPLETE_SEARCH_PROVENANCE"), ("verified", 224), ("attempts", 224), ("failed_attempts", 0)),
    ),
    Asset(
        "paper2/results/restart_public_refit_verified_v2/verification.json",
        "fixed-public unnoised-refit audit",
        (("status", "PASS_COMPLETE_REFIT"), ("verified", 40), ("planned", 40), ("pending", 0), ("privacy_claim", False)),
    ),
    Asset(
        "paper2/results/restart_matched_noise_verified_v2/verification.json",
        "matched-noise evaluation audit",
        (("status", "PASS_COMPLETE_MATCHED_STUDY"), ("verified", 160), ("planned", 160), ("pending", 0), ("noise_events", 23174), ("application_privacy_certified", False)),
    ),
    Asset(
        "paper2/results/restart_matched_noise_analysis_v1/analysis.json",
        "matched-noise complete analysis",
        (("status", "VERIFIED_COMPLETE_MATCHED_ANALYSIS"), ("model_reinference", False), ("application_privacy_certified", False)),
    ),
    Asset(
        "paper2/results/restart_baseline_figures_v2/manifest.json",
        "three accepted original/guard figures",
        (("status", "GENERATED_FROM_COMPLETE_SEPARATELY_AUDITED_ANALYSES"), ("model_training", False), ("model_reinference", False)),
    ),
    Asset(
        "paper2/results/restart_public_refit_figures_v2/manifest.json",
        "three accepted public-refit figures",
        (("status", "GENERATED_FROM_COMPLETE_AUDITED_ANALYSIS"), ("model_reinference", False)),
    ),
    Asset(
        "paper2/results/restart_matched_noise_figures_v1/manifest.json",
        "three accepted matched-noise figures",
        (("status", "GENERATED_FROM_COMPLETE_AUDITED_ANALYSIS"), ("model_reinference", False)),
    ),
    Asset(
        "paper2/results/restart_claim_matrix_validation_v1/verification.json",
        "claim-matrix validation",
        (("status", "PASS"), ("claims", 10), ("blocked", 4)),
    ),
    Asset(
        "paper2/results/restart_regression_literature_validation_v1/verification.json",
        "literature-registry validation",
        (("status", "PASS"), ("study_count", 8), ("algorithm_source_count", 3)),
    ),
    Asset(
        "paper2/results/restart_full_manuscript_v1/manifest.json",
        "assembled-manuscript manifest",
        (("status", "ASSEMBLED_FROM_VALIDATED_COMPONENTS"), ("model_training", False), ("model_inference", False), ("paper1_modified", False)),
    ),
    Asset(
        "paper2/results/restart_full_manuscript_validation_v1/verification.json",
        "assembled-manuscript validation",
        (("status", "PASS"), ("component_count", 6), ("figure_count", 9), ("section_count", 9), ("registered_source_count", 11)),
    ),
)


PORTABILITY_BLOCKERS = (
    "Selected historical manifests contain checkout-specific absolute paths.",
    "requirements-runtime-observed.txt records one working Windows environment; it is not a cross-platform lock file.",
    "No fresh-environment installation and end-to-end clean-room execution has been recorded.",
    "No Linux execution has been verified.",
    "Raw-source redistribution permissions and release licensing have not been completed.",
    "Production privacy accounting, secure aggregation, and external deployment infrastructure are not established.",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_windows_absolute_paths(text: str) -> list[str]:
    """Return unique Windows absolute paths without exposing duplicate values."""
    return sorted(set(match.group(0).strip() for match in WINDOWS_ABSOLUTE_PATH.finditer(text)))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _runtime_check(path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    packages = {
        line.split("==", 1)[0].strip().lower()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "==" in line
    }
    missing = sorted(CORE_RUNTIME_PACKAGES - packages)
    if missing:
        errors.append(f"runtime record lacks core packages: {', '.join(missing)}")
    if "not a cross-platform lock" not in text.lower():
        errors.append("runtime record must explicitly say it is not a cross-platform lock")
    return errors, sorted(packages)


def audit(root: Path = ROOT, assets: Iterable[Asset] = ASSETS) -> dict[str, Any]:
    root = root.resolve()
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    portability_paths: list[dict[str, Any]] = []

    for asset in assets:
        path = root / asset.path
        record: dict[str, Any] = {"path": asset.path, "role": asset.role, "exists": path.is_file()}
        if not path.is_file():
            errors.append(f"missing required asset: {asset.path}")
            records.append(record)
            continue
        if not path.resolve().is_relative_to(root):
            errors.append(f"asset resolves outside repository: {asset.path}")
            records.append(record)
            continue
        if "KG-Hybrid" in path.parts:
            errors.append(f"Paper 1 asset is not allowed in Paper 2 inventory: {asset.path}")
        record.update({"bytes": path.stat().st_size, "sha256": sha256(path)})

        text = path.read_text(encoding="utf-8", errors="replace")
        absolute_paths = find_windows_absolute_paths(text)
        if absolute_paths:
            portability_paths.append({"path": asset.path, "count": len(absolute_paths)})
        if path.suffix.lower() == ".json":
            try:
                data = _read_json(path)
            except (json.JSONDecodeError, ValueError) as exc:
                errors.append(f"invalid JSON {asset.path}: {exc}")
            else:
                for key, expected in asset.json_expectations:
                    actual = data.get(key)
                    if actual != expected:
                        errors.append(f"{asset.path}: expected {key}={expected!r}, got {actual!r}")
        records.append(record)

    runtime_path = root / "paper2/requirements-runtime-observed.txt"
    runtime_packages: list[str] = []
    if runtime_path.is_file():
        runtime_errors, runtime_packages = _runtime_check(runtime_path)
        errors.extend(runtime_errors)

    local_status = "PASS" if not errors else "FAIL"
    return {
        "status": "PASS_LOCAL_EVIDENCE_BLOCKED_PORTABLE_REPRODUCTION" if not errors else "FAIL_LOCAL_EVIDENCE",
        "local_evidence_status": local_status,
        "portable_clean_room_status": "BLOCKED",
        "required_asset_count": len(records),
        "verified_asset_count": sum(bool(record.get("sha256")) for record in records),
        "runtime_packages": runtime_packages,
        "absolute_path_findings": portability_paths,
        "portability_blockers": list(PORTABILITY_BLOCKERS),
        "errors": errors,
        "inventory": records,
        "model_training_performed": False,
        "model_inference_performed": False,
        "scope": "Local evidence integrity and release-readiness inventory only; not clean-room reproduction.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite versioned output: {args.output}")
    result = audit()
    args.output.mkdir(parents=True)
    (args.output / "readiness.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "required_asset_count", "verified_asset_count", "errors")}, indent=2))
    return 0 if result["local_evidence_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
