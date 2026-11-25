"""Generate a candidate dependency-license inventory without clearing release rights."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import re
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = ROOT / "paper2/requirements-runtime-observed.txt"
LINUX_LOCK = ROOT / "paper2/requirements-linux-lock.txt"


def requirement_names(path: Path) -> list[str]:
    names = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)", line)
        if match:
            names.append(match.group(1))
    return names


def locked_versions(path: Path) -> dict[str, str]:
    """Read exact package pins from a pip/uv requirements file."""
    values = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*([A-Za-z0-9_.-]+)==([^\s\\]+)", raw)
        if match:
            values[canonicalize_name(match.group(1))] = match.group(2)
    return values


def _portable_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _license_evidence(dist: metadata.Distribution) -> tuple[str, str]:
    """Return license/notice file names and hashes without copying their text."""
    evidence = []
    for entry in dist.files or []:
        name = Path(str(entry)).name.lower()
        if not (name.startswith(("license", "licence", "copying", "notice"))):
            continue
        path = Path(dist.locate_file(entry))
        if path.is_file():
            evidence.append(f"{entry}:{hashlib.sha256(path.read_bytes()).hexdigest()}")
    return ";".join(item.split(":", 1)[0] for item in evidence), ";".join(evidence)


def _dependency_names(dist: metadata.Distribution) -> list[str]:
    names = []
    for raw in dist.requires or []:
        try:
            requirement = Requirement(raw)
            if requirement.marker is None or requirement.marker.evaluate():
                names.append(requirement.name)
        except (InvalidRequirement, ValueError):
            # The malformed metadata remains visible through the package row's
            # status instead of silently authorizing a release.
            continue
    return names


def inventory(path: Path = REQUIREMENTS, *, environment_scope: str = "observed_runtime") -> tuple[dict, list[dict]]:
    if environment_scope not in {"observed_runtime", "clean_wsl_linux"}:
        raise ValueError("Unknown dependency inventory environment scope")
    direct_names = requirement_names(path)
    direct = {canonicalize_name(name) for name in direct_names}
    pins = locked_versions(path)
    queue = list(direct_names)
    visited: set[str] = set()
    rows = []
    while queue:
        name = queue.pop(0)
        canonical = canonicalize_name(name)
        if canonical in visited:
            continue
        visited.add(canonical)
        try:
            dist = metadata.distribution(name)
            classifiers = [c for c in dist.metadata.get_all("Classifier") or [] if "License" in c]
            license_files, license_hashes = _license_evidence(dist)
            expression = (dist.metadata.get("License-Expression") or "").strip()
            rows.append({
                "package": dist.metadata.get("Name") or name,
                "direct_requirement": canonical in direct,
                "locked_version": pins.get(canonical, ""),
                "installed_version_observed": dist.version,
                "installed_version_matches_lock": (
                    "NOT_LOCKED" if canonical not in pins else str(dist.version == pins[canonical]).upper()
                ),
                "license_expression_observed": expression,
                "license_field_observed": (dist.metadata.get("License") or "").strip().replace("\n", " ")[:500],
                "license_classifiers_observed": ";".join(classifiers),
                "license_files_observed": license_files,
                "license_file_sha256": license_hashes,
                "project_url": dist.metadata.get("Home-page") or dist.metadata.get("Project-URL") or "",
                "status": "OBSERVED_CLEAN_WSL_LINUX" if environment_scope == "clean_wsl_linux" else "OBSERVED_ENVIRONMENT_ONLY",
                "independent_review_status": "PENDING_INDEPENDENT_LICENSE_REVIEW",
            })
            queue.extend(_dependency_names(dist))
        except metadata.PackageNotFoundError:
            rows.append({
                "package": name,
                "direct_requirement": canonical in direct,
                "locked_version": pins.get(canonical, ""),
                "installed_version_observed": "",
                "installed_version_matches_lock": "FALSE" if canonical in pins else "NOT_LOCKED",
                "license_expression_observed": "",
                "license_field_observed": "",
                "license_classifiers_observed": "",
                "license_files_observed": "",
                "license_file_sha256": "",
                "project_url": "",
                "status": "NOT_INSTALLED_IN_ENVIRONMENT",
                "independent_review_status": "PENDING_INDEPENDENT_LICENSE_REVIEW",
            })
    rows.sort(key=lambda row: (not row["direct_requirement"], canonicalize_name(row["package"])))
    observed = [row for row in rows if row["status"].startswith("OBSERVED_")]
    with_license_evidence = [
        row for row in observed
        if row["license_expression_observed"] or row["license_field_observed"]
        or row["license_classifiers_observed"] or row["license_file_sha256"]
    ]
    locked_rows = [row for row in rows if row["locked_version"]]
    version_mismatches = [row["package"] for row in locked_rows if row["installed_version_matches_lock"] != "TRUE"]
    report = {
        "status": (
            "DEPENDENCY_LICENSE_INVENTORY_CLEAN_WSL_PENDING_INDEPENDENT_REVIEW"
            if environment_scope == "clean_wsl_linux"
            else "DEPENDENCY_LICENSE_INVENTORY_PENDING_REMOTE_LINUX_REVIEW"
        ),
        "requirements_file": _portable_path(path),
        "environment_scope": environment_scope,
        "direct_requirement_count": len(direct),
        "inventory_package_count": len(rows),
        "transitive_package_count": sum(not row["direct_requirement"] for row in rows),
        "observed_in_environment": len(observed),
        "missing_from_environment": len(rows) - len(observed),
        "packages_with_license_evidence": len(with_license_evidence),
        "packages_without_license_evidence": len(observed) - len(with_license_evidence),
        "locked_package_count": len(locked_rows),
        "locked_version_mismatches": version_mismatches,
        "environment_matches_lock": not version_mismatches,
        "license_inventory_complete": False,
        "release_allowed": False,
        "independent_review_required": True,
        "clean_linux_receipt_required": True,
        "model_training": False,
        "model_inference": False,
    }
    return report, rows


def main(output: Path, requirements: Path = REQUIREMENTS, environment_scope: str = "observed_runtime") -> None:
    output = Path(output).resolve()
    if output.exists() or ROOT / "paper2/results" not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    report, rows = inventory(requirements, environment_scope=environment_scope)
    output.mkdir(parents=True)
    (output / "license_inventory.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "dependency_license_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "review_notes.md").write_text(
        "# Dependency-license inventory review\n\n"
        "This inventory includes the installed dependency closure and hashes any "
        "license/notice files exposed by package metadata. It is an observed "
        "bundled-runtime inventory only, not legal advice or evidence "
        "for the final Linux environment and does not authorize redistribution. "
        "Bind it to a clean Linux lock, record package/wheel provenance, and obtain "
        "independent license review before changing the release registry.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, default=REQUIREMENTS)
    parser.add_argument("--environment-scope", choices=("observed_runtime", "clean_wsl_linux"), default="observed_runtime")
    args = parser.parse_args()
    main(args.output, args.requirements, args.environment_scope)
