"""Prepare an ownership/license decision packet for the Paper 2 code scope.

No license is selected or written by this tool.  It exposes conflicting or
missing license signals so the project owner can make an informed decision.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from paper2.restart.paper2_completion_audit import ROOT, RESULTS, atomic_json, sha


INCLUDED_SUFFIXES = {".py", ".md", ".toml", ".json", ".sh", ".txt"}
EXCLUDED_PARTS = {"results", "data", "__pycache__", ".venv", "source_snapshot"}


def scoped_files(scope: Path) -> list[Path]:
    selected = []
    for path in scope.rglob("*"):
        # Exclude generated trees before touching entries beneath them. Some
        # archived Linux runtime symlinks are intentionally not stat-able from
        # Windows, and they are never distributable source scope.
        if any(part in EXCLUDED_PARTS for part in path.relative_to(scope).parts):
            continue
        try:
            is_source = path.is_file() and path.suffix.lower() in INCLUDED_SUFFIXES
        except OSError:
            continue
        if is_source:
            selected.append(path)
    return sorted(selected)


def prepare(output: Path, scope: Path = ROOT / "paper2", root_readme: Path = ROOT / "README.md") -> dict:
    output, scope, root_readme = Path(output).resolve(), Path(scope).resolve(), Path(root_readme).resolve()
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    if not scope.is_dir() or not root_readme.is_file():
        raise ValueError("Paper 2 source scope and root README are required")
    files = scoped_files(scope)
    if not files:
        raise ValueError("No distributable Paper 2 source files found")
    top_level_license = next((ROOT / name for name in ("LICENSE", "LICENSE.txt", "COPYING") if (ROOT / name).is_file()), None)
    readme_text = root_readme.read_text(encoding="utf-8", errors="replace")
    readme_claims_mit = bool(re.search(r"licensed under the MIT License|License:\s*MIT", readme_text, flags=re.I))
    embedded_licenses = []
    for path in scope.rglob("LICENSE*"):
        if path.is_file() and not any(part in EXCLUDED_PARTS for part in path.relative_to(scope).parts):
            embedded_licenses.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path),
                                      "summary": path.read_text(encoding="utf-8", errors="replace")[:160].replace("\n", " ")})
    inventory = [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size} for path in files]
    output.mkdir(parents=True)
    import csv
    with (output / "paper2_code_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "sha256", "bytes")); writer.writeheader(); writer.writerows(inventory)
    conflicts = []
    if top_level_license is None and readme_claims_mit:
        conflicts.append("root_readme_claims_mit_but_top_level_license_file_is_missing")
    if embedded_licenses:
        conflicts.append("paper2_subpackages_have_separate_license_files_requiring_scope_review")
    packet = {
        "status": "PROJECT_CODE_LICENSE_DECISION_KIT_READY_OWNER_DECISION_REQUIRED",
        "scope": "paper2 source/docs/configuration only; excludes raw data and generated results",
        "source_file_count": len(inventory), "top_level_license_present": top_level_license is not None,
        "top_level_license_path": top_level_license.relative_to(ROOT).as_posix() if top_level_license else None,
        "root_readme_claims_mit": readme_claims_mit, "embedded_license_files": embedded_licenses,
        "license_signal_conflicts": conflicts, "project_code_license_selected": False, "release_allowed": False,
        "interpretation": "Owner must confirm ownership, compatibility with embedded/upstream components, and selected SPDX terms before a top-level license is added.",
    }
    atomic_json(output / "project_code_license_packet.json", packet)
    (output / "OWNER_DECISION_INSTRUCTIONS.md").write_text(
        "# Project-code license decision\n\n"
        "1. Confirm you have authority to license every file in the listed Paper 2 scope.\n"
        "2. Review the embedded subpackage license files and third-party dependency terms.\n"
        "3. Select an SPDX license identifier, scope, notice/copyright holder, and effective date.\n"
        "4. Add a top-level license only after that decision; then reconcile any README claim with the selected terms.\n\n"
        "Do not use this packet to imply rights to raw data, derived data, trained weights, predictions, or caches.\n",
        encoding="utf-8",
    )
    atomic_json(output / "owner_license_decision.template.json", {
        "schema_version": 1, "owner_or_authorized_reviewer": "", "authority_basis": "", "decision_date": "",
        "spdx_license_identifier": "", "copyright_notice": "", "scope_confirmed": "",
        "embedded_license_compatibility_reviewed": "", "readme_reconciled": "", "rationale": "",
        "packet_sha256": sha(output / "project_code_license_packet.json"),
        "required_decision": "select a license, reject release, or request changes; leaving fields blank authorizes nothing",
    })
    atomic_json(output / "manifest.json", {
        "status": packet["status"], "root_readme_sha256": sha(root_readme),
        "code_inventory_sha256": sha(output / "paper2_code_inventory.csv"),
        "packet_sha256": sha(output / "project_code_license_packet.json"), "release_allowed": False,
    })
    return packet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", type=Path, default=ROOT / "paper2")
    parser.add_argument("--root-readme", type=Path, default=ROOT / "README.md")
    args = parser.parse_args()
    print(prepare(args.output, args.scope, args.root_readme))
