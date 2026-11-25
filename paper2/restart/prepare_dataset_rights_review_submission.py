"""Prepare a source-bound dataset-rights review kit without asserting rights.

The authoritative record may be public to download while redistribution rights
remain unspecified. This module therefore prepares a contact/review packet and
never changes the release licensing registry.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from paper2.restart.paper2_completion_audit import ROOT, RESULTS, atomic_json, read, sha


LICENSING = ROOT / "paper2/configs/release_licensing_registry_v1.json"
DATA_AUDIT = ROOT / "paper2/results/dataset_client_validation_v1/audit.json"
PUBLIC_DATA_MANIFEST = ROOT / "paper2/results/restart_public_data_v1/manifest.json"


def prepare(
    output: Path,
    licensing_path: Path = LICENSING,
    data_audit_path: Path = DATA_AUDIT,
    public_data_manifest: Path = PUBLIC_DATA_MANIFEST,
) -> dict:
    output = Path(output).resolve()
    licensing_path, data_audit_path, public_data_manifest = map(Path, (licensing_path, data_audit_path, public_data_manifest))
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    licensing, audit, public_manifest = map(read, (licensing_path, data_audit_path, public_data_manifest))
    dataset = licensing.get("dataset", {})
    if (dataset.get("doi") != "10.5281/zenodo.14160885" or dataset.get("version") != "2.0.0"
            or dataset.get("redistribution_approved") is not False):
        raise ValueError("Expected unresolved Gaudio et al. Zenodo v2.0.0 rights registry")
    if audit.get("status") != "PASS_DATASET_CLIENT_AUDIT" or public_manifest.get("status") != "PREPARED_NO_TRAINING":
        raise ValueError("A verified dataset audit and prepared-data manifest are required")
    output.mkdir(parents=True)
    packet = {
        "status": "DATASET_RIGHTS_REVIEW_SUBMISSION_KIT_READY_PENDING_OWNER_OR_LEGAL_REVIEW",
        "dataset_name": dataset.get("name"), "version": dataset.get("version"), "doi": dataset.get("doi"),
        "record_url": dataset.get("record_url"), "license_identifier_observed": dataset.get("license_identifier"),
        "redistribution_approved": False, "dataset_audit": {
            "systems": audit.get("systems"), "experiments": audit.get("experiments"),
            "source_manifest_sha256": audit.get("source_manifest_sha256"),
            "regression_rows_sha256": audit.get("regression_rows_sha256"),
        },
        "prepared_data_manifest_sha256": sha(public_data_manifest),
        "requested_scopes": [
            "raw source files", "prepared/split derivative arrays", "model weights, predictions, and caches",
            "aggregated tables and figures", "code-only release with DOI download instructions",
        ],
        "release_allowed": False,
        "interpretation": "Packet records provenance and requested scopes only. Download availability and a DOI do not establish redistribution permission.",
    }
    atomic_json(output / "dataset_rights_packet.json", packet)
    (output / "RIGHTS_REQUEST_TEMPLATE.md").write_text(
        "# Dataset rights request\n\n"
        "Subject: Request to clarify reuse and redistribution terms for Zenodo 10.5281/zenodo.14160885 (v2.0.0)\n\n"
        "We are preparing a reproducibility package based on the cited dataset. The current project will not redistribute any source or derived data unless permission is documented. "
        "Please state whether each requested scope in `dataset_rights_packet.json` is permitted, prohibited, or requires conditions/attribution. "
        "A response should identify the rights holder, date, applicable license/terms, and any required citation or access restrictions.\n",
        encoding="utf-8",
    )
    atomic_json(output / "owner_or_legal_review.template.json", {
        "schema_version": 1, "reviewer_or_rights_holder": "", "affiliation": "", "review_date": "",
        "authority_basis": "", "dataset_rights_decision": "", "license_identifier_or_terms_url": "",
        "approved_scopes": [], "conditions_and_attribution": "", "packet_sha256": sha(output / "dataset_rights_packet.json"),
        "required_decision": "approve specified scopes, prohibit, or request changes; no blank approval is valid",
    })
    atomic_json(output / "manifest.json", {
        "status": packet["status"], "licensing_registry_sha256": sha(licensing_path),
        "dataset_audit_sha256": sha(data_audit_path), "prepared_manifest_sha256": sha(public_data_manifest),
        "packet_sha256": sha(output / "dataset_rights_packet.json"), "release_allowed": False,
    })
    return packet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--licensing", type=Path, default=LICENSING)
    parser.add_argument("--data-audit", type=Path, default=DATA_AUDIT)
    parser.add_argument("--public-data-manifest", type=Path, default=PUBLIC_DATA_MANIFEST)
    args = parser.parse_args()
    print(prepare(args.output, args.licensing, args.data_audit, args.public_data_manifest))
