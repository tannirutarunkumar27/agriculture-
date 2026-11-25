"""Create a single immutable handoff kit for independent agronomic review.

The kit deliberately contains blank decisions and a blank attestation.  It is
not an approval mechanism: an external reviewer must complete the copies and
the existing domain-review gate must audit the returned CSV before any
guidance-study protocol can use a rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper2.restart.agronomic_domain_review_gate import REGISTRY, make_template
from paper2.restart.paper2_completion_audit import RESULTS, atomic_json, read, sha
from paper2.restart.prepare_agronomic_review_packet import packet_rows


ATTESTATION_SCHEMA = {
    "schema_version": 1,
    "reviewer_name": "",
    "reviewer_affiliation": "",
    "reviewer_role_or_credential": "",
    "reviewer_contact": "",
    "review_date": "",
    "reviewer_independent": "",
    "conflict_of_interest_disclosed": "",
    "attestation": (
        "I independently reviewed every submitted rule against its cited source "
        "and recorded include/exclude decisions with scope limitations."
    ),
    "registry_sha256": "",
    "decision_template_sha256": "",
}


def prepare(output: Path, registry_path: Path = REGISTRY) -> dict:
    output, registry_path = Path(output).resolve(), Path(registry_path).resolve()
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    registry = read(registry_path)
    rows = packet_rows(registry)  # validates the pending, source-bound registry
    template = make_template(registry)
    output.mkdir(parents=True)
    template.to_csv(output / "domain_review_decisions.csv", index=False)
    # Keep cited facts separate from mutable reviewer decisions.
    import pandas as pd
    pd.DataFrame(rows).to_csv(output / "source_packet.csv", index=False)
    attestation = dict(ATTESTATION_SCHEMA)
    attestation["registry_sha256"] = sha(registry_path)
    attestation["decision_template_sha256"] = sha(output / "domain_review_decisions.csv")
    atomic_json(output / "reviewer_attestation.template.json", attestation)
    (output / "SUBMISSION_INSTRUCTIONS.md").write_text(
        "# Agronomic review submission\n\n"
        "1. Review each cited source in `source_packet.csv`.\n"
        "2. Complete every column in `domain_review_decisions.csv`; use only `include` or `exclude`.\n"
        "3. Complete a copy of `reviewer_attestation.template.json`, including independence and conflict declarations.\n"
        "4. Return the two completed files unchanged in schema. Project staff must run the domain-review gate.\n\n"
        "This kit cannot authorize deployment, ranking, model training, or a causal agronomic claim. "
        "It supports only an independently reviewed, bounded soft-prior study.\n",
        encoding="utf-8",
    )
    report = {
        "status": "AGRONOMIC_REVIEW_SUBMISSION_KIT_READY_PENDING_INDEPENDENT_REVIEW",
        "rules": len(rows), "guidance_enabled": False, "model_training": False,
        "interpretation": "Blank, source-bound reviewer kit only; an external qualified reviewer must return completed evidence.",
    }
    atomic_json(output / "submission_kit.json", report)
    atomic_json(output / "manifest.json", {
        "status": report["status"], "registry_sha256": sha(registry_path),
        "decision_template_sha256": sha(output / "domain_review_decisions.csv"),
        "source_packet_sha256": sha(output / "source_packet.csv"),
        "attestation_template_sha256": sha(output / "reviewer_attestation.template.json"),
        "instructions_sha256": sha(output / "SUBMISSION_INSTRUCTIONS.md"),
        "guidance_enabled": False,
    })
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    args = parser.parse_args()
    print(prepare(args.output, args.registry))
