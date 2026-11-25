"""Fail-closed structural/source checks for the first Paper 2 draft sections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_HEADINGS = (
    "# I. Introduction",
    "# II. Related Work",
    "## A. Federated optimization under heterogeneous clients",
    "## B. Federated agricultural yield prediction",
    "## C. Grouped evaluation and support-aware reporting",
    "## D. Privacy mechanisms and simulation scope",
    "## E. Position relative to prior work",
)

REQUIRED_SCOPE_TERMS = (
    "607",
    "eight geographic groups",
    "conditional regression",
    "equal-group",
    "pooled",
    "utility simulation",
    "does not establish",
)

FORBIDDEN_ASSERTIONS = (
    "we propose a novel federated",
    "recommends the best crop",
    "guarantees privacy",
    "formally privacy-preserving",
    "knowledge-guided framework",
)


def validate(draft: str, registry: dict) -> list[str]:
    errors: list[str] = []
    lower = draft.lower()
    for heading in REQUIRED_HEADINGS:
        if heading not in draft:
            errors.append(f"missing heading: {heading}")
    for term in REQUIRED_SCOPE_TERMS:
        if term.lower() not in lower:
            errors.append(f"missing scope term: {term}")
    for assertion in FORBIDDEN_ASSERTIONS:
        if assertion in lower:
            errors.append(f"forbidden unsupported assertion: {assertion}")

    records = list(registry.get("closest_studies", [])) + list(
        registry.get("algorithm_sources", [])
    )
    for record in records:
        url = record.get("source_url") if isinstance(record, dict) else None
        if not url or url not in draft:
            label = record.get("id", record.get("name", "unknown")) if isinstance(record, dict) else "unknown"
            errors.append(f"missing registered source link: {label}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    draft = args.draft.read_text(encoding="utf-8")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    errors = validate(draft, registry)
    result = {
        "status": "PASS" if not errors else "FAIL",
        "draft": str(args.draft),
        "registry": str(args.registry),
        "registered_study_links": len(registry.get("closest_studies", [])),
        "registered_algorithm_links": len(registry.get("algorithm_sources", [])),
        "required_heading_count": len(REQUIRED_HEADINGS),
        "errors": errors,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
