"""Evidence and scope checks for final Paper 2 narrative sections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_HEADINGS = (
    "# VI. Discussion",
    "## A. Interpreting the baseline and support results",
    "## B. Equal-group, pooled, and support-conditioned views",
    "## C. Aggregation noise and privacy interpretation",
    "## D. Relation to prior agricultural federated learning",
    "# VII. Limitations and Threats to Validity",
    "# VIII. Reproducibility",
    "# IX. Conclusion",
)

REQUIRED_FACTS = (
    "354 of 607 systems",
    "all 265 systems",
    "four fixed-public held-out groups had negative mean r2",
    "286 of 672 candidates",
    "80 of 224 candidates",
    "11 of 56 selected checkpoints",
    "approximately 1297--1378",
    "secure aggregation was not implemented",
    "1,120 original inner candidates",
    "320 original outer evaluations",
    "320 post-hoc guard evaluations",
    "224 fixed-public inner candidates",
    "40 unnoised fixed-public refits",
    "160 matched-noise evaluations",
    "nine accepted scientific figures",
    "not a new federated optimizer",
)

FORBIDDEN_ASSERTIONS = (
    "proves universal superiority",
    "provides best-crop recommendations",
    "guarantees end-to-end privacy",
    "independent field validation was achieved",
    "secure aggregation protected",
)


def validate(text: str) -> list[str]:
    normalized = " ".join(text.lower().split())
    errors: list[str] = []
    for heading in REQUIRED_HEADINGS:
        if heading not in text:
            errors.append(f"missing heading: {heading}")
    for fact in REQUIRED_FACTS:
        if " ".join(fact.lower().split()) not in normalized:
            errors.append(f"missing evidence/scope fact: {fact}")
    for assertion in FORBIDDEN_ASSERTIONS:
        if assertion in normalized:
            errors.append(f"forbidden assertion: {assertion}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    errors = validate(args.draft.read_text(encoding="utf-8"))
    result = {
        "status": "PASS" if not errors else "FAIL",
        "draft": str(args.draft),
        "required_heading_count": len(REQUIRED_HEADINGS),
        "required_fact_count": len(REQUIRED_FACTS),
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
