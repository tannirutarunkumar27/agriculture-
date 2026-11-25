"""Evidence checks for the Paper 2 Results draft."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_HEADINGS = (
    "# V. Results",
    "## A. Original eight-method outer evaluation",
    "## B. Support-conditioned and post-hoc sensitivity analysis",
    "## C. Fixed-public unnoised refit",
    "## D. Matched clipping and aggregation-noise utility",
    "## E. Summary of empirical findings",
)

REQUIRED_FACTS = (
    "320 evaluations",
    "1.2461 +/- 0.0000",
    "1.7335 +/- 0.0000",
    "36.7482",
    "post-hoc",
    "1.067407 +/- 0.021053",
    "1.201189 +/- 0.038476",
    "four groups had negative mean r2",
    "342 systems",
    "265 auzeville systems",
    "2.218085 +/- 0.491871",
    "5.628243 +/- 0.500009",
    "29/40",
    "40/40",
    "23,174 events",
    "1297.067440 to 1377.757121",
    "do not establish application differential privacy",
)

FORBIDDEN_ASSERTIONS = (
    "statistically superior",
    "independent field confirmation",
    "guarantees privacy",
    "best-crop recommendation was achieved",
    "establishes universal unseen-species capability",
)


def validate(text: str) -> list[str]:
    normalized = " ".join(text.lower().split())
    errors: list[str] = []
    for heading in REQUIRED_HEADINGS:
        if heading not in text:
            errors.append(f"missing heading: {heading}")
    for fact in REQUIRED_FACTS:
        if " ".join(fact.lower().split()) not in normalized:
            errors.append(f"missing evidence fact: {fact}")
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
