"""Evidence-bound checks for the Paper 2 Data and Methods draft."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_HEADINGS = (
    "# III. Data and Problem Formulation",
    "## A. Source data and unit of analysis",
    "## B. Geographic grouping and support",
    "## C. Predictor contract",
    "## D. Scientific scope",
    "# IV. Methods",
    "## A. Nested geographic evaluation",
    "## B. Regression baselines",
    "## C. Inner selection and outer refitting",
    "## D. Fixed-public refit and matched noise simulation",
    "## E. Evaluation measures",
    "## F. Reproducibility safeguards",
)

REQUIRED_FACTS = (
    "607 crop-management systems",
    "eight groups",
    "Auzeville (354)",
    "Kassel (4)",
    "Reading (4)",
    "56 inner folds",
    "1,120 completed validation candidates",
    "320 outer evaluations",
    "286 selected the final allowed round",
    "80 of 224 candidates",
    "11 of 56 selected checkpoints",
    "54-dimensional",
    "C=4\\eta=0.04",
    "not as certified application differential privacy",
)

FORBIDDEN_ASSERTIONS = (
    "best-crop recommendation model",
    "guarantees differential privacy",
    "secure aggregation was implemented",
    "independent field validation",
    "knowledge-guided regression method",
)


def validate(text: str) -> list[str]:
    errors: list[str] = []
    lower = " ".join(text.lower().split())
    for heading in REQUIRED_HEADINGS:
        if heading not in text:
            errors.append(f"missing heading: {heading}")
    for fact in REQUIRED_FACTS:
        if " ".join(fact.lower().split()) not in lower:
            errors.append(f"missing evidence fact: {fact}")
    for assertion in FORBIDDEN_ASSERTIONS:
        if assertion in lower:
            errors.append(f"forbidden assertion: {assertion}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = args.draft.read_text(encoding="utf-8")
    errors = validate(text)
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
