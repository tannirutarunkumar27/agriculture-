"""Validate assembled Paper 2 structure, boundaries, sources, and assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from paper2.restart.assemble_full_manuscript import COMPONENTS, FIGURES


SECTION_HEADINGS = tuple(f"# {roman}." for roman in ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"))
ABSTRACT_FACTS = (
    "607 recorded cereal-legume",
    "eight geographic groups",
    "1.2461 t/ha",
    "1.7335 t/ha",
    "1.067407 +/- 0.021053",
    "1.182853, 2.218085, and 5.628243",
    "does not establish best-crop recommendation",
    "secure aggregation",
    "end-to-end differential privacy",
)
FORBIDDEN_TITLE_TERMS = ("knowledge-guided", "recommendation", "privacy-preserving")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(text: str, registry: dict, manifest: dict, root: Path, manuscript_dir: Path) -> list[str]:
    errors: list[str] = []
    normalized = " ".join(text.lower().split())
    title = text.splitlines()[0].lower() if text.splitlines() else ""
    for term in FORBIDDEN_TITLE_TERMS:
        if term in title:
            errors.append(f"forbidden title term: {term}")
    for fact in ABSTRACT_FACTS:
        if " ".join(fact.lower().split()) not in normalized:
            errors.append(f"missing abstract/scope fact: {fact}")
    positions = []
    for heading in SECTION_HEADINGS:
        count = sum(1 for line in text.splitlines() if line.startswith(heading))
        if count != 1:
            errors.append(f"expected one section heading {heading}, found {count}")
        positions.append(text.find(heading))
    if any(value < 0 for value in positions) or positions != sorted(positions):
        errors.append("sections I-IX are missing or out of order")

    source_records = list(registry.get("closest_studies", [])) + list(registry.get("algorithm_sources", []))
    for record in source_records:
        url = record.get("source_url")
        if not url or url not in text:
            errors.append(f"missing registered source URL: {record.get('id', record.get('name'))}")
    for url in (
        "https://arxiv.org/abs/1605.02065",
        "https://research.google/pubs/practical-secure-aggregation-for-privacy-preserving-machine-learning/",
    ):
        if url not in text:
            errors.append(f"missing privacy source URL: {url}")

    for name, expected in manifest.get("components", {}).items():
        path = manuscript_dir / name
        if not path.is_file() or digest(path) != expected:
            errors.append(f"component hash mismatch: {name}")
    for relative, expected in manifest.get("figures", {}).items():
        path = root / relative
        if not path.is_file() or digest(path) != expected:
            errors.append(f"figure hash mismatch: {relative}")
    if len(manifest.get("components", {})) != len(COMPONENTS):
        errors.append("manifest component count mismatch")
    if len(manifest.get("figures", {})) != len(FIGURES):
        errors.append("manifest figure count mismatch")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manuscript-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = args.draft.read_text(encoding="utf-8")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    errors = validate(text, registry, manifest, args.root, args.manuscript_dir)
    result = {
        "status": "PASS" if not errors else "FAIL",
        "draft_sha256": digest(args.draft),
        "component_count": len(manifest.get("components", {})),
        "figure_count": len(manifest.get("figures", {})),
        "section_count": len(SECTION_HEADINGS),
        "registered_source_count": len(registry.get("closest_studies", [])) + len(registry.get("algorithm_sources", [])),
        "errors": errors,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
