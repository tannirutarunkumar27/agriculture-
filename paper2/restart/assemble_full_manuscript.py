"""Assemble the validated Paper 2 Markdown components and bind their hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


COMPONENTS = (
    "PAPER2_FRONT_MATTER.md",
    "PAPER2_INTRODUCTION_RELATED_WORK_DRAFT.md",
    "PAPER2_DATA_METHODS_DRAFT.md",
    "PAPER2_RESULTS_DRAFT.md",
    "PAPER2_DISCUSSION_CONCLUSION_DRAFT.md",
    "PAPER2_REFERENCES.md",
)

FIGURES = (
    "paper2/results/restart_baseline_figures_v2/original_baseline_rmse.pdf",
    "paper2/results/restart_baseline_figures_v2/original_xgboost_geographic_rmse.pdf",
    "paper2/results/restart_baseline_figures_v2/posthoc_guard_sensitivity.pdf",
    "paper2/results/restart_public_refit_figures_v2/rmse_overview.pdf",
    "paper2/results/restart_public_refit_figures_v2/geographic_rmse.pdf",
    "paper2/results/restart_public_refit_figures_v2/support_rmse.pdf",
    "paper2/results/restart_matched_noise_figures_v1/rmse_overview.pdf",
    "paper2/results/restart_matched_noise_figures_v1/geographic_rmse.pdf",
    "paper2/results/restart_matched_noise_figures_v1/support_rmse.pdf",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("Assembly output and manifest must be new")

    paths = [args.manuscript_dir / name for name in COMPONENTS]
    missing = [str(path) for path in paths if not path.is_file()]
    figure_paths = [args.root / value for value in FIGURES]
    missing += [str(path) for path in figure_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs: " + ", ".join(missing))

    sections = [path.read_text(encoding="utf-8").strip() for path in paths]
    assembled = "\n\n".join(sections) + "\n"
    args.output.write_text(assembled, encoding="utf-8")
    manifest = {
        "version": "paper2-full-manuscript-draft-v1",
        "status": "ASSEMBLED_FROM_VALIDATED_COMPONENTS",
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "components": {path.name: sha256(path) for path in paths},
        "figures": {value: sha256(path) for value, path in zip(FIGURES, figure_paths)},
        "model_training": False,
        "model_inference": False,
        "paper1_modified": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "components": len(paths), "figures": len(FIGURES)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
