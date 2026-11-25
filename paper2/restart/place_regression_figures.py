"""Place audited existing figures in a new manuscript version; never replot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from paper2.restart.revise_regression_manuscript import replace_once


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "paper2/manuscript/PAPER2_REGRESSION_MANUSCRIPT_V3.md"
SOURCE_SHA256 = "5edc23ae8714a1cc0840ccdbd1e681a77acb6b91c8581931941da0cb7a9fc0dd"

FIGURES = (
    (1, "restart_baseline_figures_v2", "original_baseline_rmse",
     "XGBoost had the lowest observed equal-group and pooled RMSE within this frozen"),
    (2, "restart_baseline_figures_v2", "original_xgboost_geographic_rmse",
     "## B. Support-conditioned and post-hoc sensitivity analysis"),
    (3, "restart_baseline_figures_v2", "posthoc_guard_sensitivity",
     "## C. Fixed-public unnoised refit"),
    (4, "restart_public_refit_figures_v2", "support_rmse",
     "The fixed-public result is a later development stage with a different feature"),
    (5, "restart_matched_noise_figures_v1", "rmse_overview",
     "Mean paired increases in equal-group RMSE relative to the cached reference were"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_figures() -> list[dict]:
    verified = []
    for number, package, stem, anchor in FIGURES:
        directory = ROOT / "paper2/results" / package
        manifest_path = directory / "manifest.json"
        captions_path = directory / "captions.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for path in (captions_path, directory / f"{stem}.png", directory / f"{stem}.pdf"):
            if sha256(path) != manifest["assets"][path.name]:
                raise ValueError(f"Audited figure asset changed: {path}")
        captions = json.loads(captions_path.read_text(encoding="utf-8"))
        matched = [entry["caption"] for entry in captions if entry["name"] == stem]
        if len(matched) != 1:
            raise ValueError(f"No unique accepted caption for {package}/{stem}")
        verified.append({
            "number": number,
            "package": package,
            "stem": stem,
            "anchor": anchor,
            "caption": matched[0],
            "manifest_sha256": sha256(manifest_path),
            "png_sha256": manifest["assets"][f"{stem}.png"],
            "pdf_sha256": manifest["assets"][f"{stem}.pdf"],
        })
    return verified


def place(source_text: str, figures: list[dict]) -> str:
    text = source_text
    if len(figures) != 5 or [fig["number"] for fig in figures] != [1, 2, 3, 4, 5]:
        raise ValueError("Expected the five prospectively selected manuscript figures")
    for figure in figures:
        path = f"../results/{figure['package']}/{figure['stem']}.png"
        block = (f"![Figure {figure['number']}: {figure['stem'].replace('_', ' ')}]({path})\n\n"
                 f"**Fig. {figure['number']}.** {figure['caption']}\n\n")
        text = replace_once(text, figure["anchor"], block + figure["anchor"])
    text = replace_once(text,
        "The accepted\nfigures `original_baseline_rmse` and `original_xgboost_geographic_rmse` show the\ncomplete method and group patterns, including all seed values and group counts.",
        "Figures 1 and 2 show the complete method and group patterns, including\nall seed values and group counts.")
    text = replace_once(text,
        "therefore Table 2 and `posthoc_guard_sensitivity` are explicitly post-hoc and",
        "therefore Table 2 and Fig. 3 are explicitly post-hoc and")
    text = replace_once(text,
        "`geographic_rmse` figure.",
        "saved geographic RMSE plot; Fig. 4 shows the support panels.")
    text = replace_once(text,
        "seed values, and support counts.",
        "seed values, and support counts; Fig. 5 shows the overall matched result.")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("Figure-placed manuscript and manifest must be new")
    if sha256(SOURCE) != SOURCE_SHA256:
        raise ValueError("V3 manuscript source changed")
    figures = verified_figures()
    text = place(SOURCE.read_text(encoding="utf-8"), figures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(text)
    manifest = {
        "status": "FIGURES_PLACED_FROM_AUDITED_ASSETS_NO_REPLOT",
        "source_sha256": sha256(SOURCE),
        "output_sha256": sha256(args.output),
        "figures": [{key: value for key, value in fig.items() if key != "anchor"}
                    for fig in figures],
        "model_training_or_inference": False,
        "plot_regeneration": False,
    }
    with args.manifest.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": manifest["status"], "figures": len(figures),
                      "output_sha256": manifest["output_sha256"]}))


if __name__ == "__main__":
    main()
