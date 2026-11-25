"""Generate a comparison from canonical saved predictions.

Recorded robustness is exported separately: historical checkpoints/protocols
must not be silently joined with the corrected probability evidence.
Missing models/metrics remain missing rather than being invented or refitted.
"""
import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.experiments.publication_evidence import rebuild_calibration
from src.utils.review_figures import comparison_plot


def main():
    table, _ = rebuild_calibration()
    table["Evaluation"] = "Direct saved-model probabilities; processed test partition N=330"
    table.to_csv(ROOT / "outputs/tables/comprehensive_comparison.csv", index=False)
    (ROOT / "outputs/tables/comprehensive_comparison_latex.txt").write_text(
        table.to_latex(index=False, float_format=lambda value: f"{value:.6f}"), encoding="utf-8")
    report_path = ROOT / "outputs/robustness/full_robustness_report.json"
    if report_path.exists():
        recorded = json.loads(report_path.read_text(encoding="utf-8"))
        rows = []
        for experiment, models in recorded.items():
            if not isinstance(models, dict):
                continue
            for model, settings in models.items():
                if not isinstance(settings, dict):
                    continue
                for setting, accuracy in settings.items():
                    if not isinstance(accuracy, (int, float)):
                        continue
                    rows.append({"experiment": experiment, "recorded_model": model,
                                 "setting": setting, "accuracy": accuracy,
                                 "source": str(report_path.relative_to(ROOT)),
                                 "provenance": "Historical recorded experiment; not the new compact-model study"})
        pd.DataFrame(rows).to_csv(ROOT / "outputs/tables/recorded_robustness_separate_protocol.csv", index=False)
    comparison_plot()
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
