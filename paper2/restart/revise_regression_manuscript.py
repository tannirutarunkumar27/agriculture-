"""Create a scoped Paper 2 manuscript revision without changing archived v1.

This is a deterministic text/evidence transformation, not model inference.
It refuses source drift and output overwrite. The statistical paragraph uses
the already audited post-hoc geographic sensitivity JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "paper2/manuscript/PAPER2_FULL_MANUSCRIPT_DRAFT.md"
SENSITIVITY = ROOT / "paper2/results/reviewer_sensitivity_v1/analysis.json"
BIBLIOGRAPHY = ROOT / "paper2/manuscript/references.bib"
EXPECTED_SOURCE_SHA256 = "313df73e537732d426aaababe898d6fe3edea5c22707bc40adc2e681b2d17f77"

URL_TO_REFERENCE = {
    "https://proceedings.mlr.press/v54/mcmahan17a.html": 1,
    "https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html": 2,
    "https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html": 3,
    "https://doi.org/10.1109/DELCON54057.2022.9752836": 4,
    "https://doi.org/10.1016/j.eswa.2023.122847": 5,
    "https://doi.org/10.1002/spe.3416": 6,
    "https://doi.org/10.1109/TAI.2025.3534149": 7,
    "https://doi.org/10.1038/s41598-025-94244-2": 8,
    "https://doi.org/10.1016/j.eswa.2025.128312": 9,
    "https://doi.org/10.1007/s10791-026-10291-x": 10,
    "https://doi.org/10.1016/j.compag.2026.111851": 11,
    "https://arxiv.org/abs/1605.02065": 12,
    "https://research.google/pubs/practical-secure-aggregation-for-privacy-preserving-machine-learning/": 13,
}
FIRST_CITATION_ORDER = (1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 8, 13, 12, 14)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Expected one replacement anchor; found {count}: {old[:90]!r}")
    return text.replace(old, new, 1)


def paired_section(payload: dict) -> tuple[str, dict]:
    if set(payload["group_sizes"]) != {
        "Angers", "Auzeville", "Copenhagen+Taastrup", "Grignon", "Jyndevad",
        "Kassel", "Reading", "Sanmarcoargentano",
    } or len(payload["seeds"]) != 5:
        raise ValueError("Unexpected frozen geographic or seed contract")
    selected = [row for row in payload["paired_group_rmse_differences"]
                if row["method"] == "random_forest" and row["reference"] == "xgboost"]
    if len(selected) != 8 or len({row["group"] for row in selected}) != 8:
        raise ValueError("Expected one random-forest contrast for each group")
    selected.sort(key=lambda row: row["group"])
    for row in selected:
        if row["n_systems"] != payload["group_sizes"][row["group"]] or len(row["rmse_difference_by_seed"]) != 5:
            raise ValueError("Paired group count/seed mismatch")
        if abs(mean(row["rmse_difference_by_seed"].values()) - row["rmse_difference_mean"]) > 1e-12:
            raise ValueError("Paired difference does not match saved seeds")
    values = [row["rmse_difference_mean"] for row in selected]
    average = mean(values)
    leave_auzeville = mean(row["rmse_difference_mean"] for row in selected if row["group"] != "Auzeville")
    rf_lower = sum(value < 0 for value in values)
    lines = [
        "## E. Paired geographic comparison and statistical scope",
        "",
        "The primary comparison used the same held-out groups for all methods. To",
        "show how an aggregate ranking depends on geography, Table 5 reports the",
        "post-hoc within-group difference in RMSE between random forest and",
        "XGBoost, averaged across the five computational seeds. Positive values",
        "favor XGBoost; negative values favor random forest. These are paired",
        "descriptive contrasts, not independent regional replications.",
        "",
        "**Table 5. Post-hoc random-forest minus XGBoost RMSE by held-out group",
        "(t/ha; mean over five matched seeds).**",
        "",
        "| Held-out group | Systems | RMSE difference |",
        "| --- | ---: | ---: |",
    ]
    lines += [f"| {row['group']} | {row['n_systems']} | {row['rmse_difference_mean']:+.4f} |"
              for row in selected]
    lines += [
        "",
        f"The equal-group mean difference was {average:+.4f} t/ha. Random forest",
        f"had lower RMSE in {rf_lower} of eight groups, even though XGBoost had the",
        f"lower overall mean. Leaving Auzeville out reduced the mean difference to",
        f"{leave_auzeville:+.4f} t/ha. These analyses were conducted after outer",
        "results had been viewed; the five seeds share the same experiments and",
        "cannot be treated as five independent geographic samples. No p-value,",
        "geographic confidence interval, or equivalence claim is made. The",
        "37 source experiments, rather than 607 system rows, also define an",
        "important dependence structure. Independent comparable regions and a",
        "prospectively specified group-level estimand would be required for a",
        "confirmatory superiority test.",
        "",
    ]
    return "\n".join(lines) + "\n", {"mean_group_difference_t_ha": average,
                               "leave_auzeville_mean_t_ha": leave_auzeville,
                               "rf_lower_group_count": rf_lower}


def revise(source_text: str, sensitivity: dict) -> tuple[str, dict]:
    text = source_text
    text = replace_once(text,
        "# Geographically Grouped Federated and Centralized Regression for Conditional Harvest Prediction in Cereal-Legume Field Experiments",
        "# Geographically Grouped Evaluation of Centralized and Simulated Federated Models for Cereal–Legume Harvest Regression")
    abstract_start, abstract_end = text.index("## Abstract\n"), text.index("# I. Introduction\n")
    abstract = """## Abstract

Predicting agricultural outcomes across sites is difficult when field
experiments differ in sample size, crop composition, and measured feature
support. We evaluate centralized and simulated horizontal federated models for
retrospective conditional harvest seed-dry-matter regression on 607 recorded
cereal–legume systems from 37 experiments grouped into eight geographic
holdouts. Models use nested group-held-out selection, training-only
preprocessing, and five fixed refit seeds. In the original eight-method
comparison, XGBoost had the lowest observed equal-group RMSE (1.2461 t/ha)
and pooled RMSE (1.7335 t/ha); the evaluated neural models were much less
stable on a held-out group with unsupported input columns. A separately
frozen, later fixed-public refit reached equal-group RMSE of 1.067407 +/-
0.021053 t/ha. It was developed after the original outer results were seen;
its changed representation, weighting, and budget prevent attributing that
difference to any one factor. Under a matched aggregation-noise simulation,
increasing noise reduced predictive utility. The results
show why geographic and pooled errors, input support, and development-stage
boundaries must be reported together. They do not establish crop selection,
causal benefit, a new federated algorithm, formal privacy, or field deployment.

**Index Terms—** Agricultural regression, cereal–legume systems, geographic
validation, simulated federated learning, model support, reproducibility.

"""
    text = text[:abstract_start] + abstract + text[abstract_end:]
    text = replace_once(text,
        "The active benchmark was constructed from version 2.0.0 of a compiled dataset\nof cereal-legume field experiments.",
        "The active benchmark was constructed from version 2.0.0 of the\nGaudio et al. cereal--legume field-experiment dataset [14].")
    text = text.replace("Paper 2 uses these methods", "We use these methods")
    text = text.replace("rank Paper 2 against", "rank this study against")
    text = text.replace("in Paper 2 is", "in this study is")
    text = text.replace("whereas Paper\n2 simulates", "whereas this study simulates")
    text = text.replace("Paper 2 therefore reports", "We therefore report")
    text = text.replace("The narrower value of Paper 2 is", "The narrower value of this study is")
    text = replace_once(text, "f_{\\theta}(\\mathbf{x})=mathbf{w}_2^{\\mathsf T}",
                        "f_{\\theta}(\\mathbf{x})=\\mathbf{w}_2^{\\mathsf T}")
    section, stats = paired_section(sensitivity)
    text = replace_once(text, "## E. Summary of empirical findings\n", section + "## F. Summary of empirical findings\n")
    text = replace_once(text, "# VIII. Reproducibility\n", "# VIII. Reproducibility and Availability\n")
    text = replace_once(text,
        "Releasing a portable package would require a separate path-relocation strategy,\nfresh-environment installation test, and cache/audit validation without changing\nthe archived evidence.",
        "Releasing a portable package would require a separate path-relocation strategy,\nfresh-environment installation test, redistribution review, and cache/audit\nvalidation without changing the archived evidence.\n\nThe original version-2.0.0 source dataset is available from Zenodo [14].\nThe derived splits, source hashes, settings, cached predictions, analysis\noutputs, and figure values are retained in the local project; this repository\nhas not been published as a public reproduction package. The observed Windows\npackage versions are documented, but they are not a cross-platform dependency\nlock. Raw-data redistribution rights, a fresh-environment run, and Linux\nreproduction remain unchecked. Accordingly, the current evidence supports\nlocal auditability, not independent clean-room reproducibility.")
    text = replace_once(text,
        "This study provided a reproducible characterization of conditional cereal-\nlegume harvest regression",
        "This study provided a locally auditable comparison of conditional cereal-\nlegume harvest regression")
    text = replace_once(text,
        "Author initials, pagination, and IEEE punctuation should receive a final\nbibliographic metadata check before submission.",
        "14. N. Gaudio *et al.*, “A global dataset gathering 37 field experiments\n    involving cereal-legume intercrops and their corresponding sole crops,”\n    Zenodo, version 2.0.0, 2024, doi: 10.5281/zenodo.14160885.\n\nAuthor initials, pagination, and IEEE punctuation should receive a final\nbibliographic metadata check before submission.")
    text = replace_once(text,
        "4. “A federated learning-based crop yield prediction for agricultural",
        "4. T. Manoj, K. Makkithaya, and V. G. Narendra, “A federated learning-based crop yield prediction for agricultural")
    text = replace_once(text, "9. Nirosha and Vennila,", "9. U. Nirosha and G. Vennila,")
    text = replace_once(text, "10. Utku *et al.*,", "10. A. Utku *et al.*,")
    text = replace_once(text, "11. Koksal, Bennin, and Tekinerdogan,", "11. D. D. Koksal, K. E. Bennin, and B. Tekinerdogan,")
    body, references = text.split("# References\n", 1)
    pattern = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
    used = set()
    citation_numbers = {old: new for new, old in enumerate(FIRST_CITATION_ORDER, start=1)}
    def cite(match: re.Match[str]) -> str:
        url = match.group(2)
        if url not in URL_TO_REFERENCE:
            raise ValueError(f"Unmapped manuscript citation: {url}")
        old_number = URL_TO_REFERENCE[url]
        used.add(old_number)
        return f"{match.group(1)} [{citation_numbers[old_number]}]"
    body = pattern.sub(cite, body)
    if set(range(1, 14)) - used:
        raise ValueError("Some registered references are not cited in the body")
    reference_body, _editorial_note = references.split("\nAuthor initials, pagination", 1)
    headings = list(re.finditer(r"(?m)^(\d+)\. ", reference_body))
    reference_blocks = {
        int(match.group(1)): reference_body[match.start():headings[index + 1].start() if index + 1 < len(headings) else len(reference_body)].strip()
        for index, match in enumerate(headings)
    }
    if set(reference_blocks) != set(FIRST_CITATION_ORDER):
        raise ValueError("Reference-list identities do not match the citation registry")
    reordered = "\n".join(
        re.sub(r"^\d+\. ", f"{citation_numbers[old]}. ", reference_blocks[old], count=1)
        for old in FIRST_CITATION_ORDER
    )
    text = body + "# References\n\n" + reordered + "\n"
    text = text.replace("dataset [14]", f"dataset [{citation_numbers[14]}]")
    if "Gaudio et al. cereal--legume field-experiment dataset [14]" not in text:
        raise ValueError("Dataset citation missing")
    if "=mathbf{w}" in text or "## E. Summary of empirical findings" in text:
        raise ValueError("Source defect or old heading survived")
    if re.search(r"\bwe (propose|introduce) (a )?(knowledge-guided|privacy-preserving)", text, re.I):
        raise ValueError("Unsupported contribution wording")
    return text, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("Versioned manuscript and manifest must be new")
    if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise ValueError("Archived manuscript source changed")
    if "gaudio2024cereallegume" not in BIBLIOGRAPHY.read_text(encoding="utf-8"):
        raise ValueError("BibTeX dataset entry missing")
    sensitivity_bytes = SENSITIVITY.read_bytes()
    revised, stats = revise(SOURCE.read_text(encoding="utf-8"), json.loads(sensitivity_bytes))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(revised)
    manifest = {
        "status": "VERSIONED_SCOPED_REGRESSION_MANUSCRIPT_DRAFT",
        "source_sha256": sha256(SOURCE), "sensitivity_sha256": hashlib.sha256(sensitivity_bytes).hexdigest(),
        "bibliography_sha256": sha256(BIBLIOGRAPHY), "output_sha256": sha256(args.output),
        "paired_geographic_statistics": stats,
        "training_or_inference_performed": False,
        "portable_reproduction_claim": False,
    }
    with args.manifest.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": manifest["status"], "output_sha256": manifest["output_sha256"]}))


if __name__ == "__main__":
    main()
