# Replacements for the three reviewed figures

These figures were generated from saved numerical evidence, not AI image
generation or manually drawn data. No models were retrained. Existing
manuscript text, PDFs, tables, and older images were left unchanged.

| Replace | Upload this PNG | Protocol |
| --- | --- | --- |
| Figure 1 | fig1_reliability_saved_checkpoint.png | Designated 330-row saved-checkpoint probability evaluation; ten equal-width bins; unrestricted Random Forest |
| Figure 2 | fig2_confusion_oof_normalized.png | Completed noncompact five-fold CV; 2,200 OOF predictions; 100 per class; row normalization |
| Figure 3 | fig3_risk_coverage_direct_hybrid.png | Direct corrected noncompact KG-Hybrid confidence selection on 330 rows; not XGBoost routing |

The PNG files are exported at 600 dpi. SVG versions are supplied for vector
editing/export; standard LaTeX can use the PNGs directly without an SVG package.

Upload this folder beside your main LaTeX file and replace the three old
figure environments with those in replacement_figures.tex. Do not append
duplicate environments. The supplied labels match the pasted results section:
fig:reliability, fig:confusion, and fig:risk_coverage. Wide figure environments
are used to keep labels readable in a two-column paper; adapt float placement
to your journal template. Replace the hard-coded Table V reference with your
existing table label if available.

Checks performed by the generator:

- Reliability ECE, Brier, and NLL match the independently recalculated saved
  probability table to six decimal places. All 330 records belong to bins.
- OOF sample identities, raw labels, shared stratified folds, per-fold metrics,
  and confusion matrices are validated using the existing CV validation code.
- The pooled confusion matrix agrees exactly with the independently exported
  matrix, with 2,185 correct classifications and 15 errors.
- Risk at 100% coverage equals the direct hybrid error rate: 1/330, or
  approximately 0.303%. Confidence ties are retained together.

The CSV files retain plotted values, and figure_validation_manifest.json
records input hashes and validation results. Reproduce from KG-Hybrid with:

```powershell
& ..\.venv\Scripts\python.exe -m src.utils.generate_updated_manuscript_figures
```

The uploaded journal PDF is not rebuilt by this script. Replace its image
assets and recompile the actual paper source to remove the old figures.
