# Reviewer-issue resolution

The code defects and reporting pipelines have been repaired or verified. This is **not** a claim that all twelve scientific/editorial objections are resolved, that the uploaded PDF has changed, or that the manuscript is submission-ready.

## Issue-by-issue status

| Review issue | Code/evidence response | Remaining limitation |
|---|---|---|
| 1. Cancelling adaptive gate | Verified forward computation is `LayerNorm(g * concat + beta * context)`; gate changes representations, receives gradients and updates. Regression tests cover these properties. | The manuscript's old detached-term equation must be replaced. |
| 2. Constant knowledge bias | Verified per-sample query attention over encoded crop profiles, followed by context projection. Distinct samples produce distinct attention/context; tests cover sensitivity and gradients. | This is learned compatibility attention, not a direct physical constraint on sensor values. Update the manuscript mathematics accordingly. |
| 3. Calibration contradiction | Tables, reliability bins and labels now derive from the same validated saved probability arrays, with source hashes. Corrected-checkpoint KG ECE is **0.121839**, not a manually imposed 0.130556 or 0.0048. | Update abstract/table text and rebuild the PDF. The new study's ECE is a different protocol and must not be substituted into the historical table. |
| 4. IEEE placeholders | No code-based scientific repair applies. | Verify real authors/affiliations, submission metadata and figure references in the actual manuscript/PDF. No identities or DOI were invented. |
| 5. Placeholder references | No fabricated bibliographic replacements were made. | Every cited work must be identified and verified against its real publication. |
| 6. Simpler models perform better | Ran seven model variants over five seeds with shared partitions/backbones and a controlled CE objective. Saved accuracy, macro-F1, ECE, Brier, NLL, parameter counts and paired comparisons, including Holm adjustments. | Results do **not** establish an accuracy advantage for the full model over no-K or XGBoost. Repeated splits overlap and N=5 has limited power. |
| 7. Unreproducible prior/source mapping | Exported the exact active 22x7 prior, crop/feature order, code location and provenance limitations. Distinguished manual weights from the separately defined range knowledge base. | A verified institutional per-cell derivation and deterministic range-to-weight rule are **not available**; the weights must be described as manually specified. |
| 8. Unconstrained learned K | Added/verified explicit fixed and anchored-trainable policies, persisted K0, mean-squared anchor loss and drift diagnostics. New study uses lambda=1 for anchored models and confirms unchanged fixed matrices. | L2 anchoring is soft, not hard interval clipping. Older checkpoint metadata does not confirm anchoring; new code cannot retroactively change old runs. |
| 9. Excessive embedding size | New study uses a compact vocabulary containing all actual terminal nodes, including unvisited leaves. Mean full gated count is **1,741,638**, rather than approximately 15.5M. Unknown nodes fail rather than being clipped. | Compact study checkpoints have their own booster/vocabulary/scaler. They have not been silently substituted into production or promoted as field-validated models. |
| 10. Other models more robust | Figure/table generators use recorded robustness evidence and remove hard-coded superiority annotations. Historical robustness is exported separately from corrected clean/calibration results. | No universal robustness advantage is established. The new compact models need their own stress/field evaluations before making new robustness claims. |
| 11. Early exit bypasses hybrid | Exported threshold-by-threshold exit/fallback counts, route accuracies, direct-model accuracy and cascade accuracy. Production records its chosen route; raw sensors are standardized exactly once before inference/imputation. | Deterministic saved-model route evidence excludes MC Dropout and latency. Do not attribute XGBoost exits to the hybrid or interpret this audit as a production benchmark. |
| 12. Unclear contribution | New controlled comparison separates knowledge policies and forward gating from auxiliary-loss effects. Removed fabricated feature-complementarity/SHAP examples from affected figure generators. | A scientific contribution cannot be manufactured by code. Narrow the claims to what the evidence establishes; the optional shuffled-K control changes policy as well as crop order and is not a pure one-factor contrast. |

## New controlled comparison

Protocol: five seeds (42, 123, 2024, 3407, 9999), stratified 70/15/15 partitions, training-only scaler fitting, at most 25 epochs, validation-NLL early stopping, CE plus lambda=1 knowledge anchor where applicable. Each hybrid starts with the same seed-specific booster, pretrained FT backbone, leaf-embedding initialization and classifier initialization. These are repeated held-out splits, not independent field trials.

| Model | Mean accuracy | Mean macro-F1 | Mean ECE | Mean Brier | Mean NLL | Mean trainable parameters |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 99.5152% | 0.995124 | 0.018456 | 0.010562 | 0.031415 | Non-neural; tree counts separately recorded |
| FT-Transformer | 98.6061% | 0.986095 | 0.009981 | 0.022274 | 0.042556 | 212,054 |
| Hybrid without K or gate | 99.5758% | 0.995753 | 0.005071 | 0.006200 | 0.011589 | 1,666,668 |
| Hybrid with fixed K, no gate | 99.5152% | 0.995144 | 0.005042 | 0.006648 | 0.011522 | 1,692,012 |
| Hybrid with anchored K, no gate | 99.4545% | 0.994535 | 0.007033 | 0.007597 | 0.012481 | 1,692,166 |
| Hybrid with anchored K and gate | 99.4545% | 0.994558 | 0.004269 | 0.008564 | 0.015230 | 1,741,638 |
| Hybrid with shuffled fixed K and gate | 99.0909% | 0.990895 | 0.005901 | 0.014158 | 0.026652 | 1,741,484 |

The full gated variant has the lowest mean ECE in this study, but it is not the most accurate. Its ECE improvement over XGBoost is not significant after the recorded familywise adjustment (paired t Holm p=0.201748; Wilcoxon Holm p=1). Do not claim that knowledge or gating is necessary for clean accuracy.

Detailed mean/SD results: `../knowledge_comparison_review_fixes/summary_flat.csv`. Checkpoints, probability arrays, sample IDs, learned K matrices, partitions, training-only scaler statistics and run manifest are retained under the same experiment directory.

## Historical evidence and routing

- Complete recovered CV: six models, five folds, 13,200 OOF predictions. Sample IDs, raw labels, shared folds, metrics and confusion matrices were checked. Historical Decision Tree depth=6 is a retained limitation.
- At threshold 0.90 on the canonical healthy-input saved-model evidence, XGBoost exits **317/330 (96.0606%)**; hybrid fallback is **13/330**. Cascade accuracy is **99.6970%**, direct XGBoost **99.0909%**, and direct corrected hybrid **99.6970%**. No latency or MC Dropout claim follows from these numbers.
- Historical saved sklearn objects emit a version warning (saved with 1.8.0, loaded with 1.9.0). Numerical evidence was checked; retraining/version-pinning is preferable for future reproducible releases.
- Existing manuscript text and bibliography were not rewritten under this code-only request. Regenerated publication assets were synchronized; the old PDF remains unchanged.

## Verification

- **99/99** regression tests passed, including forward gate/knowledge behavior, fixed/anchored priors, compact leaf mapping, calibration consistency, OOF integrity, raw-input scaling, multiplicity adjustment and atomic artifact replacement.
- **8/8** API integration checks passed, including normal prediction, missing sensors, schema validation and batch prediction. Missing API dependencies were added to requirements and installed in the project environment with approval.
- Independently validated **35/35 model-seed results and 11,550/11,550 held-out predictions**, including raw sample labels, exact shared split identities, training-only scaler statistics, saved metrics and prior drift.
- Source compilation succeeded. Publication tables/figures use empirical sources; output is published atomically to tolerate Windows preview readers.

## Reproduce without retraining

From `KG-Hybrid` in PowerShell:

```powershell
& ..\.venv\Scripts\python.exe -m unittest discover -s tests -v
& ..\.venv\Scripts\python.exe src/api/test_api.py
& ..\.venv\Scripts\python.exe src/experiments/validate_knowledge_comparison.py
& ..\.venv\Scripts\python.exe src/experiments/publication_evidence.py
& ..\.venv\Scripts\python.exe src/utils/generate_paper_ready_package.py
& ..\.venv\Scripts\python.exe src/utils/generate_paper_tables_and_plots.py
```

To train another comparison, select a **new empty output directory** with `src/experiments/run_knowledge_comparison.py --output-dir ... --epochs 25`. The runner refuses to overwrite completed evidence. Do not reuse historical result names for new protocols.
