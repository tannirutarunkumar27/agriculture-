# Table-to-figure replacements

Original mapping refers to IEEE_Access_Tarun (4).pdf. Tables/figures will be
renumbered after editing the LaTeX source.

| Original table | Figure stem | Recommended width |
| --- | --- | --- |
| IX: registered neural parameters | table_ix_registered_parameters | two columns |
| X: zero replacement | table_x_feature_masking | one column |
| XI: imputation comparison | table_xi_imputation | one column |
| XII: Gaussian noise | table_xii_gaussian_noise | one column |
| XIII: feature scaling | table_xiii_feature_scaling | one column |
| X + XII + XIII combined alternative | combined_robustness_three_panel | two columns |

Use the combined robustness figure OR its three individual counterparts, not
both. Keep the imputation figure separate because it uses a different checkpoint.
Remove the duplicated Table VI (same data as X), and Table VII (summary of XII and
XIII), and replace their references with references to the retained figures.
Keep Tables IV and V for exact classification and calibration values, and Table
VIII for route counts. Tables I-III can also remain tables.

PDF is the preferred vector input to LaTeX. PNG files are 600 DPI; SVG files are
editable vector exports. Preview PNGs are only for quick inspection.

Use replacement_figures.tex for captions, labels, and placement. Put this folder
beside the main .tex file, update any former Table references, and retain the
classification/calibration tables. Existing manuscript files are not edited.

All plots use existing evidence, without training or invented observations.
Hybrid parameter counts are independently aggregated from the five seed JSON
files, using sample SD (ddof=1); counts exclude the external XGBoost ensemble.
The old robustness report labels scaling as cross_dataset internally; the new
figures intentionally call it standardized feature scaling, not external data.
The imputation CSV's separate XGBoost column is omitted to preserve the
within-model comparison in Table XI. Robustness trial-level SDs are not available,
so these plots contain no fabricated uncertainty bars. Lines connect recorded
operating points; they are not estimates from additional experiments.

Sources and SHA-256 hashes are in plot_source_data.json and validation_manifest.json.
Reproduce from the repository root:

    .venv\Scripts\python.exe KG-Hybrid\src\utils\generate_table_replacement_figures.py
