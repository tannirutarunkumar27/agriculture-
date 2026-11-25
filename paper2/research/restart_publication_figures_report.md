# R10a audited figure pipeline and unnoised-study figures

2026-10-14. R10a unnoised figure subtask VERIFIED_COMPLETE; overall R10 remains
IN_PROGRESS. The R14 matched-noise batch is still running independently.

Implemented `restart/publication_figures.py` and five regression/smoke tests in
`restart/test_publication_figures.py`. The pipeline runs a fresh COMPLETE cached
study audit and summary before extracting any plot values. It rejects partial
matrices, mismatched seed means/SD, incomplete geography and incorrect privacy
scope. No training or model reinference. The original tested aggregation is
unchanged. All plotted means, SD, seed estimates and support counts are saved in
`plot_values.json`; captions and asset/source hashes are retained.

Current accepted artifacts: `results/restart_public_refit_figures_v2/`:

- `rmse_overview`: equal-geographic-group versus pooled RMSE, five individual
  seed estimates and descriptive mean/SD; both panels share an x-axis range.
- `geographic_rmse`: all eight holdouts, sample counts and descriptive seed SD.
- `support_rmse`: seen/unseen species and composition panels, explicit system
  and geographic-group support; shared RMSE scale across all four panels.

Each figure has PNG, vector PDF and editable SVG versions. Fresh audit at
`verified_analysis/audit/verification.json` is PASS_COMPLETE_REFIT:40/40,zero
failed attempts. The figures describe the completed unnoised R13 study only.
R14 matched/noisy plots require its complete audit; no partial noisy scores have
been plotted or used for tuning.

Visual review inspected all three v2 PNGs: titles/axes/counts readable, error
bars visible, all seed markers within limits, no cropped labels. PDF/SVG exports
come from the same Matplotlib figures; separate PDF rendering was not performed.
An initial v1 overview had a seed marker clipped by automatic x limits and
excessive single-category vertical spread. Fixed explicit bounds and added a
targeted axis-visibility/shared-scale test. V1 is preserved but superseded.
All five final tests PASS,zero failures/errors at
`results/restart_publication_figure_tests_v3`. Earlier four-test runs also passed;
the visual issue was found by image inspection, not those earlier assertions.

Commands executed using the existing `.venv/Scripts/python.exe`:

```text
-m paper2.restart.test_publication_figures --output paper2/results/restart_publication_figure_tests_v1
-m paper2.restart.publication_figures --study refit --output paper2/results/restart_public_refit_figures_v1
-m paper2.restart.test_publication_figures --output paper2/results/restart_publication_figure_tests_v2
-m paper2.restart.publication_figures --study refit --output paper2/results/restart_public_refit_figures_v2
-m paper2.restart.test_publication_figures --output paper2/results/restart_publication_figure_tests_v3
```

Installed Matplotlib3.11.1 was reused; no dependencies installed. Raw data,
historical model outputs, frozen code/packages, Paper1 and open LaTeX untouched.
Error bars are descriptive seed SD, not regional confidence intervals. Unseen
support is265 systems from Auzeville only, not eight-region unseen validation.
The figures do not establish agricultural utility or application privacy.

At 05:34 UTC inspection, the original R14 process/session69296 remained active;
92/160 completion files had been observed and no FAILED attempt journals were
found. This is a live execution snapshot, not a new independent audit (only the
four-run starter has its completed independent R14 audit). No duplicate writer
or training restart was launched this session. Its progress.json intentionally
remains at the starter count until the entire active batch exits.

Next: when the R14 writer exits, run the full independent audit and matched
summary as recorded in `restart_matched_noise_execution_report.md`. Then run:

```text
-m paper2.restart.publication_figures --study matched --output paper2/results/restart_matched_noise_figures_v1
```

Use new versioned paths if outputs already exist. Visually inspect the five-
variant figures before acceptance; the tests cover schema/numerical guards but
do not certify the eventual five-variant layout. Remaining R10 work includes
baseline-study figures, manuscript synthesis and reproducibility materials.
Agronomic/ranking validity, certified privacy/SecAgg and remote infrastructure
gates remain unchanged; see KNOWN_ISSUES.md.
