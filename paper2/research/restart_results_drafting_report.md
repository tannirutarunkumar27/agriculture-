# Paper 2 Results drafting report

Date: 2026-10-15  
Task: R10h, evidence-constrained Results section

## Outcome

The 1,530-word Results section is complete and separates four distinct stages:
the original prospectively frozen eight-method outer study, the exposed-test
post-hoc zero-support sensitivity, the later fixed-public unnoised refit, and
the prospectively matched clipping/noise utility simulation.

The draft contains the exact primary and pooled metrics, geographic counts and
R2 findings, support-panel counts, post-hoc qualifier, clipping/noise results,
weak-fit counts, event-ledger size and conditional accounting bounds. It maps
the narrative to all nine accepted figures without treating the stages as one
matched ranking. Seed SD remains descriptive and all adverse findings remain.

## Verification and repaired issue

The first gate rejected the phrase "universal unseen-species capability" even
though the draft said that such an interpretation must not be made. The check
was narrowed to reject the affirmative claim "establishes universal unseen-
species capability." This was a validator false positive; no result or claim
was weakened. The existing test rejecting an unsupported statistical-
superiority sentence remained in place.

Final commands:

```powershell
& '.\.venv\Scripts\python.exe' -B paper2/restart/validate_results_draft.py --draft paper2/manuscript/PAPER2_RESULTS_DRAFT.md --output paper2/results/restart_results_validation_v1
& '.\.venv\Scripts\python.exe' -B -m unittest paper2.tests.test_validate_results_draft -v
```

Final result: PASS for six required headings and 17 evidence facts; two tests
passed, zero failures/errors.

## Files

- `manuscript/PAPER2_RESULTS_DRAFT.md`
- `restart/validate_results_draft.py`
- `tests/test_validate_results_draft.py`
- `results/restart_results_validation_v1/verification.json`
- `manuscript/PAPER2_MANUSCRIPT_SKELETON.md` (draft pointer only)

No model training or inference, raw-data edit, dependency installation, Paper 1
edit, open-LaTeX edit, credential use, network exposure, or cloud action occurred.

## Next action

Draft Discussion, Limitations and Threats to Validity, Reproducibility, and
Conclusion. Compare published studies only qualitatively, explain metric/group
weighting and support limitations, and retain prior exposure, grid-edge,
simulation-only privacy, missing agronomic/ranking evidence, and absent remote
deployment as explicit limitations.
