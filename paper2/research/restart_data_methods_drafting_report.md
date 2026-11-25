# Paper 2 Data and Methods drafting report

Date: 2026-10-15  
Task: R10g, evidence-constrained Data and Methods sections

## Outcome

The Data and Problem Formulation and Methods sections are now drafted for the
active conditional cereal-legume harvest-regression study. The 1,850-word draft
is based on frozen data, search, outer-study, fixed-public, matched-noise, audit,
and threat-model artifacts; no completed experiment was rerun.

The data section records the 607 system-level units, target construction,
eight geographic groups and their exact support, fold-fitted and fixed-public
predictor contracts, excluded variables, retrospective scope, and disabled
ranking/knowledge claims. The methods section specifies the 56-fold nested
grouped protocol, five seeds, eight baseline methods, MLP and aggregation
equations, frozen selection/refit rules, grid-edge limitations, matched clipping
and Gaussian-noise simulation, zCDP accounting boundary, metrics, support panels,
and one-time inference/provenance safeguards.

## Verification and repaired issue

The first validation run correctly failed because its literal phrase check did
not normalize a line-wrapped fact and the draft spelled one required count as a
word. The validator was repaired to normalize whitespace, the evidence count
was written explicitly as `80 of 224`, and a regression test now covers a fact
split across lines. No scientific value or experimental result changed.

Final commands:

```powershell
& '.\.venv\Scripts\python.exe' -B paper2/restart/validate_data_methods_draft.py --draft paper2/manuscript/PAPER2_DATA_METHODS_DRAFT.md --output paper2/results/restart_data_methods_validation_v1
& '.\.venv\Scripts\python.exe' -B -m unittest paper2.tests.test_validate_data_methods_draft -v
```

Final result: PASS for 12 required headings and 14 required evidence facts;
two tests passed, zero failures/errors.

## Files

- `manuscript/PAPER2_DATA_METHODS_DRAFT.md`
- `restart/validate_data_methods_draft.py`
- `tests/test_validate_data_methods_draft.py`
- `results/restart_data_methods_validation_v1/verification.json`
- `manuscript/PAPER2_MANUSCRIPT_SKELETON.md` (draft pointers only)

No training, inference, raw-data edit, dependency installation, Paper 1 edit,
open-LaTeX edit, credential use, network exposure, or cloud operation occurred.

## Next action

Draft the Results section from the accepted original-baseline, post-hoc guard,
fixed-public refit, and matched-noise artifacts. Preserve exact study separation,
equal-group versus pooled metrics, adverse group/R2 findings, support counts,
seed-SD interpretation, and the simulation-only privacy boundary.
