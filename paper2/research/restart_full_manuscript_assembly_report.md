# Paper 2 full manuscript assembly report

Date: 2026-10-15  
Task: R10j, full evidence-bound Markdown draft assembly

## Outcome

The validated section components were assembled into a 6,998-word Paper-2-only
manuscript draft with an evidence-constrained title, abstract, index terms,
Sections I--IX, and a 13-entry working reference list. The title does not claim
knowledge guidance, crop recommendation, or privacy preservation. The abstract
contains the verified original, fixed-public, and matched-noise results and
states the post-hoc and simulation-only boundaries.

The assembly manifest binds six component hashes and all nine accepted PDF
figure hashes. The validator requires exactly one ordered copy of Sections I
through IX, every registered closest-study and algorithm source URL, the zCDP
and secure-aggregation primary records, abstract facts, unchanged component
hashes, and unchanged figure assets.

## Verification

```powershell
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.assemble_full_manuscript --root . --manuscript-dir paper2/manuscript --output paper2/manuscript/PAPER2_FULL_MANUSCRIPT_DRAFT.md --manifest paper2/results/restart_full_manuscript_v1/manifest.json
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.validate_full_manuscript --draft paper2/manuscript/PAPER2_FULL_MANUSCRIPT_DRAFT.md --manifest paper2/results/restart_full_manuscript_v1/manifest.json --registry paper2/research/restart_regression_closest_work.json --root . --manuscript-dir paper2/manuscript --output paper2/results/restart_full_manuscript_validation_v1
& '.\.venv\Scripts\python.exe' -B -m unittest paper2.tests.test_validate_full_manuscript -v
```

Assembly: six components and nine figures. Validation: PASS, nine ordered
sections, 11 registered literature/algorithm sources, zero errors. Two tests
passed, including rejection of a privacy-preserving title.

## Files

- `manuscript/PAPER2_FULL_MANUSCRIPT_DRAFT.md`
- `manuscript/PAPER2_FRONT_MATTER.md`
- `manuscript/PAPER2_REFERENCES.md`
- `restart/assemble_full_manuscript.py`
- `restart/validate_full_manuscript.py`
- `tests/test_validate_full_manuscript.py`
- `results/restart_full_manuscript_v1/manifest.json`
- `results/restart_full_manuscript_validation_v1/verification.json`

No training/inference, raw-data modification, dependency installation, Paper 1
or open-LaTeX edit, credentials, network exposure, or cloud operation occurred.

## Remaining publication work

This is a complete evidence-bound Markdown draft, not a submission-ready IEEE
source. Final work includes authoritative bibliographic metadata normalization,
author/affiliation and acknowledgement content supplied by the researchers,
journal-specific LaTeX conversion and page/layout decisions, main-versus-
supplementary figure selection, and a clean-room reproducibility-package check.
