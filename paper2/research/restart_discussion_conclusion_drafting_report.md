# Paper 2 final narrative sections report

Date: 2026-10-15  
Task: R10i, Discussion through Conclusion

## Outcome

Discussion, Limitations and Threats to Validity, Reproducibility, and Conclusion
are now drafted in 1,728 evidence-constrained words. The discussion interprets
the original support failure, post-hoc guard, unequal metric weighting,
fixed-public stage, and matched aggregation-noise utility without collapsing
them into a single ranking. Published agricultural FL work is compared only
qualitatively because targets and protocols differ.

The limitations retain dataset imbalance, one-resource development scope,
retrospective treatment bundles, missing cultivar/density comparability, prior
outer exposure, fixed-grid/cap hits, dependent folds/seeds, local CPU simulation,
absence of real institutional federation, large conditional epsilon bounds,
no secure aggregation, disabled knowledge/ranking, and no field/economic claim.

The reproducibility section records all completed study counts, frozen evidence,
once-only inference policy, audits, nine figure packages, and current Windows-
local/nonportable boundary. The conclusion is restricted to characterization of
conditional harvest regression and controlled noise utility.

## Verification

```powershell
& '.\.venv\Scripts\python.exe' -B paper2/restart/validate_discussion_conclusion_draft.py --draft paper2/manuscript/PAPER2_DISCUSSION_CONCLUSION_DRAFT.md --output paper2/results/restart_discussion_conclusion_validation_v1
& '.\.venv\Scripts\python.exe' -B -m unittest paper2.tests.test_validate_discussion_conclusion_draft -v
```

Result: PASS for eight required headings and 16 evidence/scope facts; two tests
passed, zero failures/errors. A regression test rejects an affirmative end-to-
end privacy guarantee.

## Files

- `manuscript/PAPER2_DISCUSSION_CONCLUSION_DRAFT.md`
- `restart/validate_discussion_conclusion_draft.py`
- `tests/test_validate_discussion_conclusion_draft.py`
- `results/restart_discussion_conclusion_validation_v1/verification.json`
- `manuscript/PAPER2_MANUSCRIPT_SKELETON.md` (draft pointers only)

No model training/inference, raw-data edit, dependency installation, Paper 1 or
open-LaTeX edit, credential use, network exposure, or cloud operation occurred.

## Next action

Assemble the validated sections into one Paper-2-only manuscript draft, add an
evidence-constrained title and abstract, consolidate references, validate every
claim/figure/table source, and then perform reproducibility-package checks.
