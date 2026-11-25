# R10 evidence-to-claim and manuscript-skeleton report

Date: 2026-10-14

## Completed

- Added a canonical ten-entry evidence-to-claim registry in JSON and a readable
  summary. Six entries identify supported or explicitly qualified findings;
  four block recommendation/field utility, agronomic knowledge, end-to-end
  privacy/SecAgg and novelty/cross-paper-superiority claims.
- Added a Paper-2-only manuscript skeleton aligned to the active conditional
  harvest regression task. Paper 1 and the open LaTeX document were untouched.
- Added a fail-closed registry validator. It checks unique IDs, dispositions,
  mandatory qualifiers, existence of every enabling evidence path and absence
  of enabling evidence for blocked claims.

Command:

```text
.venv\Scripts\python.exe -m paper2.restart.validate_claim_matrix --output paper2/results/restart_claim_matrix_validation_v1
```

Result: PASS; 10 claims checked, four blocked claim classes retained.

## Important boundary

The older `closest_paper_comparison.md` and `method_specification.md` describe
the superseded classification/KG-FCR direction. They remain historical records
and are not valid as the active regression paper's final related-work or method
sections. The skeleton flags that review rather than silently reusing them.

## Next action

Update the closest-paper review for conditional agricultural regression and
federated yield prediction using primary sources, then draft the manuscript
sections against the validated claim registry. Final novelty language remains
blocked until that review is complete.
