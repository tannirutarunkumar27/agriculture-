# Paper 2 Introduction and Related Work drafting report

Date: 2026-10-14  
Task: R10f, evidence-constrained first manuscript sections

## Outcome

The Introduction and Related Work are now drafted for the active conditional
cereal-legume harvest-regression study. The 1,271-word draft follows the
validated claim matrix and the regression closest-work registry rather than the
historical classification/KG-FCR direction.

The Introduction defines the observed-system regression target, 607 systems,
eight geographic groups, three research questions, and five reproducibility/
evaluation contributions. It explicitly excludes best-crop selection, causal
yield improvement, a novel optimizer, validated knowledge guidance, external
numeric superiority, field utility, and formal production privacy.

Related Work covers the original FedAvg, FedProx, and FedNova sources; all eight
registered agricultural studies; grouped/support-aware interpretation; and the
separation between federated optimization, differential privacy, secure
aggregation, and the project's matched noise-utility simulation.

## Files and verification

- Draft: `manuscript/PAPER2_INTRODUCTION_RELATED_WORK_DRAFT.md`
- Validator: `restart/validate_manuscript_sections.py`
- Result: `results/restart_manuscript_sections_validation_v1/verification.json`
- Skeleton updated only with pointers to the draft; Paper 1 remains unchanged.

Executed:

```powershell
& '.\.venv\Scripts\python.exe' -B paper2/restart/validate_manuscript_sections.py --draft paper2/manuscript/PAPER2_INTRODUCTION_RELATED_WORK_DRAFT.md --registry paper2/research/restart_regression_closest_work.json --output paper2/results/restart_manuscript_sections_validation_v1
```

Result: PASS; seven required headings, eight study links and three algorithm
links verified, zero errors. No training, inference, data change, dependency
installation, Paper 1 edit, or LaTeX edit occurred.

## Next action

Draft Data and Problem Formulation plus Methods directly from frozen protocols
and audited results. Numerical values must come from verified artifacts, and
the post-hoc guard and simulation-only privacy stages must remain separate.
