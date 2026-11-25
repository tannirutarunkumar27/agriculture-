# Paper 2 regression literature update report

Date: 2026-10-14  
Task: R10e, regression-focused closest-work comparison

## Outcome

The active Paper 2 literature basis now matches the implemented conditional
cereal-legume harvest dry-matter regression task. The earlier KG-FCR
classification comparison remains available as historical evidence but is
explicitly marked as inactive for manuscript positioning.

Eight primary-source studies were recorded in a machine-readable registry.
They cover horizontal soybean-yield federation, localized/pruned forecasting,
centralized and decentralized FL topologies, edge/cloud recurrent prediction,
Earth-observation agricultural proxy prediction, graph-recurrent yield models,
multi-country cereal forecasting, and vertical crop-yield federation. Original
FedAvg, FedProx, and FedNova sources are separately retained.

The review establishes a negative but necessary result: federated crop-yield
prediction, heterogeneous agricultural clients, recurrent/hybrid models, and
local data retention cannot be claimed as new. No inspected study is a matched
numeric benchmark for Paper 2 because targets, units, client definitions,
partitions, metrics, or budgets differ. The registry therefore prohibits
cross-paper numerical superiority claims for every listed study.

## Paper 2 positioning retained

Paper 2 is restricted to a reproducible evaluation contribution: conditional
harvest regression across held-out experimental groups, simulated horizontal
site federation, centralized/federated baselines, equal-group and pooled
metrics, support panels, adverse findings, explicit separation of the post-hoc
guard, and matched aggregation-noise utility simulation. The evidence does not
establish best-crop recommendation, independently validated agronomic guidance,
field utility, a novel federated optimizer, or production privacy.

## Files

- `research/restart_regression_closest_work.md`
- `research/restart_regression_closest_work.json`
- `restart/validate_regression_literature.py`
- `tests/test_restart_regression_literature.py`
- `results/restart_regression_literature_validation_v1/verification.json`

## Verification

Executed from the repository root with the existing environment:

```powershell
& '.\.venv\Scripts\python.exe' -B paper2/restart/validate_regression_literature.py --registry paper2/research/restart_regression_closest_work.json --output paper2/results/restart_regression_literature_validation_v1
& '.\.venv\Scripts\python.exe' -B -m unittest paper2.tests.test_restart_regression_literature -v
```

Registry validation: PASS, eight studies, three algorithm sources, zero errors.
Scoped tests: three passed, zero failed. No training, inference, raw-data edit,
dependency installation, Paper 1 edit, or LaTeX edit occurred.

## Next action

Draft the Paper 2 Related Work section from this registry and the Introduction
from the validated evidence-to-claim matrix. Keep external results clearly
attributed and qualitative; do not transform incomparable scores into a ranking.
