# Step 4: knowledge module implemented; scientific gate unresolved

Date: 2026-10-13. Engineering implementation and verification completed;
real-data knowledge-guided training was NOT executed. Stage G remains partial,
not agriculturally validated. No privacy mechanism was added.

## Completed

- Class-aligned soft suitability/logit-bias module with original-unit inputs,
  original-observation masks, per-row scope metadata and missing-rule neutrality.
- Strict declaration checks for dataset/class/feature binding, unit/measurement/
  period compatibility, independent archived source references and review.
- Empty approved bank blocks nonzero alpha. Existing manually specified Paper 1
  priors were not promoted to scientific evidence.
- Explicit multi-input local training adapter, plain SGD and optional FedProx
  penalty, reusing the existing three aggregation algorithms unchanged.
- Schema identity binds public rules, class order, alpha, neural layout and
  preprocessing. Mismatched client updates reject through existing aggregation.
- Selected-weight save/load with identity and integrity checks, overwrite
  rejection and exact synthetic prediction roundtrip. This is not resumable
  round/RNG/optimizer state; the real experiment runner remains future work.
- Mathematical specification, complexity, boundary conditions and prospective
  ablation design saved. The ablation draft is NOT frozen or executed.
- Canonical/Maharashtra source identities and pending numerical feature
  contracts recorded. No units were inferred from column names or distributions.

## Files

Created: `knowledge.py`, `knowledge_training.py`,
`configs/knowledge_registry_v1.json`, `scripts/check_knowledge_gate.py`,
`tests/test_knowledge.py`, `research/knowledge_module_specification.md`, this
report and generated `results/knowledge_gate_v1/gate_report.json`.

Updated: STATUS, README, provisional method specification and agronomic-source
check log. All five frozen baseline source files and frozen protocol/runner
identities remain unchanged. Original raw datasets, earlier research outputs,
Paper 1 and the user's open LaTeX document were preserved.

## Executed commands and outcomes

```powershell
# Initial focused suite: 13 tests passed before three additional checks:
& '.\.venv\Scripts\python.exe' -B -m unittest discover -s paper2/tests -p test_knowledge.py -v
# Current real-data eligibility, no training/evaluation:
& '.\.venv\Scripts\python.exe' -B paper2/scripts/check_knowledge_gate.py --output paper2/results/knowledge_gate_v1
# Final full suite: 96 passed, zero failed, including 16 knowledge tests:
& '.\.venv\Scripts\python.exe' -B paper2/scripts/verify_implementation.py --output paper2/results/tests_v10
# Existing frozen baseline integrity:
& '.\.venv\Scripts\python.exe' -B paper2/scripts/freeze_baseline_protocol.py --output paper2/results/frozen_baseline_protocol_v1 --verify-only
```

Full suite exit code 0, PASS; saved log/verification in `results/tests_v10`.
Tests include trapezoid values, confidence-weighted rules, class permutation,
missing/out-of-scope neutrality, imputed-value exclusion, evidence/unit/scope
rejection, alpha=0 exact baseline update equivalence, guidance changing the
supervised update, deterministic local updates with unchanged global model,
malformed settings/labels, schema mismatch and saved-weight tamper rejection.
Two-round, two-client tests exercised all three algorithms with unequal step
counts and exact saved/restored predictions. Rules/labels/approvals in those
tests are explicitly synthetic software fixtures, not agricultural observations.
No synthetic accuracy is reported as a research result.

Frozen verification PASS: 235 runs, 45 partitions, original protocol digest
`01964d81236d1f97924bb20b840520939f77d89064667e9b19a03ac1f82cad62`.
Freeze flags describe the historical pre-training freeze, not current completion.
Previously completed 235-run baseline evidence remains separate and untouched.
The existing nonfatal joblib/NumPy deprecation is retained in the full test log.
No failing implementation test occurred in this step.

## Evidence and unresolved scientific gate

Official [FAO ECOCROP](https://www.fao.org/geospatial/data-and-tools/data-portals/ecocrop/)
and [TNAU rainfed maize](https://agritech.tnau.ac.in/agriculture/agri_maize_rainfed_maize.html)
were rechecked. Their precipitation descriptors and fertilizer application
guidance do not establish the measurement contracts needed for these datasets.
No independently validated crop bounds were approved.

Real-data gate: **BLOCKED_FOR_REAL_DATA_GUIDANCE**, both datasets, **zero
approved compatible rules**, no training or test evaluation. The saved contract
fields are explicitly null/unverified drafts, not asserted physical units.
Schema checks validate declarations, not source authenticity or agronomic
correctness. A reviewer must inspect archived source material and independently
document applicability, exceptions and measurement compatibility before approval.

No implicit conversions, min/max-to-optimum invention, label-derived bounds,
synthetic generator rules treated as independent evidence or unsupported
calibration/privacy/field-benefit claims are allowed. Fixed rule metadata does
not implement secure aggregation, differential privacy or client isolation.

## Next concrete action

Obtain original measurement documentation for canonical and Maharashtra:
nutrient units/measurement method, rainfall aggregation period, temperature
statistic, pH method, region/time/stage/water-regime coverage. Then retrieve
crop-specific source records with taxonomy, documented numeric bounds and
applicability. Record reviewer rationale, resolve exceptions, and approve only
compatible rules; unsupported features/crops remain neutral.

Once approval exists, implement a new versioned KG experiment coordinator with
round resume, validation-only alpha selection, frozen controls/ablations and
test journaling before any real-data guided training. Do not modify or reuse
baseline-v1 artifacts as if they were guided results. Independent confirmation,
formal privacy and distributed deployment remain later separate gates.
