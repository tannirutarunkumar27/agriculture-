# Audit repair and fixed-public refit preparation

2026-10-14. User authorized inspecting/fixing mistakes and proceeding to the
next implementation part. R12H audit-hardening and R13 protocol/train-only
core subtasks VERIFIED_COMPLETE. Overall R13 remains IN_PROGRESS.

## Verified mistake and targeted repair

The frozen R12 verifier checked models/metrics but did not require an attempt
journal or validate its sequence, identity, terminal status and completion
receipt hash. A synthetic regression deletes one generated temporary journal:
the legacy audit still returns PASS_COMPLETE_INNER_SEARCH. This is a verifier
coverage gap, not evidence that actual model scores were wrong.

New restart/control_search_integrity.py supplements the frozen code. It rejects
missing/empty/misbound/nonterminal journals, invalid attempt lifecycles, lost
failure tracebacks and changed completion hashes. It also binds the complete
metric-audit rows to actual completion receipts/history and checks exact
selection equality. Interrupted/failed attempts remain visible; legitimate
cache-verified interruption handling is supported. R13 verifies this gate.
Existing frozen sources and result packages were not rewritten.

Fresh full final/best inner-model audit PASS_COMPLETE_INNER_SEARCH at
restart_control_search_verified_v4, and strict provenance audit
PASS_COMPLETE_SEARCH_PROVENANCE at restart_control_search_integrity_v1.
All224 actual runs have valid journals,224 attempts, zero failed attempts.
No numerical error was found in these checked latest-stage outputs. This is
not an assertion that every historical project component has been exhaustively
audited. Earlier results, raw files, Paper1 and open LaTeX remain untouched.

Obsolete real-data/runner-pending and continue-v2 instructions were removed
from the active KNOWN_ISSUES ledger. Historical failures and phase counts remain
in prior reports/STATUS/CHANGELOG. Closed v1/v2 matrices must not be resumed.

## R13 prospective plan and training implementation

Separate files public_refit_protocol.py and public_refit_training.py implement
the unnoised equal-client control refit, not sample-weighted FedAvg, a new
knowledge-guided method, or a private release.40 planned runs:eight geographic
outer groups x seeds123,2024,3407,9999,2718. Seed42 remains inner tuning only.
The five seeds were previously used; they are not new regional replications.

All verified inner selections use scale1 and LR .01. Fixed final-round refit
budgets are133,129,114,149,106,109,106,128 in outer_0..7 order. No outer scores
were consulted for these choices. MLP hidden32, batch32, exactly four local
SGD minibatches/client/round, gradient clip1, no momentum/weight decay, equal
client parameter average, no global clipping/noise. This preserves the frozen
R12 optimizer/work rather than silently using the old80-round refit engine.

The training loader accesses only x_train/y_train/ids_train, with hash and
partition checks. The fit API accepts training tensors only; signatures bind
their bytes, shape, run, settings and partition. Atomic round checkpoints
support exact resume and completed-round finalization. Completed fit receipts
bind normalized model/checkpoint assets and unit/checkpoint semantics. The
predict_raw helper restores raw t/ha units exactly once; future inference must
use this explicit scale contract. No outer-test checkpoint selection exists.

Package results/restart_public_refit_protocol_v1 freezes configuration, plan,
selections, strict provenance, source snapshots, data/runtime and parent/audit
bindings. Freeze/verify commands PASS_FROZEN_REFIT_PLAN_AND_CORE,40 runs.
This is a plan and tested train-only core, not an execution runner or completed
experiment.0 real-data refit runs,0 new outer predictions. A future execution
coordinator must own a writer lock, implement once-only inference, journal
failures and verify cached predictions/support panels before expensive runs.

## Tests and errors actually observed

Final scoped/dependency total61 passed, zero final errors/failures:
18 new regression/planner/refit tests,17 existing search tests,26 dependency
tests. These include legacy-gap reproduction/strict rejection; corrupt attempt
and metric receipt rejection; fixed plan/seeds and semantic/source/parent
mutations; exact optimizer equivalence with the frozen inner engine; exact
resume; completed-cache no retraining; changed training data/token rejection;
partial model/fit-unit metadata rejection; client isolation; explicit unit
restoration; training loader refusing to fetch object-valued test-array fixtures.

Evidence:restart_public_refit_tests_v4, restart_control_search_tests_v5 and
restart_public_refit_dependency_tests_v1. Not a full-repository suite and not
an agronomic/privacy certification.

Intermediate evidence preserved:v1's14 tests PASS; v2's16-test suite had one
FileNotFoundError because the new synthetic metric-binding fixture mocked
verification without creating its synthetic package manifest. Production
guards were not weakened. Created that fixture manifest, added binding tests,
then v3's16 and final v4's18 tests PASS. No real-study failure occurred.

## Commands executed

```powershell
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.test_public_refit --output paper2/results/restart_public_refit_tests_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.verify_control_search --output paper2/results/restart_control_search_verified_v4
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.test_public_refit --output paper2/results/restart_public_refit_tests_v2
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.control_search_integrity --output paper2/results/restart_control_search_integrity_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.test_public_refit --output paper2/results/restart_public_refit_tests_v3
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.test_control_search_runner --output paper2/results/restart_control_search_tests_v5
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.test_public_refit --output paper2/results/restart_public_refit_tests_v4
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.test_work_scaled_simulation --output paper2/results/restart_public_refit_dependency_tests_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.public_refit_protocol freeze --output paper2/results/restart_public_refit_protocol_v1
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.public_refit_protocol verify --output paper2/results/restart_public_refit_protocol_v1
```

Authored files:four new Python modules/tests, this report and six project
ledgers. Generated files:new audit/test versions and frozen refit package.
No dependencies installed, cloud costs, credentials, public ports or automation
changes. No active Python writer remains. Frozen Paper2 sources/packages,
Paper1, raw data, historical results and open manuscript were preserved.

## Next action and unchanged limits

Implement/test a separate bounded unnoised coordinator with explicit execute/
resume, raw-unit once-only outer-inference receipts, ambiguous-inference
refusal, failure history and independent cache/support-metric audit; freeze
its execution code, then run/audit a bounded starter. Do not execute via the
old capped80-round engine or reuse test-selected checkpoints.

The scope is development conditional harvest regression, not crop choice or
field utility. Grid edges,80/224 budget hits and11/56 selected inner fits worse
than matched constants remain genuine findings, not fixed software errors.
Any matched noisy follow-up needs a separate prospective mechanism/ledger/
accounting freeze before evaluation; R13'sunnoised package does not authorize
privacy claims. Agronomic-unit/ranking comparability, certified privacy/SecAgg
and remote deployment gates remain unresolved.
