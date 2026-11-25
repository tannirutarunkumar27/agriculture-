# Paper 2 implementation phase report

Date: 2026-10-12. Scope: closest-work/source checks, reproducible partitions and
established neural federated baselines. Public-data CPU simulation only.

Historical phase report. The subsequent implementation and validation-only
convergence results are saved in `CONVERGENCE_PHASE_REPORT.md`; latest status
and test totals are in `STATUS.md`. Earlier evidence below remains unchanged.

## Research evidence saved

`research/closest_paper_comparison.md`: focused matrix with source links,
verified task/schema/split information, externally reported results clearly
labeled, access gaps, and differences from the proposed experiment. Existing
FedAvg crop classification and knowledge-guided recommendation preclude broad
"first" claims. X-FedAgriTwin exact clients/splits could not be verified through
the available publisher interface. No numerical superiority claim or fabricated
gap was made.

`research/agronomic_source_checks.md` and `agronomic_registry.json`: independent
source and unit audit, zero approved rules. Fertilizer rates are not measured
soil nutrients; annual precipitation requirements are not daily rainfall.
Manual prior weights and synthetic-generator rules are not independent source
validation. Baselines run; knowledge guidance remains disabled.

## Code/configuration created

- `data.py`: explicit three-dataset schemas; source hashing; duplicate-profile
  and field union groups; per-class train/validation/test splitting; IID,
  label-wise Dirichlet, unequal-size and metadata-based district clients;
  disjointness/coverage checks; train-only standardization and soil vocabulary.
- `federated.py`: shared MLP, stateless local plain SGD, proximal penalty,
  sample-weighted averaging, plain-SGD FedNova normalization, selected-client
  and parameter/schema/round/base-state checks. Server aggregation never accepts
  feature rows, but coordinator/host is not cryptographically isolated.
- `train.py`: centralized/federated runner, participation sampling, validation
  NLL checkpoint selection, per-round metrics/work/timing, logical tensor bytes,
  atomic round checkpoints, strict resume identities, final test outputs.
- `configs/partition_matrix.json`, `configs/smoke.json`: executed configurations.
- `scripts/create_clients.py`, `prepare_partition_matrix.py`,
  `run_smoke_matrix.py`, `verify_implementation.py`: repeatable commands.
- `tests/test_partitions.py`, `tests/test_federated.py`: regression/integration
  checks, complementing the original six data-audit tests.
- `requirements-runtime-observed.txt`: observed working versions, not a
  certified cross-platform lock or installation performed this phase.

## Executed commands and outputs

Exact commands are saved in `STATUS.md` and `README.md`.

| Check | Executed result | Saved evidence |
|---|---|---|
| Final expanded test suite | 34 passed / 0 failed | `results/tests_v2/unittest.log`, `verification.json`; earlier 31-test run retained in tests_v1 |
| Partition matrix | 42 partitions / 210 clients | `results/partitions_v1/manifest.json`, `client_distributions.csv`, partition JSONs |
| Three-round smoke matrix | 12 successful runs | `results/smoke_v1/verification.json`, `smoke_results.csv`, individual run directories |
| Resume regression | Exact state/prediction match to uninterrupted execution | Test log; integration artifacts use temporary directories, not retained |
| Source integrity | Source hashes checked before training | Each partition and run manifest |

Each run directory contains `manifest.json`, `rounds.json`, `checkpoint.pt`,
`summary.json`, `test_predictions.npz`. Test predictions are derived outputs;
logs do not dump raw training feature records or credentials.

## Interpretation and limitations

All training results are deliberately short diagnostics. For example, canonical
FedAvg test accuracy after three rounds is about 12.42%; centralized about
50.61%. These are NOT the previous KG-Hybrid scores, do not establish optimizer
superiority and must not enter the paper as mature performance claims.

Canonical seed-42 splits: 1540/330/330. Maharashtra: 3145/684/684. Manipur:
56940/18980/18980, or 60/20/20, because each class has five independent fields.
Split groups and held-out IDs are shared across client strategies for each seed.
Manipur remains synthetic; district clients are not genuine deployed farms.
Maharashtra region fields do not certify measurement provenance. Global
training statistics are a centralized PUBLIC simulation reference, not private
preprocessing. Actual local step budgets differ despite identical epoch settings
and are recorded. No test-based tuning is authorized by these diagnostics.

FedProx is an explicit uniform-participation, sample-weighted variant, not a
verbatim reproduction of original weighted-sampling/equal-averaging pseudocode.
FedNova supports constant-LR plain SGD only. Payload bytes exclude real network
overhead. Message compatibility checks are not cryptographic authentication or
poisoning defenses. No secure aggregation, DP/accountant, asynchronous dropout,
distributed server or Linux validation has been implemented.

## Next actions

1. Obtain independent crop-specific rule records and verify dataset units,
   rainfall periods, measurement meanings, scope and exceptions. Do not replace
   missing evidence with source-name defaults.
2. Define validation-only convergence/hyperparameter pilots and fixed comparison
   budgets; add/verify matched LR/RF/XGBoost centralized references as appropriate.
3. Run repeated-seed IID/non-IID/unequal/participation studies; report global and
   client-specific held-out metrics with defined local test assignments.
4. Enable class-aligned knowledge only after approval gates pass, then conduct
   no-knowledge/corrupted-knowledge controls. Full novelty review remains open.
5. Select and validate an established privacy framework independently, including
   a threat model and accounting where applicable. Prepare Linux/distributed
   deployment only against supplied actual infrastructure.
