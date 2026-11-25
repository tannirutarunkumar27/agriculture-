# RR06 matched neural runner and bounded smoke test

The active execution package is
`results/restart_matched_neural_execution_v3/`, bound to matched-neural
protocol v4. It snapshots the protocol, runner, training primitive, inner-fold
loader, atomic writer and aggregation sources. Dry-run verifies all 504 planned
candidates without training.

The bounded smoke output is
`results/restart_matched_neural_study_v3/`. Exactly one candidate completed:
`outer_0_inner_1_sgd_constant_h32_centralized_sample_weighted`. Its checkpoint
contains 240 consecutive inner-validation observations, 24 optimizer steps in
every round, and an identity-bound completion receipt. Best inner-validation
RMSE was 0.979827 t/ha at round 2; last-round RMSE was 2.472472 t/ha. This is a
runner/convergence smoke result, not an outer result or a selected paper result.

The first smoke execution (`study_v2`) used only four centralized steps per
round against 24 total federated steps. It is retained as provenance but is
superseded and must not be combined with v3. Protocol v4 prospectively defines
centralized work as four steps times the number of training clients.

The study remains partial (1/504); selection is unavailable. Scaling requires
explicit resumptions of the v3 study, followed by a complete receipt and
convergence audit before any outer-test inference.
