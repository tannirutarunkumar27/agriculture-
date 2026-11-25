# Central-DP plus SecAgg+ pilot protocol v2

The active frozen package is
`results/restart_dp_secagg_pilot_protocol_v2/`. It defines one prospective
simulation-only round over the seven fixed training clients of `outer_0`.

- Flower 1.42.0 `SecAggPlusWorkflow` and `secaggplus_mod` protect individual
  update vectors from the server during aggregation, subject to Flower's
  protocol, client, transport, and threshold assumptions.
- Flower central DP with client-side fixed clipping uses `C=1.0`, Gaussian
  noise multiplier `8.0`, and `delta=1e-5`.
- The corresponding one-round conditional zCDP conversion is epsilon
  `0.6076282390`. It is not a full-pipeline or multi-run privacy budget.
- Test evaluation, client metrics, hyperparameter selection, deployment,
  ranking, and knowledge guidance are disabled.

The central-DP mechanism assumes a trusted server for the aggregate. It does
not establish local DP, distributed-noise protection from the server, secure
deployment, or application-level privacy certification. The v1 package is
preserved because its inherited metadata mistakenly marked DP disabled; v2 is
the corrected protocol and must be used for any future execution.
