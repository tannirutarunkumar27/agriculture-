# Central-DP plus SecAgg+ pilot protocol v4

This was the frozen execution candidate
`results/restart_dp_secagg_pilot_protocol_v4/`. It defines one prospective,
simulation-only training round over the seven fixed training clients of
`outer_0`. The companion launch command is dry-run by default and was verified
against this exact frozen package.

- The dedicated `secagg_dp_pilot` Flower manifest points only to the
  DP-specific client and server components; it cannot select the non-DP pilot
  app by mistake.
- Flower central DP with client-side fixed clipping uses `C=1.0`, Gaussian
  noise multiplier `8.0`, and `delta=1e-5`. The frozen run metadata explicitly
  sets `differential_privacy_enabled=true`.
- SecAgg+ is configured for seven participants with a five-client reconstruction
  threshold. It protects individual update vectors during aggregation only
  under Flower's protocol, client, transport, and threshold assumptions.
- The one-round conditional zCDP conversion is epsilon `0.6076282390`. It is
  not a full-pipeline, multi-run, or deployment privacy budget.
- Outer-test evaluation, client metrics, hyperparameter selection, deployment,
  crop ranking, and knowledge guidance are disabled.

The central-DP mechanism assumes a trusted server for the aggregate. It does
not establish local DP, distributed-noise protection from that server, secure
deployment, remote farms, or application-level privacy certification. Versions
v1--v3 are retained as immutable provenance only: v1 had DP-disabled metadata,
v2 corrected its contract but predates the launcher, and v3 exposed a residual
DP-disabled run-config flag during dry-run. Use v4 for any future external
execution after a non-sandboxed Flower runtime is available. It was superseded
by v5, which additionally freezes the full helper and launcher source closure.
