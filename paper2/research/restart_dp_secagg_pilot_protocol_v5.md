# Central-DP plus SecAgg+ pilot protocol v5

The active frozen package is
`results/restart_dp_secagg_pilot_protocol_v5/`. It defines one prospective,
simulation-only training round over the seven fixed training clients of
`outer_0`; dry-run validates its complete source and data binding without
starting Flower.

- The dedicated `secagg_dp_pilot` manifest selects only the central-DP plus
  SecAgg+ client and server components.
- Central DP uses client-side fixed clipping (`C=1.0`), Gaussian noise
  multiplier `8.0`, and `delta=1e-5`; frozen run metadata explicitly has
  `differential_privacy_enabled=true`.
- SecAgg+ uses seven participants and a five-client reconstruction threshold.
- The one-round conditional zCDP conversion is epsilon `0.6076282390`; it is
  not a privacy budget for model selection, repeated runs, evaluation, or
  deployment.
- The source manifest now covers the app manifest, DP/SecAgg components,
  task, protocol helpers, accounting, validation writer, freezer, and launcher.

This is still a prospective simulation contract, not evidence of a completed
federated run. It makes no claim of local DP, distributed-noise privacy from
the server, remote farm participation, secure deployment, crop recommendation,
or application-level privacy certification. Execute it only in a suitable
non-sandboxed Flower runtime and preserve the resulting logs as a separate,
immutable execution record.
