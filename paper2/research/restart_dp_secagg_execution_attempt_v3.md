# DP plus SecAgg+ execution attempt v3

On the frozen v5 protocol, the external runner created
`results/restart_dp_secagg_pilot_execution_v3/` and passed the protocol
verification gate before launching Flower.

Flower created its isolated configuration under that execution directory and
began a local SuperLink on `127.0.0.1:39091`, but its 15-second readiness check
failed in this sandbox. The launcher returned exit code 1 and deliberately did
not retry. No clients joined, so no federated training result exists.

The immutable execution record includes the protocol-manifest hash and SHA-256
hashes of both Flower logs. Its manifest records
`outer_test_inference=false` and `raw_updates_exported=false`. This is an
environmental launch failure, not a model-performance or privacy result. Run
the v5 package once in a non-sandboxed Flower environment before claiming an
executed central-DP or secure-aggregation experiment.
