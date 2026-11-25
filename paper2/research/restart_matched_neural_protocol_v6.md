# RR06 matched neural protocol v6

The active RR06 protocol is
`results/restart_matched_neural_protocol_v6/`. It freezes 504 inner-only
development candidates.

Centralized and federated candidates now match both total optimizer steps and
processed-example exposure per round. The centralized example budget is
computed from the fixed client sizes, batch size and four-step federated
schedule for each inner fold. Centralized optimizer state persists across
rounds and checkpoints; federated client optimizer state resets each
client-round.

Every round records optimizer steps, examples processed, elapsed seconds and
inner-validation metrics. The runner rejects any work mismatch before saving
the round. Selection is unavailable until all 504 candidates complete, and no
outer-test arrays or stronger privacy/generalisation claims are enabled.
