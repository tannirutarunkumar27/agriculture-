# RR06 matched neural protocol v5

V5 was superseded after measured work accounting showed that optimizer steps
matched but processed-example exposure did not. The active RR06 protocol is
`results/restart_matched_neural_protocol_v6/`. It contains 504 inner-only
development candidates.

Two work controls are explicit. Centralized candidates perform four optimizer
steps times the number of participating training clients, matching the total
federated optimizer-step count per round. Centralized optimizer state persists
across rounds and resumable checkpoints; federated client optimizer state
resets each client-round. This matters for AdamW moment estimates and is now
part of the frozen protocol rather than an implicit implementation detail.

V6 additionally matches the centralized per-round processed-example budget to
the sum produced by the fixed federated client/batch schedule for that fold.

The runner records optimizer steps, examples processed, elapsed seconds and
per-round inner-validation metrics. Selection remains prohibited until all 504
candidates complete. No outer-test arrays, privacy claim, ranking, knowledge
guidance or independent-confirmation claim are enabled.
