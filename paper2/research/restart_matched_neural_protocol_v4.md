# RR06 matched neural protocol v4

V4 was superseded after the nine-path smoke audit found that centralized AdamW
discarded its optimizer moments at every round boundary. The active RR06
protocol is `results/restart_matched_neural_protocol_v5/`. It freezes 504 inner-only
development candidates across 56 nested inner folds.

The v2 smoke execution showed that a centralized round used four optimizer
steps while six federated clients used 24 total steps. V4 removes that compute
confounder prospectively: each federated client retains four local steps and
the centralized candidate performs `4 × number_of_training_clients` pooled
steps per round. The runner records actual optimizer steps and elapsed seconds.

V5 additionally persists centralized optimizer state across rounds and
checkpoints, while federated client optimizers reset explicitly each
client-round. It records processed-example counts as well as optimizer steps.

The study still compares five plain-SGD candidates and four AdamW/cosine
candidates per fold. Restricted FedNova remains plain-SGD-only. Selection is
inner-only and unavailable until every candidate completes. No outer-test
arrays, ranking, knowledge guidance, privacy claim, or independent-validation
claim are enabled.
