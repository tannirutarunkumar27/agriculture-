# RR06 matched neural protocol v2

The v2 protocol was superseded by v3 after the restart package marker was added
to the source closure. The current active protocol is
`results/restart_matched_neural_protocol_v3/`. It is a verified, no-training,
inner-only development plan with 504 candidate runs across the existing 56
nested inner folds.

It preserves the five matched SGD candidates from RR05 and adds four AdamW
with cosine-decay candidates per fold: centralized, sample-weighted FedAvg,
sample-weighted FedProx, and equal-client FedAvg. Restricted FedNova appears
only in the plain-SGD panel because its current implementation is not valid
for AdamW/weight-decay updates.

The forthcoming runner must use only each fold's train and validation arrays,
write deterministic resumable checkpoints and receipts, record per-round
validation metrics and actual elapsed seconds, and select settings per
algorithm/objective by equal-inner-group mean best RMSE with frozen-order tie
breaking. It must not access outer-test arrays, select on outer results, or
describe this development study as independent confirmation.

The earlier v1 freeze is retained as provenance but failed semantic verification
because its tuple-valued Python configuration was serialized as JSON lists.
Use v3 only.
