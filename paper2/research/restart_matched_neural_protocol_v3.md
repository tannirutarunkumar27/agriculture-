# RR06 matched neural protocol v3

This package was superseded by v4 after the bounded smoke run exposed an
optimizer-work mismatch between centralized and federated rounds. The active
RR06 protocol is `results/restart_matched_neural_protocol_v4/`. It freezes 504 inner-only
development candidates across 56 nested inner folds: five plain-SGD matched
baselines and four AdamW/cosine candidates per fold.

The source closure includes the restart package marker, so standard package
test discovery and `python -m paper2.restart...` execution use the same import
layout as the frozen protocol. The protocol verifies its data-manifest hash,
runtime, source hashes, JSON configuration and complete plan before any later
runner may train a candidate.

No training or outer-test inference occurred when either protocol was frozen.
FedNova remains plain-SGD-only; it is not compared under AdamW because the
implemented restricted FedNova update does not support that optimizer.
