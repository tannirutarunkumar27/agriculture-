# RR06 process-parallel execution plan

The active pair is `restart_matched_neural_protocol_v6` and
`restart_matched_neural_execution_v7`. Four spawned worker processes execute
disjoint candidates; this is intentionally process parallelism, not threading.
PyTorch model initialization and RNG control are process-global, so threads
could race deterministic seeds. Each worker uses one PyTorch thread and writes
only to its own candidate directory.

The parent coordinator alone holds the study writer lock and commits attempt
journals, completion hashes, progress and eventual selection. Each candidate
is identity-bound to the frozen execution manifest. A crash after a completion
receipt is repaired only by verifying the receipt and recording
`CACHE_VERIFIED_AFTER_INTERRUPTION`.

Run batches are capped and independently audited before the next batch. The
current `study_v7` has 81/504 verified candidates, representing nine complete
inner-fold panels. Independent audit v10 passed all 81 receipts. The next
batches continue the remaining inner folds with four workers. Selection remains
unavailable until all 504 candidates have valid completion receipts and pass
the independent audit.
