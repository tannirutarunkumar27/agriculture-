# RR06 runner recovery fix and active acceptance study

The active execution package is
`results/restart_matched_neural_execution_v6/`, bound to matched-neural
protocol v6. It supersedes execution v5 because the runner recovery journal
logic changed.

The corrected runner now handles the crash window in which a valid completion
receipt is committed before the attempt journal changes from `STARTED` to
`COMPLETE`. On resume it verifies the completion assets, records
`CACHE_VERIFIED_AFTER_INTERRUPTION` with the completion hash, and refuses any
other inconsistent completed-run journal state. The independent auditor accepts
only ordinary completions or this verified recovery state.

The active bounded study is `results/restart_matched_neural_study_v6/`.
All nine configurations for the first four inner folds completed. The latest
independent audit in `results/restart_matched_neural_audit_v6/` passed all 36
receipts, work histories and journals. Every fold uses 24 optimizer steps per
round; their fixed processed-example budgets are 384, 408, 384 and 416.
The study remains partial with 468 candidates pending; selection is unavailable
and no outer-test inference occurred.

The earlier v5 study and its 18 runs remain immutable provenance but must not
be merged with the active v6 study because their execution-source identities
differ.
