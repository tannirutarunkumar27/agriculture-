# RR06 matched neural runner v5 acceptance and partial execution

The active protocol/execution pair is
`results/restart_matched_neural_protocol_v6/` and
`results/restart_matched_neural_execution_v5/`. Earlier smoke studies are
preserved as provenance but must not be combined with this study.

The active partial study is
`results/restart_matched_neural_study_v5/`. Eighteen candidates have completed:
all nine configurations for `outer_0_inner_1` and `outer_0_inner_2`. The
independent receipt auditor passed all 18 candidates in
`results/restart_matched_neural_audit_v2/`.

Every configuration within a fold has identical measured work. Inner fold 1
uses 24 optimizer steps and 384 processed examples per round; inner fold 2 uses
24 optimizer steps and 408 examples per round because its fixed client-size
distribution differs. Centralized candidates match the federated fold budget.
Centralized optimizer state persists across rounds; federated client optimizer
state resets each client-round.

The study remains partial: 18/504 complete, 486 pending. No selection file
exists and no outer-test inference occurred. Scaling must continue through
explicit bounded resume batches, followed by a complete independent audit
before any RR07 development outer evaluation.
