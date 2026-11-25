# RR06 pending-run report

The independently audited RR06 study currently has 441 verified runs out of
504 planned runs. The remaining 63 runs are all in `outer_7` (inner folds 0–6,
9 runs per inner fold). No selection is allowed until these runs are complete
and independently audited.

Pending algorithm counts are: 14 centralized controls, 28 FedAvg runs, 14
FedProx runs, and 7 restricted FedNova runs. The aggregation-objective counts
are 49 sample-weighted and 14 equal-client runs.

This report is a receipt-level inventory only; it performs no training and no
outer-test inference. The next execution action is to run exactly these 63
IDs from the frozen execution package on the configured remote GPU host, then
rerun the independent verifier before finalization.
