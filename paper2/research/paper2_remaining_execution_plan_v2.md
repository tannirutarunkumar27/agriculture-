# Paper 2 remaining execution plan v2

## 1. Matched-neural GPU study

Run a clean 504-candidate v9 CUDA study using
`paper2/remote_gpu/run_remote_rr06.ps1`. Do not copy or rebind the 441 v7 CPU
receipts: v7 and v9 have different execution identities. Acceptance requires
504 independently verified receipts, zero unreviewed failures, then RR06
selection finalization and the frozen RR07 development outer study.

## 2–3. Secure aggregation and differential privacy

The portable v6 package freezes a one-round seven-client Flower 1.42.0
SecAgg+ pilot with client-side fixed clipping and central Gaussian DP. Fourteen
core tests plus the independent execution-audit tests pass. Execute it only on
a separate remote Linux runtime using `run_remote_dp_secagg.ps1`, retrieve its
receipts, then run `verify_dp_secagg_execution.py`. This pilot is necessary
evidence, not application privacy certification; a multi-round release budget,
private preprocessing/selection, transport/authentication, and independent
security review remain required for an end-to-end claim.

## 4. Linux verification

Both remote launchers are fail-closed for localhost and verify runtime versions,
CUDA where applicable, data hashes, and new output paths. Final verification
requires an SSH endpoint and credentials supplied by the infrastructure owner.

## 5. Agronomic rules

The rule-bank schema validator is implemented. The current independent rule
bank contains zero approved rules, so guidance remains disabled. Every future
rule must provide crop taxonomy, exact source locator, original bounds/units,
dataset transformation, region/stage/water scope, independence, and documented
domain approval.

## 6. Restricted observed-set ranking

The prior “missing cultivar/density/mixture data” statement was too broad. The
source management table supplies complete cultivar and density fields for all
915 component rows, and complete mixture design/pattern fields for all 616
intercrop component rows across 607 systems. The recovered asset is saved in
`restart_ranking_covariates_v2`. Ranking remains disabled until each experiment's
design, references, comparable alternatives, tie/replicate handling, and
restricted harvest-dry-matter utility protocol are independently approved.
