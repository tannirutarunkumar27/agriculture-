# RR07 development outer-evaluation gate

The RR07 planner is implemented in
`restart/rr07_development_protocol.py`. It creates a 160-run plan: eight
geographic outer folds, four selected algorithm/objective paths per fold, and
five computational seeds.

The planner cannot run or freeze from a partial RR06 study. It requires the
independent `PASS_FINALIZED_INNER_SELECTION` artifact, verifies all eight outer
folds are represented, and keeps outer-test inference disabled in the plan.
The eventual execution runner must be separately frozen and must evaluate each
outer test exactly once, reporting equal-group and pooled metrics separately.

No RR07 plan has been frozen yet because the active GPU study has not completed
504/504 candidates. This is an implementation gate only; no outer arrays were
loaded and no outer inference occurred.
