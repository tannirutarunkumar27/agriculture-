# RR07 execution gate

The RR07 execution gate is implemented in
`restart/rr07_execution_gate.py`. Once RR06 finalization produces the frozen
160-run plan, this gate snapshots the plan/data binding and the future
outer-runner source closure into a separate execution package.

It currently performs no outer inference and loads no outer test arrays. It
rejects partial/unfrozen plans, changed data manifests, altered plan artifacts,
duplicate run IDs, and any plan that enables outer inference. A later execution
runner must be added and frozen separately before remote GPU evaluation.
