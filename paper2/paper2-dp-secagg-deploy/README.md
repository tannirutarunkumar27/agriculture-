# Paper 2 authenticated DP + SecAgg deployment app

This is a self-contained Flower 1.42.0 deployment-runtime app. It exists
separately from the simulation app so that the deployed FAB does not rely on
the research repository being present on a SuperNode.

It is not a publication claim or a production release. A SuperNode must supply
its own `client-id`, `data-root`, and `fold-id` through `--node-config`; client
data paths are deliberately not embedded in the FAB or the server run config.
The frozen study requires exactly seven authenticated SuperNodes and a separate
security/privacy review before any privacy claim is made.
