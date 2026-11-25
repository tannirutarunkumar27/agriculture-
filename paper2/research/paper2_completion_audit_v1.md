# Paper 2 completion audit

The repository-local implementation and evidence were reconciled without
starting training or changing archived studies. Data/split validation, inner
search, unnoised refit, noise simulation, and feature-timing controls are
complete within their declared scopes.

The remaining blockers are external or scientifically unresolved:

1. A clean 504-candidate v9 study requires a real remote GPU host. The archived
   v7 CPU study remains at 441/504; its receipts cannot be rebound to v9.
2. Secure aggregation and a one-round central-DP pilot are implemented/frozen,
   but have not been executed on a remote Linux runtime; the privacy gate remains closed.
3. Linux deployment cannot be verified without server credentials.
4. Independently validated agronomic rules are unavailable.
5. Cultivar/density/mixture covariates were recovered completely for the current
   607 systems, but crop-system ranking remains blocked by original-experiment
   comparability, publication-reference, and utility-protocol review.

Therefore Paper 2 is not complete as an end-to-end privacy-preserving crop
recommendation system. The supported claim remains retrospective federated /
conditional regression on the declared public dataset, with simulation-only
privacy diagnostics.
