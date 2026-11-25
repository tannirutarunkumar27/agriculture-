# Public transform and noisy aggregation implementation

Date: 2026-10-14. R08 remains IN_PROGRESS. Implemented the next mathematical
mechanism stage and tested a synthetic integration. No real-data privacy-utility
experiments or certified privacy releases were performed.

## Fixed transform

public_preprocessing.py defines a 54-dimensional schema: nine tanh-scaled
numeric inputs with separate missing indicators, three categorical fields
with one missing indicator/eight public SHA256 bins each, and nine fixed
species/intercrop flags. Constants are engineering choices in source code,
not fitted client medians, means, variance, vocabulary or support masks.
They are not agronomic suitability ranges or independently optimal scales.
Replacing another client's records cannot change a given row's representation.
Tanh bounds numeric inputs; missing values become zero with an explicit flag.
Category hashes are deterministic and allow collisions; they do not provide
cryptographic concealment. Species flags are retained on unseen cases because
there is no private support-gate fitting. Generalization remains an empirical
question under this changed representation.

The transform definition is public conditional on these fixed choices; client
features, fitted outcomes and private logs are not thereby public. A complete
study must declare the public roster/schema and public initialization assumptions.

## Aggregation primitive

noisy_aggregation.py accepts floating CPU reference/client parameter states
for every fixed public slot. It verifies matching parameter keys, shapes,
dtypes and finite values, flattens the complete update, clips once to norm C,
and averages equally over slots. It does not use private sample counts.
Independent simulated Gaussian noise has std=multiplier*(2C/m), including
the replacement factor two. It emits the matching Gaussian accounting event
and only public metadata. It exposes no client norm, local loss or raw update
in returned logs. The server remains trusted and sees incoming updates.

Accounting follows the Gaussian composition/conversion in
[Bun and Steinke (2016)](https://arxiv.org/html/1605.02065).
The mechanism is a separate variant of equal-weight federated averaging;
existing sample-weighted baselines and restricted FedNova are unchanged.
It is not a secure-aggregation protocol or a novel algorithm claim.

NumPy RNG and floating-point evaluation are simulation machinery. The API
requires simulation_only=True and rejects production mode and zero noise.
This permits tests and eventual public-data utility experiments; an application
guarantee requires a reviewed noise sampler, numerical implementation and
release/recovery policy. A fixed published noise seed would defeat privacy;
the synthetic seed belongs exclusively to reproducible test data.

## Actual checks and results

Executed:

```powershell
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.check_private_primitives --output paper2/results/restart_private_primitives_v1
```

16 tests passed, zero failures/errors: eight accounting tests, four transform
tests and four aggregation tests. Checks include replacement sensitivity,
known clipped vectors, overflow-stable norms, invalid event rejection,
preprocessing invariance under other-client replacement, unknown-category
schema stability, missing/extreme input handling, required participation,
whole-vector rather than per-parameter clipping, and noise-scale agreement.

A synthetic run reused the existing regression local-training client and
trained two clients with six synthetic records each for three rounds. Every
round updated the common model through clipped/noisy aggregation. Final model
parameters were finite; three events were composed. This verifies execution,
not predictive utility or an actual privacy guarantee. No real CSVs were read.

Evidence: results/restart_private_primitives_v1/unittest.log,
verification.json (source hashes), public_transform.json and smoke.json.
New sources: public_preprocessing.py, noisy_aggregation.py,
test_private_primitives.py and check_private_primitives.py. No package install,
frozen baseline change, raw data change or Paper 1/manuscript modification.

Next: prepare separately versioned real-data tensors with this fixed transform;
freeze matching equal-weight controls and simulated noisy experiments. Before
claiming application privacy, validate against maintained accounting/sampling
tools and implement a persistent shared release ledger with safe recovery.
Count all releases across overlapping clients, seeds and configurations;
separately protect any metrics if the evaluation data is private. Secure
aggregation remains a separate protocol dependency.
