# Privacy foundation and threat model

2026-10-14. R08 is IN_PROGRESS. A contract, clipping primitive and mathematical
accountant are implemented/tested. Private training and secure aggregation have
not been implemented. All existing empirical results remain public-data results.

## Protected unit and trust

The active clients are geography groups, not measured individual farms. The
initial design protects replacement of an entire regional client's dataset in
one of a fixed public set of slots. Other clients and the public schema are
held fixed. The initial central-DP model trusts the curator/server to inspect
local updates and add noise before releases. An external observer may inspect
the approved noisy model transcript. This design does not protect raw records,
unnoised updates or intermediate checkpoints from that trusted server.

For each adaptive round at a common previously released global model, client
i computes delta_i. Clip its entire flattened parameter vector once to L2 norm
C, then take the equally weighted mean of m fixed slots. Replacement of one
clipped vector changes this mean by at most 2C/m. Do not replace the denominator
with a private participating count or reuse sample-count weighting without a
new sensitivity derivation. Adaptive future rounds must use only previously
released noisy states, not shared nonprivate optimizer/client state.

Use isotropic Gaussian noise of coordinate standard deviation s relative to
that sensitivity Delta. The mathematical event has rho=Delta^2/(2*s^2).
Sum rho across adaptive releases and convert with
epsilon=rho+2*sqrt(rho*log(1/delta)). This is the Gaussian zCDP composition
bound, verified against Propositions 1.3/1.6 and Lemma 1.7, with vector Gaussian
extension in Section 2.2 of [Bun and Steinke (2016)](https://arxiv.org/html/1605.02065).
No client-sampling amplification is assumed. The bound is conditional on an
implemented mechanism actually satisfying all those premises.

## What is public and what must change

Client slot identities, model topology and transform definitions must be
public and independent of protected data, or separately released with privacy
accounting. Existing centralized median/scaler/vocabulary/support-gate fitting
uses training client data. A client's replacement can change the transformation
of every other client's update; clipping only the replaced update does not
then justify the 2C/m bound. Therefore those fitted artifacts cannot be treated
as public for a future whole-client privacy claim. Freeze independent public
feature constants/schema, or develop and account a private statistics mechanism.
The diagnostic gate is not automatically private preprocessing.

Same constraints apply to client counts, participation, validation metrics,
early stopping, hyperparameter selection, target statistics and clipping-bound
selection. A released private validation/test metric needs its own mechanism
and accounting; internal trusted-curator computation is different from release.
Existing nonprivate baseline/guarded models cannot supply private initialization
for a claim about their own underlying clients. Use public data-independent
initialization and an explicitly reviewed transcript policy.

Counts/weights, per-client norms/updates, private metric logs, raw arrays and
noise seeds must not be exported as protected outputs. Secure noise generation,
finite-precision effects and persistence/duplicate-release recovery require
review before application guarantees. Simulation RNG tests alone are inadequate.
All runs sharing a protected client compose, including multiple seeds/methods
and overlapping folds. Seeds/folds are not disjoint privacy populations.

## Secure aggregation is a separate later stage

[Bonawitz et al., CCS 2017](https://research.google/pubs/practical-secure-aggregation-for-privacy-preserving-machine-learning/)
provides an established protocol for revealing aggregates under explicit
collusion/dropout assumptions. It is distinct from the 2016 workshop precursor
in the older screening file. Protocol transport, identity authentication,
minimum surviving-client conditions and malicious-client assumptions need
review; no protocol is implemented in this repository. Adding server-side DP
after collecting individual updates does not conceal them from that server.
If server privacy is required, revise the trust model and use a vetted secure
aggregation/distributed-noise implementation. No custom cryptography.

## Implementation and actual verification

- restart/privacy_accounting.py: stable float64 L2 clipping; replacement
  sensitivity; strict Gaussian-event composition; delta conversion. Rejects
  invalid schemas, false sampling assumptions, overflow/underflow and bad arrays.
- restart/test_privacy_accounting.py: eight tests passed. Includes known-vector
  clipping, huge finite-vector stability, factor-two replacement sensitivity,
  closed-form budgets, composition, monotonicity and invalid-event rejection.
- results/restart_privacy_foundation_v2: persistent tests and illustrative budgets.
  Earlier v1 evidence retained. Code is not integrated into model training.
- restart/privacy_transcript.py: append-only simulation release ledger that
  accepts only aggregate Gaussian-event declarations and fixed public metadata.
  It rejects raw updates, per-client diagnostics, checkpoints, and noise seeds
  from a released artifact. This is a transcript-policy guard, **not** secure
  aggregation, distributed noise generation, or a production DP certificate.
- restart/flower_secagg.py: version-pinned compatibility gate for Flower
  `SecAggPlusWorkflow` (v1.42.0), with a seven-client/equal-weight pilot
  configuration and a five-share reconstruction threshold. It delegates all
  cryptographic operations to Flower; it has no client/server execution,
  independent security audit, or production deployment result yet.
- opacus, dp_accounting, cryptography and flwr were absent from the project
  environment during inspection. No dependency was installed.

```powershell
& '.\.venv\Scripts\python.exe' -B -m paper2.restart.check_privacy_foundation --output paper2/results/restart_privacy_foundation_v2
```

For an illustrative 80-release run at delta=1e-5, std/Delta=8 yields an epsilon
upper bound about 5.99. Twenty such runs involving the same client have a
composed bound about 36.49. These are mathematical planning numbers, not
executed privacy-utility results or certification of existing models.

Next: define independent public preprocessing and transcript policy, implement
clipped equal-slot aggregation/noise in a separate training variant, verify the
accounting against an established library when available, test release recovery
and the actual Gaussian sampler, then prospectively freeze experiments.
