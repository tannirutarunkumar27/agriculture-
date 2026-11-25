# Candidate method specification — provisional, not validated

2026-10-13 implementation update: the class-aligned soft-bias module and local
training adapter are implemented and tested with synthetic fixtures. Real-data
guidance remains blocked (zero approved compatible rules); no agricultural
validation or novelty claim follows. See `knowledge_module_specification.md`
for the implemented input/scope/evidence contract and explicit boundaries.

## Architectural decisions

Begin with a shared small MLP and a single immutable feature/label schema per
dataset. Compare all federated algorithms on identical models and starting
weights. Keep XGBoost as a centralized reference: independently fitted tree
leaf mappings cannot be parameter-averaged meaningfully. Do not transplant
Paper 1's pooled knowledge attention as class-aligned crop suitability.

Each dataset is a separate experiment. Canonical clients are simulated;
Maharashtra clients can use observed district labels, but not be called genuine
farm measurements until provenance is verified. Manipur district clients remain
synthetic, with field-disjoint evaluation. DES/FAOSTAT are contextual statistics,
not recommended-crop labels. Do not join weather without matching location/time.

## Candidate knowledge mechanism

For class c and available, independently validated rule j, calculate soft
suitability s_cj(x) in [0,1] in the rule's original units. Define

    q_c(x) = sum_j m_cj(x) w_cj s_cj(x) / sum_j m_cj(x) w_cj
    z'_c(x) = z_c(x) + alpha * (q_c(x) - 0.5)

Here m marks valid/applicable observed rules; w is a documented nonnegative
confidence weight. Set q_c=0.5 when the denominator is zero: missing knowledge
must be neutral. Keep the crop-to-rule binding explicit. This is a soft bias,
not a hard rejection or calibrated probability. Select alpha on validation
only; alpha=0 is the no-guidance control. Cross-entropy uses z'. No extra
knowledge loss or new aggregation rule is justified yet. This simple mechanism
must be reconsidered after the closest-work review, not branded novel.

**Safety gate:** no agronomic rules enabled until per-rule source, crop label,
feature unit, rainfall period, scope, bounds, confidence and regional exceptions
are checked. Numeric matrix validity alone is insufficient. Existing manual
prior weights are not a verified rule bank. Missing humidity in Maharashtra
is not zero humidity. Daily Manipur rainfall is not annual rainfall.

## Implemented federated baseline definitions

Full formulations checked against the cited sources in
`closest_paper_comparison.md`. The FedProx sampling/weighting variant is explicit,
and no general convergence theorem or full FedNova solver support is claimed.

For selected clients S, p_k=n_k/sum_{i in S} n_i. All clients start from theta_t.

- FedAvg: theta_{t+1}=sum_k p_k theta_k after local supervised SGD.
- FedProx: local supervised loss plus mu/2 * ||theta-theta_t||^2;
  sample-weighted aggregation. mu=0 must reproduce FedAvg under identical RNG.
- Restricted FedNova implementation: fixed shared learning rate, plain SGD without
  momentum/proximal terms, actual local step counts tau_k>0;
  delta_k=theta_t-theta_k, tau_eff=sum p_k tau_k;
  theta_{t+1}=theta_t-tau_eff*sum p_k delta_k/tau_k.
  Do not apply this normalizer to Adam, momentum, or changing learning rates.

KG-FCR initially means the validated knowledge-aware model trained with one of
these established optimizers, not a claimed new optimizer. Any modification
requires a mathematical specification and controlled tests first.

## Data and privacy protocol

Freeze untouched test IDs before partitioning; retain validation separately.
Use grouped splitting for repeated feature profiles/fields. For public-data
simulation, a training-only centralized scaler is a preprocessing reference,
not a privacy-preserving operation. Distributed deployment must use public
scaling constants, local transforms with a justified common coordinate system,
or a separately secured training-statistics protocol. No test statistics.

Reject non-finite, stale, wrong-shaped, wrong-dtype and wrong-schema updates.
Validation of message structure does not stop model poisoning. A single-process
simulation offers no isolation from the host process, SecAgg, or DP. Document
an honest-but-curious server, collusion and client-dropout assumptions before
selecting established secure aggregation; do not invent cryptography.

## Required acceptance checks

Partition coverage/disjointness/reproducibility; train-only transformations;
weighted aggregation toy examples; FedProx mu=0 equivalence; FedNova equal-step
equivalence plus unequal-step toy objective; schema rejection; non-finite
gradients; class-aligned knowledge sensitivity; missing-rule neutrality;
checkpoint/resume and deterministic small-round smoke tests. Large experiments
and claims wait until these checks pass.
