# I. Introduction

Agricultural outcome models are often developed from observations collected at
several locations, seasons, and experimental sites. These data are valuable
precisely because the environments differ, yet that variation also complicates
evaluation. A model can perform well when records from a well-represented site
dominate a pooled test set while remaining unreliable at smaller or previously
held-out locations. The distinction matters for cereal-legume experiments, in
which species composition, management, climate, and measurement support are
not evenly distributed across research groups.

Federated learning offers a way to coordinate model fitting without combining
all client records in one training table. The core algorithms are already well
established: [FedAvg](https://proceedings.mlr.press/v54/mcmahan17a.html)
aggregates locally updated models, [FedProx](https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html)
adds a proximal term for heterogeneous clients, and
[FedNova](https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html)
normalizes local updates to address objective inconsistency. Their availability
does not remove the experimental-design problem: agricultural clients may have
unequal sample sizes and disjoint biological support, and pooled metrics may
answer a different question from metrics that weight held-out groups equally.

This study examines that problem using 607 recorded cereal-legume
crop-management systems from eight geographic groups. The target is complete
harvest seed dry matter, expressed in t/ha, for an observed system. Species
composition is an input condition rather than a recommended label. We therefore
formulate the task as retrospective conditional regression, not as best-crop
selection, causal treatment comparison, or a claim that changing a system will
improve yield.

The study asks three questions. First, how do conventional centralized and
simulated horizontal federated baselines behave when each geographic group is
held out in turn? Second, how does interpretation change between equal-group
and pooled evaluation, particularly when group sizes and species support are
unequal? Third, what utility degradation is observed when prospectively matched
clipping and Gaussian aggregation-noise variants are applied to a fixed-public
representation? The third question is deliberately restricted to a utility
simulation. It does not establish end-to-end differential privacy, secure
aggregation, or a deployable private system.

The contributions are methodological and evidential:

1. a versioned nested group-held-out protocol with training-only preprocessing,
   immutable data and code identities, validation-only model selection, five
   declared refit seeds, and one-time outer-inference receipts;
2. a complete comparison of training-mean, ridge, random-forest, XGBoost,
   centralized-MLP, FedAvg, FedProx, and restricted-FedNova baselines under the
   same conditional harvest-regression task;
3. reporting that separates equal-group from pooled metrics and includes
   regional and seen/unseen species-composition support panels, rather than
   suppressing adverse or small-group results;
4. a separately identified post-hoc support sensitivity and a prospectively
   frozen matched clipping/noise simulation whose limitations are retained in
   the evidence record; and
5. reproducibility assets linking plans, source and data hashes, attempts,
   cached predictions, independent audits, exact plot values, and manuscript
   claims.

These contributions characterize model behavior within one retrospective
benchmark. They do not establish a new federated optimizer, agronomic knowledge
guidance, numerical superiority to published studies, formal production
privacy, independent field generalization, or agricultural benefit.

# II. Related Work

## A. Federated optimization under heterogeneous clients

FedAvg introduced communication-efficient iterative averaging of models trained
on decentralized data. FedProx modified the local objective with a proximal
penalty intended to improve optimization under systems and statistical
heterogeneity. FedNova addressed objective inconsistency caused by unequal local
work through normalized updates. Paper 2 uses these methods as established
baselines. Its restricted implementations and shared evaluation budget are
documented explicitly; no convergence theorem or novelty claim is inherited
from the original papers.

Agricultural heterogeneity also has a support dimension that aggregation alone
does not solve. A held-out group may contain a species or composition absent
from all training groups, while a large group may dominate a pooled metric.
For this reason, the present study reports equal-group and pooled summaries
together and provides support-conditioned panels. These summaries are
descriptive for the benchmark and are not geographic confidence intervals.

## B. Federated agricultural yield prediction

Federated crop-yield prediction predates this study. A 2022
[DELCON study](https://doi.org/10.1109/DELCON54057.2022.9752836)
applied FedAvg with residual networks to soybean-yield regression. Li et al.
later proposed [federated model pruning](https://doi.org/10.1016/j.eswa.2023.122847)
for localized and efficient soybean-yield forecasting. These papers are the
closest horizontal yield-regression contexts, but their crops, records, client
construction, model objectives, splits, metrics, and computational budgets do
not match the experimental-system benchmark used here.

Other studies broaden the systems and modeling choices. Mukherjee and Buyya
compared [centralized and decentralized federated architectures](https://doi.org/10.1002/spe.3416)
with a crop-yield application. Dey et al. presented the edge/cloud recurrent
[FLyer framework](https://doi.org/10.1109/TAI.2025.3534149), while Nirosha and
Vennila combined graph and recurrent components in a
[federated yield model](https://doi.org/10.1016/j.eswa.2025.128312).
These works demonstrate that recurrent, hybrid, edge/cloud, and decentralized
agricultural federation are established directions; they are not matched
numeric baselines for the present task.

Recent studies also illustrate materially different definitions of clients.
Utku et al. studied [multi-country cereal-yield forecasting](https://doi.org/10.1007/s10791-026-10291-x)
with annual country clients and a time-aware split. Koksal et al. examined
[vertical federated crop-yield prediction](https://doi.org/10.1016/j.compag.2026.111851)
across feature holders. The former is relevant to non-IID cereal forecasting,
but country-year time series differ from crop-management systems and held-out
experimental groups. The latter partitions features vertically, whereas Paper
2 simulates horizontal clients defined by training research groups.

Cowlishaw et al. evaluated client count, imbalance, and aggregation for
[Earth-observation agricultural predictions](https://doi.org/10.1038/s41598-025-94244-2).
That study used crop-type segmentation as a proxy because suitable yield data
were unavailable at the required resolution. It informs the discussion of
agricultural client heterogeneity, but a remote-sensing segmentation task is not
direct evidence for conditional harvest regression.

Accordingly, published scores are described only in their source-specific
contexts. We do not rank Paper 2 against external numbers because the target
definitions and units, populations, client construction, partitions, metrics,
and training budgets are not aligned.

## C. Grouped evaluation and support-aware reporting

The central evaluation unit in Paper 2 is the geographic group, not an
independently sampled record. Inner model selection leaves out one training
group at a time, and outer evaluation holds out each of eight groups. This
design reduces direct site leakage but cannot create biological support that is
absent from training data. We therefore accompany aggregate error with
group-level results and seen/unseen species-composition panels. The analysis
also keeps a training-derived zero-support guard separate because it was
designed after original outer failures had been observed.

This protocol should not be interpreted as independent multi-region field
confirmation. The records come from one compiled experimental resource,
geographic support is unequal, and cultivar, density, and other management
variables are not sufficiently harmonized for causal or recommendation claims.

## D. Privacy mechanisms and simulation scope

Federated optimization limits routine centralization of raw client tables, but
that property alone is not a formal privacy guarantee. Differentially private
training requires an explicit protected unit, clipping and noise mechanism,
composition accountant, release transcript, and threat model. Secure aggregation
is a separate cryptographic requirement intended to hide individual client
updates from the aggregator, as formalized in established protocols such as
[Bonawitz et al.](https://research.google/pubs/practical-secure-aggregation-for-privacy-preserving-machine-learning/).
The Gaussian accounting used in the project follows the stated
[zCDP composition relationship](https://arxiv.org/abs/1605.02065), but the
application-level conditions required for a certified release are not closed.

Paper 2 therefore reports only a matched aggregation-noise utility simulation.
The unnoised control and noisy variants share the fixed-public representation,
selected settings, seeds, and evaluation protocol. This isolates observed
utility changes under the implemented simulation, but it does not demonstrate
production secure aggregation, private preprocessing, protection from gradient
inversion or membership inference, or end-to-end differential privacy.

## E. Position relative to prior work

The literature supports neither a first federated-yield claim nor a numerical
superiority claim. The narrower value of Paper 2 is its auditable evaluation of
conditional harvest regression across grouped experimental support: complete
baseline and adverse-result reporting, explicit equal-group versus pooled
interpretation, support panels, separation of prospective and post-hoc stages,
and a matched noise-utility study. Whether these findings transfer to new sites,
farm decisions, or established private deployments remains an open question.
