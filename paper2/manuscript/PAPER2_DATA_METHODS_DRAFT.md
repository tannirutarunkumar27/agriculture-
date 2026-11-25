# III. Data and Problem Formulation

## A. Source data and unit of analysis

The active benchmark was constructed from version 2.0.0 of a compiled dataset
of cereal-legume field experiments. The source contained 37 experiments at nine
site labels and covered eight crop species. We treated neither individual trait
records nor crop components within an intercrop as independent observations.
Instead, one analysis row represented one `(experiment_id, management)` crop
system. A row was retained only when every expected crop component had a
verified complete-harvest seed dry-matter outcome. Component outcomes were
summed for intercrops, producing 607 crop-management systems.

The response for system (i) was harvested seed dry matter,

\[
y_i=\sum_{c\in\mathcal{C}_i} y_{ic},
\]

where \(\mathcal{C}_i\) is the set of crop components in the recorded system and
the response unit is t/ha. The prediction problem was

\[
\hat y_i=f(\mathbf{x}_i,\mathbf{s}_i,\mathbf{m}_i),
\]

where \(\mathbf{x}_i\) denotes recorded soil and contextual variables,
\(\mathbf{s}_i\) identifies the observed species composition, and
\(\mathbf{m}_i\) contains recorded management quantities. Crop composition was
therefore an input condition, not a recommended class. The estimand was the
conditional outcome of a recorded treatment bundle. It was not a causal effect,
a counterfactual yield for untested crops, or an estimate of profitability.

## B. Geographic grouping and support

Geographic groups were defined before model evaluation. Site labels within
1 km were merged by transitive union using source coordinates. Copenhagen and
Taastrup were approximately 0.37 km apart and were consequently treated as one
group. The resulting eight groups were Angers (75 systems), Auzeville (354),
Copenhagen+Taastrup (52), Grignon (64), Jyndevad (48), Kassel (4), Reading (4),
and Sanmarcoargentano (6). This policy prevented the two nearby Danish labels
from appearing on opposite sides of a split, but it did not establish that
broader spatial dependence was absent.

Group sizes were highly unequal: Auzeville contributed 354 of 607 systems,
whereas Kassel and Reading contributed four systems each. Biological support
was also uneven. Holding out Auzeville removed chickpea, lentil, and durum wheat
from training. We therefore recorded, for every test system, whether all of its
component species and whether its exact species composition had occurred in the
corresponding outer-training data. These indicators were derived from training
identities only and were not model inputs or outcome-dependent filters.

## C. Predictor contract

The fold-fitted baseline used seven recorded numerical quantities: soil
available water, clay fraction, soil pH, initial mineral nitrogen, applied
nitrogen, recorded phosphorus amount, and irrigation amount. Sowing date was
represented by sine and cosine calendar terms. Previous crop, crop season, and
recorded pest-control status were categorical. Eight fixed species-membership
indicators and a sole/intercrop indicator described crop composition.

Numeric medians, means, and standard deviations were estimated from each
training fold only. Missing numeric values received the training median and a
separate missingness indicator. An all-training-missing variable used a zero
computational placeholder with an explicit state flag; zero was not interpreted
as an observed measurement. Categorical vocabularies were likewise fitted on
training rows and contained distinct missing and unknown states. Depending on
the categories present in training, the encoded baseline dimension ranged from
45 to 49 across folds. Targets were neither imputed nor standardized with
held-out information.

Sand and silt were excluded because their source metadata descriptions appeared
interchanged and no correction could be independently justified. Realized
climate, actual harvest date, measured plant traits, target-derived variables,
geographic coordinates, country, year, experiment identifiers, and management
identifiers were not predictors. Cultivar and density were also excluded because
their coverage and cross-experiment comparability were unresolved. Recorded
fertilizer and irrigation values make this a retrospective conditional model;
the feature set is not certified as information available before sowing.

For the later fixed-public study, the same 607 outcomes, row identities, groups,
and partitions were re-encoded with a 54-dimensional, data-independent schema.
This transform used fixed definitions rather than training-fitted numerical
statistics or categorical vocabularies. It was introduced to make the matched
clipping/noise simulation mechanically interpretable; it did not make the
underlying records public or establish private preprocessing.

## D. Scientific scope

The dataset was used only for retrospective conditional harvest regression.
Ranking was disabled because management-matched alternatives were not verified
as randomized or comparable counterfactual choices, and cultivar, density, and
other treatment differences could not be safely collapsed. Agronomic guidance
was also disabled because no independently sourced rule passed the required
feature-unit, scope, and target-compatibility gates. Consequently, the study
does not provide best-crop labels, a validated agronomic knowledge mechanism,
or evidence of yield improvement caused by following model output.

# IV. Methods

## A. Nested geographic evaluation

We used nested leave-one-geographic-group-out evaluation. Each of the eight
groups served once as the outer test group. Within each outer-training set, each
of the remaining seven groups served once as an inner validation group, yielding
56 inner folds and eight outer refits. All rows sharing an experiment, site, or
geographic group remained on the same side of a split. Inner-fold assets
contained training and validation arrays only; outer-test arrays were omitted.

Hyperparameters were selected solely from the inner folds. Five fixed seeds
(123, 2024, 3407, 9999, and 2718) were then used for outer refitting. They
measured variability from initialization and training order; they were not five
independent agricultural replications. Final neural checkpoints were taken at a
fixed refit round derived from inner validation, with no outer-test early
stopping or checkpoint selection. Each planned run permitted one committed
outer inference. An inference intent, prediction cache, receipt, model identity,
source identity, data identity, and attempt journal were bound together; an
ambiguous interrupted inference was rejected rather than silently repeated.

## B. Regression baselines

Eight methods were evaluated. The training-mean predictor provided a constant
control. Ridge regression used candidate penalties \(\alpha\in\{0.1,1,10\}\).
Random forest candidates used 100 trees with minimum leaf size 1 or 3. XGBoost
candidates used 100 boosting rounds, learning rate 0.05, and maximum depth 2 or
3. Conventional candidates were selected by equal-inner-group mean validation
RMSE with first-candidate tie breaking in a frozen order.

The shared neural regressor was

\[
f_{\theta}(\mathbf{x})=mathbf{w}_2^{\mathsf T}
\operatorname{ReLU}(\mathbf{W}_1\mathbf{x}+\mathbf{b}_1)+b_2,
\]

with one hidden layer of 32 units and a scalar output. Centralized MLP, FedAvg,
FedProx, and restricted FedNova used the same architecture, initialization
scheme, mini-batch size 32, identity target scale, and mean-squared-error loss.
Optimization used constant-learning-rate SGD without momentum or weight decay.
Federated rounds used full participation and one local epoch per training group.
Logical clients were research groups in the outer-training set, not farms or
networked institutions.

For FedAvg and FedProx, selected client states were aggregated with weights
\(p_k=n_k/\sum_j n_j\), where \(n_k\) was the verified number of training rows
at client \(k\). FedProx added the local penalty

\[
\frac{\mu}{2}\lVert\theta-\theta_t\rVert_2^2,
\qquad \mu=0.01,
\]

relative to the round-start global model \(\theta_t\). In the restricted
plain-SGD FedNova implementation, client \(k\) executed \(\tau_k\) mini-batch
steps and returned \(\theta_k\). The server computed

\[
\bar\tau=\sum_k p_k\tau_k,\qquad
\theta_{t+1}=\theta_t-
\bar\tau\sum_k p_k\frac{\theta_t-\theta_k}{\tau_k}.
\]

This formulation assumed constant local learning rate, no momentum, no weight
decay, and no proximal term. Update validation rejected missing, duplicate,
stale, non-finite, shape-incompatible, or misbound client states and checked
declared sample and step counts. These integrity checks were software safeguards,
not cryptographic client authentication.

## C. Inner selection and outer refitting

The original baseline search evaluated learning rates
\(\{3\times10^{-4},10^{-3},3\times10^{-3}\}\) for each of the four neural
methods for 80 rounds on all 56 inner folds. A common learning rate per outer
fold was chosen by averaging validation RMSE equally across the seven inner
groups and four neural algorithms. For each algorithm, the outer-refit duration
was the median of the seven first-occurring minimum-RMSE checkpoint rounds at
the selected learning rate. This produced 1,120 completed validation candidates
and 320 outer evaluations: eight groups, five seeds, and eight methods.

The fixed search was intentionally not extended after outer outcomes were
available. Of 672 neural candidates, 286 selected the final allowed round, and
four of eight outer folds selected the highest learning rate in the grid. These
facts were retained as budget and grid-edge limitations rather than interpreted
as convergence or optimality.

## D. Fixed-public refit and matched noise simulation

A later development stage used the 54-dimensional fixed-public representation.
Its separate inner search evaluated the prospectively bounded public-transform
controls for 240 rounds. All eight outer groups selected target scale 1 and
learning rate 0.01; the median selected refit durations ranged from 106 to 149
rounds. 80 of 224 candidates reached the 240-round boundary, and 11 of 56
selected checkpoints had a worse training fit than their matched constant.
These adverse diagnostics were not removed and the grid was not enlarged after
outer-score exposure.

The unnoised fixed-public refit used equal client-slot averaging, full
participation, the same 32-unit regressor and five seeds. The matched follow-up
reused its selected learning rate, refit duration, initialization convention,
batch size, and four local SGD steps. Local gradients were clipped to L2 norm 1.
With learning rate \(\eta=0.01\), the resulting public model-update bound was
set to \(C=4\eta=0.04\). The server averaged seven fixed client slots. For the
noisy variants it added coordinate-wise Gaussian noise calibrated to the
replacement sensitivity

\[
\Delta=\frac{2C}{m},\qquad m=7,
\]

at noise multipliers 2, 4, and 8. A clipped, unnoised control isolated clipping
from noise. Noise streams were separated from data-order streams, and round
events committed atomically with checkpoints to prevent duplicate accounting
after recovery.

For a Gaussian event with coordinate standard deviation \(s\), the recorded
zCDP quantity was

\[
\rho=\frac{\Delta^2}{2s^2},
\]

summed over adaptive releases and converted at \(\delta=10^{-5}\) using

\[
\varepsilon=\rho+2\sqrt{\rho\log(1/\delta)}.
\]

No client-sampling amplification was assumed. This accounting described an
idealized noisy-only transcript. Nonprivate model selection, unnoised controls,
evaluation releases, deterministic research RNG, private preprocessing, and a
deployed secure-aggregation protocol were outside the guarantee. Accordingly,
the experiment was reported as a matched aggregation-noise utility simulation,
not as certified application differential privacy.

## E. Evaluation measures

For outer group \(g\) containing \(n_g\) systems, we computed

\[
\operatorname{RMSE}_g=\sqrt{\frac{1}{n_g}
\sum_{i\in g}(y_i-\hat y_i)^2}.
\]

The primary score for a seed was the arithmetic mean of the eight group RMSEs,
giving each geography equal weight. The secondary pooled RMSE was calculated
over all 607 outer predictions and therefore weighted groups by their system
counts. Equal-group and pooled MAE were reported analogously. R2 was retained
as a secondary descriptive measure and recorded as unavailable for groups in
which it was undefined; it was never replaced with zero. Negative predictions
were retained rather than clipped.

Metrics were also calculated for all systems, training-seen species, training-
unseen species, training-seen exact compositions, and training-unseen exact
compositions. The all-system result remained primary, so support panels could
not hide difficult cases. Reported variability was the sample standard deviation
across the five fixed seeds. Because folds shared training data and seeds did
not create new locations, these deviations were descriptive and were not
treated as geographic confidence intervals or confirmatory significance tests.

## F. Reproducibility safeguards

Plans were frozen before their corresponding outer inference stages. Execution
was bounded and resumable under an exclusive writer lock. Manifests bound raw
and prepared data, source snapshots, configurations, dependency versions, fold
identities, model assets, cached predictions, and receipts. Completed caches
were independently audited without loading models for repeat test inference.
Failures and recovery attempts were retained in journals, and existing result
directories were not overwritten. These safeguards establish local software
and evidence consistency; they do not constitute Linux deployment validation,
field validation, secure aggregation, or formal privacy certification.
