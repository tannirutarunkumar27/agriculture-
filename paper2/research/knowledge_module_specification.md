# Step 4: class-aligned knowledge engineering specification

Status: implementation tested; **agronomic approval and real-data KG training
blocked**. This is the previously documented soft-bias candidate, not a novel
optimizer or validated agricultural recommendation system.

## Inputs and semantics

`RuleBank` binds rules to exact ordered crop labels and ordered raw numerical
features. Each rule requires dataset identity, exact unit/measurement/period,
four source-documented bounds, weight, source URL/version/record locator,
scientific name, archived source digest, independence declaration, scope and
reviewer approval. The feature contract requires independently documented
measurement provenance. Hashes identify content, not authentic scientific truth;
the reviewer must inspect the actual archived sources and dataset documentation.
Approval strings or a `verified=true` flag alone are not scientific evidence.

No implicit unit conversions or estimated units are supported. An empty rule
bank allows alpha=0 only; nonzero alpha fails loudly. Numeric min/max-only
source records cannot be promoted to four-point bounds by inventing optima.
Such records need a separately justified mechanism rather than this validator.

The neural model takes finite encoded/scaled `x_model`. Knowledge takes separate
original-unit `raw`, a Boolean ORIGINAL-observation mask and one context per
sample. Imputed values remain marked unobserved. Unknown scope fields never
match a restricted rule; '*' represents documented reviewed broad applicability.
Explicit regional exceptions must be resolved by the reviewer when specifying
scope; free-text exceptions are retained evidence, not executable exclusion rules.
For now scopes are exact region, season, crop stage and water-regime strings.
This deliberately limited vocabulary must not erase documented local exceptions.

## Formulation

For verified bounds a < b <= c < d:

    s_cj(x) = clip(min((x_j-a)/(b-a), (d-x_j)/(d-c)), 0, 1)
    q_c = sum_j m_cj w_cj s_cj / sum_j m_cj w_cj
    q_c = 0.5 if no observed/applicable rule covers crop c
    z'_c = z_c + alpha (q_c - 0.5)
    L_k = mean cross_entropy(z'(x_i), y_i)

Here m includes the observation AND rule-scope mask. Weights are fixed positive
confidence declarations in (0,1], not learned evidence scores. Coverage is
returned separately from q: neutrality must not be mistaken for known moderate
suitability. Class alignment is explicit; permuting class labels moves the
corresponding bias, rather than pooling away crop identity. This is a soft logit
bias, not a hard rejection or calibrated crop-growth probability.

The local adapter uses established plain SGD and the existing optional FedProx
penalty. Existing sample-weighted FedAvg/FedProx and restricted FedNova
aggregation are reused unchanged. Neural parameters alone enter aggregation;
fixed public rules are not averaged. The schema digest binds bank, class order,
alpha, neural layout and preprocessing identity, so incompatible updates reject.
No new convergence, privacy or optimization guarantee is implied.

Per batch of B samples, R rules and C classes, guidance costs O(BR + BC), in
addition to the MLP; accumulators require O(BC) and rule metadata O(R).

## Persistence and integration boundary

`save_guided_weights` and `load_guided_weights` store/check selected neural
weights together with model/rule/preprocessing identity and file hash. They
refuse overwrite and incompatible metadata. Load only locally produced trusted
artifacts; hashes are not signatures. These are NOT round/RNG/optimizer resume
checkpoints. A production experiment runner with round resume, validation
selection, test journaling and per-run provenance is a future separate version.

The existing frozen baseline runner and five baseline source files were not
changed. The multi-input wrapper intentionally does not plug into the old
single-input runner. Using it requires the new explicit local adapter; no
hidden reconstructed raw values, missingness assumptions or unit guesses.

## Prospective ablation design — draft, NOT frozen or executed

After independently approved rules and contracts exist:

1. Freeze source/rule/contract/class/preprocessing/code hashes and the protocol
   before training. Preserve baseline-v1 immutable historical results.
2. Select alpha from a prospectively specified nonnegative grid on validation
   only, including alpha=0. No current alpha is validation-selected.
3. Retrain matched alpha=0 and guided controls with identical initialization,
   splits, client selection, budgets and algorithm. Separate alpha=0 inference
   from evaluating a model trained with nonzero guidance: they are different
   ablations.
4. Evaluate IID, class-skew, unequal-size and district scenarios across the
   same five seeds. Keep datasets and taxonomy separate. Report coverage by
   class/region, Macro-F1, NLL and paired differences, including degradation.
5. Include explicitly labeled permutation/incorrect-rule negative controls,
   partial coverage and missing-feature sensitivity, never claiming manipulated
   rules are validated agronomy. Use test data only after validation selection.

Until the gate is resolved, only synthetic engineering tests may use nonzero
guidance. No agricultural scores are produced from fabricated rule approvals.

## Current evidence gap and source recheck

On 2026-10-13, the official [FAO ECOCROP page](https://www.fao.org/geospatial/data-and-tools/data-portals/ecocrop/)
again specified annual precipitation among environmental descriptors. The
[TNAU rainfed maize page](https://agritech.tnau.ac.in/agriculture/agri_maize_rainfed_maize.html)
again described fertilizer application quantities for different soil contexts.
Neither resolves the canonical/Maharashtra raw nutrient measurement protocol,
rainfall period or region/stage applicability. No crop-specific thresholds were
approved. Available local raw documentation remains for synthetic Manipur, not
the two current baseline datasets. Daily rainfall and relative nutrient indices
must not be equated to annual precipitation and fertilizer rates.

Required next evidence: original dataset measurement/unit dictionary, rainfall
aggregation and temperature statistic, soil pH measurement method, collection
region/time/stage and crop-specific archived source records with applicability
and reviewer rationale. Do not infer these from value ranges or fitted labels.
