# Fresh Paper 2 implementation plan

Updated 2026-10-13. Active dataset: cereal-legume trials version 2.0.0 only.
Historical canonical/Maharashtra/Manipur classification work is preserved but
does not supply new-study results. TAMASA and Chiapas remain alternative data,
not silently combined with these experiments or used for multicrop labels.

## Research scope

Primary task is retrospective conditional harvest seed-dry-matter regression
for recorded crop/management systems. No fabricated best-crop labels or economic
utility. Ranking is secondary and disabled pending experiment comparability.
Knowledge integration and formal privacy are later separately validated phases.
The old cross-entropy/logit-guidance architecture is not appropriate unchanged
for this continuous response. Do not finalize a privacy-preserving title yet.

## Order and current completion

1. Dataset provenance/outcome/comparability audit: completed and tested.
2. Target/evaluation draft: completed; no full experiment settings frozen.
3. Versioned regression rows and geography-grouped nested splitting: completed.
   Train-only preprocessing, client membership and support flags verified.
4. Fresh regression client and bounded validation-only baseline smoke: completed.
   This is not a convergence study or final performance comparison.
5. Completed: freeze and execute bounded validation-only regression search,
   inner-group aggregation/tie/refit rules and resume/failure journaling.
   All 1,120 candidates and eight outer selections verified; 14 tests passed.
   Fixed-budget limitations are recorded in restart_validation_completion_report.md.
6. Next: prospectively freeze repeated-seed outer-refit/evaluation settings,
   implement and test one-time inference receipts and bounded safe execution,
   then execute the nested repeated-seed baseline study. Report equal-geographic-group
   and pooled RMSE/MAE separately, include all-test and supported-composition
   panels, retain negative predictions and unavailable R2 explicitly.
7. Review independent agronomic evidence and design a regression-compatible
   knowledge mechanism with unit/scope gates and matched negative controls.
8. Freeze and execute knowledge ablations, then review ranking comparability.
   No observed-system ranking is universal crop-selection evidence.
9. Specify threat model, implement and independently test established privacy
   mechanisms, then run privacy-utility experiments. Do not invent cryptography.
10. Independent confirmation, figures and manuscript based on actual evidence.

## Preparation contract v1 (amends the initial task draft)

Source site labels are not sufficient geographic groups. Any cross-site source
coordinates within 1 km are conservatively merged by transitive union; existing
same-site records always stay together. Copenhagen and Taastrup are about
0.37 km apart and form one group. Thus there are **eight outer groups**, not
nine independent site holdouts. Equal-group rather than equal-source-site RMSE
is the primary planned summary under this amendment. Radius is a declared
coordinate-only policy, selected without model scores; it does not prove absence
of broader spatial correlation. Consider prospectively specified radius
sensitivity before a full-study freeze, never select a radius on test scores.

Unit: one `(experiment_id,management)` row; target is sum of complete component
harvest seed dry matter in t/ha. Original raw files are immutable. The earlier
harvest-only extractor is checked against its audited code hash and reused;
all outcomes are recomputed from raw traits, not blindly loaded from audit CSVs.

Numerical inputs: soil available water, clay fraction, pH, initial mineral N,
recorded applied N/P/irrigation, and circular sowing-calendar descriptors.
Categorical inputs: previous crop, season, recorded pest-control status.
Fixed public inputs: eight species-membership flags and sole/intercrop status.

Recorded treatments describe a retrospective conditional package. They are
NOT certified pre-sowing-known values. No unit conversion to elemental/oxide
P is made; the recorded P_amount quantity is retained without an independently
approved physiological interpretation. Sand and silt are excluded because the
source metadata appears to swap their descriptions. No guess or correction to
raw values is made. Cultivar/density controls remain unresolved and excluded
from this coarse baseline contract, so causal ranking remains disabled.

Excluded from model inputs: realized climate, actual harvest date, measured
crop traits, target-derived values, IDs, geographic coordinates/country/year.
Identifiers remain in the table for traceability and grouping ONLY. This is
not an agronomic rule approval or a presowing recommendation feature contract.

Numeric medians, means and SD are fitted on each fold's training rows only,
with separate missingness indicators. All-training-missing numerical fields
use a zero computational placeholder and an explicit state flag; zero is not
an asserted measurement. Categories have training-only vocabularies and separate
missing/unknown indicators. Feature dimensions can legitimately vary across
folds. Targets are not imputed/scaled using held-out values. Public taxonomy is
fixed; support is derived from actual training composition, not outcomes.

## Execution boundary

Splits are deterministic and independent of RNG seed or observed model score:
eight outer refits, seven inner validation choices each = 56 inner folds.
The same folds can be reused across five model initialization/client-sampling
seeds. A seed does not create a new independent regional dataset.

Inner arrays contain training and validation ONLY; outer test IDs are recorded
for isolation but test arrays are absent. Outer-refit arrays include train/test
for later final evaluation. Saved client maps contain training groups only:
six clients per inner fold, seven per outer refit. Several groups are tiny;
their unstable performance must be retained, not hidden. All components and
experimental IDs stay within their geographic partition.

Inner-validation training is complete; outer-refit/test orchestration remains
unimplemented. KG, ranking and privacy remain separately gated.
Public central preprocessing is not private preprocessing; local execution is
not network federation. Final experiment orchestration still needs safe resume,
checkpoint/test receipts and per-model asset identity. The frozen inner search
must not be silently extended: 286/672 neural candidates selected the round cap,
and four outer folds selected the highest grid LR. Any convergence sensitivity
amendment requires a prospective new version before outer-test evaluation.

Evidence and executed commands: `restart_preparation_phase_report.md`.
