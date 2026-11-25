# Paper 2 restart: step 1 feasibility audit

Date: 2026-10-13. Decision: **conditional feasibility for restricted crop-system
outcome modelling/comparison, not unrestricted best-crop recommendation**.
Existing Paper 2 implementation, its 235-run classification study, Paper 1 and
the open LaTeX document remain unchanged. No training, actual split selection,
preprocessing fitting or recommendation-label creation occurred.

## Source and reproducibility

The seven source files are directly under `KG-Hybrid/data/raw/`. All MD5 hashes
match published Zenodo record 14160885 version 2.0.0. SHA-256 hashes, code hash
and count evidence are in `../results/restart_feasibility_v1/audit.json`. Raw
hashes were rechecked after analysis: no source mutation. Integrity does not
certify scientific validity. Source workbooks/PDF were read only.

Reviewed documentation: data_report.pdf pages 2-5, metadata.xlsx and
references.xlsx. The report's page-3 diagram retains an older 33,351 trait count;
the actual version-2 CSV and report page 5 have 34,737. References are supplied
for 26 of 37 experiments; 11 lack publication references and need follow-up.

Bundled runtime: Python 3.12.14, pandas 3.0.1, numpy 2.3.5, openpyxl 3.1.5,
pypdf 6.10.0. This did not change the frozen baseline environment.

Executed successfully:

```powershell
& 'C:/Users/tanne/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B paper2/restart/audit_cereal_legume.py --output paper2/results/restart_feasibility_v1
& 'C:/Users/tanne/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B paper2/restart/test_audit_cereal_legume.py --output paper2/results/restart_audit_tests_v1
& '.\.venv\Scripts\python.exe' -B paper2/scripts/freeze_baseline_protocol.py --output paper2/results/frozen_baseline_protocol_v1 --verify-only
```

Audit refuses existing outputs; rerun only to a new version directory.

## Verified observations and harvest outcomes

| Level | Count | Interpretation |
| --- | ---: | --- |
| Experiments | 37 | Site/plot/year records |
| Sites / countries | 9 / 5 | Uneven geographic coverage |
| Species | 8 | Five legumes and three cereals |
| Management component rows | 915 | One per sole crop, two per intercrop |
| Management systems | 607 | 299 sole crops, 308 intercrops |
| Trait measurements | 34,737 | Repeated dates/variables, not independent samples |
| Daily climate rows | 27,024 | Gridded NASA POWER, repeated per experiment |

Trial IDs, climate `(experiment_id,date)` keys and management
`(experiment_id,management,species)` keys are unique. All checked foreign keys
resolve. Joining all tables on experiment ID alone would multiply records and
assign outcomes incorrectly; component-level joins are explicitly validated.

Metadata defines biomass_seed as reproductive-organ dry matter in t/ha,
corresponding to yield ONLY at harvest. Of 916 measurements, 915 exactly match
documented component harvest dates and are finite/nonnegative, yielding one
outcome per component. The one earlier measurement is excluded from audit
harvest extraction, not deleted from raw data. Traits/management harvest dates
and crop types must agree; duplicate outcomes reject, never silently average.

All 607 systems have complete component outcomes under this rule. Intercrop
audit totals SUM component seed dry masses, not average them. This descriptive
mass total is NOT profit, nutritional utility, land-equivalent ratio or a
best-crop label. Component harvest dates can differ and are retained. Raw plot
replication/variance is not represented explicitly in this tidy audit table.

## Preliminary comparison screen

Complete matching values required: experiment, season, sowing date, applied
N, applied P, irrigation and pest-control status. Missing irrigation is unknown,
not zero. Each block must have at least two distinct species/system choices
and complete outcomes. Result: **47 candidate blocks covering 450 systems**:
28 have three alternatives, 13 have two and six have five.

These are candidate management-matched blocks, NOT causal certification or 47
independent experiments. Several belong to the same experiment/site. Cultivar,
density, mixture proportion, harvest timing, original randomization and
unmeasured management require review. No winner labels or ranking/model scores
were created. Recorded management effects cannot be attributed solely to species.

Predicting the observed species from species_mix, cultivar or crop-specific
density risks target leakage. Conditional yield modelling instead treats
candidate identity as an explicit intervention input; this is a different task.
Outcome differences do not establish what untested crops would have achieved.

## Site/client and split feasibility

| Species | Sites represented |
| --- | ---: |
| Pea | 9 |
| Barley | 7 |
| Faba bean / soft wheat | 3 each |
| Lupin | 2 |
| Chickpea / lentil / durum wheat | 1 each |

**Auzeville holds 354/607 systems.** Holding it out leaves chickpea, lentil and
durum wheat absent from training. Other site sizes: Angers 75, Grignon 64,
Copenhagen and Jyndevad 48 each, Sanmarcoargentano 6, Kassel/Reading/Taastrup
four each. Country clients would be highly imbalanced and confounded with
species. These are experimental research sites, not independently measured farms.

Saved leave-site-out counts are feasibility diagnostics, NOT selected splits.
An ordinary all-species balanced five-fold site-generalization protocol is not
supported. Consider a narrower supported choice set, explicit unseen-species
tests, site/temporal grouping and minimum client size. Keep every component and
measurement of a unit together. Repeated plot/site/year records and nearby
sites also need geographic overlap review before final partitioning.

## Units, missingness and decision-time leakage

Soil pH is missing for 7/37 experiments; initial mineral N for 19/37.
Metadata distinguishes initial soil mineral N from applied fertilizer N.
Full missingness/dictionary saved; no imputation fitted and no units invented.
Climate has no nulls or -999 sentinels in current CSV: this narrow check does
not certify all ranges or field-station accuracy. Weather is gridded data.

Realized in-season weather, measured plant traits, actual harvest dates and
yield-derived values cannot be inputs to genuinely pre-sowing advice. A future
protocol must fix a decision timestamp and use only available observations or
explicit forecasts/climatology. Retrospective full-season prediction is not
prospective crop selection. Prices, costs and farmer utility are not supplied
by this audit. Dry mass alone does not identify economically optimal crops.

Documented units do not automatically approve physiological suitability rules.
No independent agronomic rule bank was approved here; tropical maize fertilizer
guidance must not be transferred to European cereal/legume soil bounds.

## Saved outputs and verification

New implementation: `../restart/audit_cereal_legume.py` and
`../restart/test_audit_cereal_legume.py`.

`../results/restart_feasibility_v1/` contains audit.json, traced component and
system outcome tables, preliminary matched blocks, species coverage,
leave-site-out feasibility, missingness, variable dictionary and extracted
source report. They are AUDIT intermediates, not an approved training dataset.

Eight synthetic audit-logic tests passed, zero failures/errors; evidence in
`../results/restart_audit_tests_v1/`. Tests cover sums, early-date exclusion,
duplicate/conflicting/unmatched keys, incomplete components, unknown/different
management and overwrite refusal. They do not establish agricultural validity.
The prior 96-test model suite was not rerun; these eight tests are separately
scoped. Existing frozen baseline package verification PASS: original 235-run
plan, 45 partitions and protocol digest unchanged. Historical freeze flags
do not describe current training completion.

## Next decision

Proceed to step 2: **define the restricted target and prospective protocol**,
not training. Recommended candidate: conditional crop-system harvest-outcome
modelling, with secondary ranking ONLY among options tested in defensible
comparison blocks. Review original methods, intervention controls and density/
cultivar support, and justify utility before constructing recommendation labels.
Separate sole-crop and intercrop interpretation.

This is not a 22-crop classifier replacement, unrestricted optimal-selection
proof or validation for Indian regional agriculture. If those claims remain
mandatory, this dataset alone is insufficient. Formal privacy, distributed
execution, agronomic approvals and independent confirmation remain future
phases. Finalize the Paper 2 title/method only after accepting the task limits.
