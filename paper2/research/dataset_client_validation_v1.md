# Paper 2 dataset and simulated-client validation (Step 7)

Date: 2026-10-15. Task: DC01. Acceptance: regenerate dataset/feature and
client-by-fold tables from the frozen row table; independently verify every
saved partition and training-only preprocessing state; do not train or infer.
Result: **PASS_DATASET_CLIENT_AUDIT**. Machine-readable evidence is under
`results/dataset_client_validation_v1/`.

## Dataset and target

The source is Gaudio et al., *A Global Dataset Gathering 37 Field Experiments
Involving Cereal-Legume Intercrops and Their Corresponding Sole Crops*, Zenodo
record 14160885, version 2.0.0. Seven archived raw assets match the published
version-2 MD5 values and are bound to SHA-256 digests in
`results/restart_regression_data_v1/manifest.json`. This study has **607
recorded management systems** in **37 experiments** from nine source site
labels, grouped into eight geographic units. The response is total component
harvest seed dry matter in **t/ha**. There are **no crop-class labels**.

The model has **21 conceptual inputs**: nine numeric, three categorical and
nine fixed indicators. The numeric inputs are soil available water (mm), clay
fraction, soil pH, initial mineral nitrogen (kg/ha), applied N and recorded P
(each kg/ha in source metadata), irrigation (mm), and two unitless circular
sowing-date coordinates. The categorical inputs are previous crop, sowing
season and recorded pest-control status. Eight binary indicators encode species
membership; one encodes intercrop status. The source's P label/unit are
reported as recorded; no independently validated chemical conversion or
agronomic rule is inferred. Full per-field missing counts and units are in
`dataset_features.csv`.

There are **299 sole-crop and 308 intercrop systems**; neither is a prediction
class. The row table has zero duplicated row IDs, zero duplicated
`(experiment_id, management)` keys and zero missing outcomes. Its missing
inputs include soil available water (321), initial mineral N (204), previous
crop (221), irrigation (96), pH (90), and clay (15) systems. The source
metadata's sand/silt conflict is handled by exclusion. Realized climate,
harvest dates, traits, IDs, geography/year and unresolved cultivar/density
are not predictors. Recorded management does **not** establish a pre-sowing
information contract.

## Geographic groups and simulated clients

| Held-out group / potential training client | Systems | Experiments | Sole | Intercrop |
| --- | ---: | ---: | ---: | ---: |
| Angers | 75 | 11 | 33 | 42 |
| Auzeville | 354 | 10 | 187 | 167 |
| Copenhagen+Taastrup | 52 | 4 | 26 | 26 |
| Grignon | 64 | 5 | 21 | 43 |
| Jyndevad | 48 | 3 | 24 | 24 |
| Kassel | 4 | 1 | 2 | 2 |
| Reading | 4 | 1 | 2 | 2 |
| Sanmarcoargentano | 6 | 2 | 4 | 2 |

Source sites within 1 km are merged transitively before splitting;
Copenhagen and Taastrup form one group. Each outer fold holds out one group
and uses the other seven as *simulated* training clients. Each of the seven
inner folds for that outer group holds out a second group for validation and
uses the other six as clients. These are real geographic groups in a public
dataset but centrally simulated clients, **not farm-owned deployments**. Their
sizes and system compositions are strongly non-IID; they were not generated
as IID or synthetic class-skew partitions. The 392-row `fold_clients.csv`
records every client's actual training-system count, experiment count,
sole/intercrop counts, species-membership counts and target median for its
specific fold. Species counts overlap for intercrops and must not be summed
as a class distribution.

## Leakage and split findings

- All eight outer and 56 inner frozen partitions passed source, code, asset,
  row-table and array hash checks. Train/validation/test row indices cover the
  table exactly once within each fold and have no overlap in group, site or
  experiment identity. Across outer folds, every system is tested once per
  study seed. Inner NPZ assets contain training and validation arrays only.
- Saved client maps contain all and only training rows, with the held-out
  geography and inner validation geography excluded. The group and client
  counts above were recalculated from the frozen row table, not recalled.
- Each saved numerical median/mean/scale, categorical vocabulary and
  preprocessing identity was recomputed from that fold's **training rows**.
  Missing/unknown flags are explicit. The saved transformed arrays and
  outcomes match fresh reconstruction. The alternate fixed-public study uses
  a separate public transform and must not be conflated with this fitted one.
- The original prospective baseline search selected settings on inner
  validation only. However, outer-test results were subsequently viewed for
  post-hoc guard and fixed-public follow-ups. This audit proves structural
  split isolation; it **does not restore untouched external validation** or
  make those later studies independently confirmatory.
- Common saved partitions, seeds and held-out outcome definitions support the
  primary method comparison. Parameter budgets, convergence limits and
  method-specific optimization remain separate fairness qualifications; see
  `implementation_audit.md` and the original protocol/analysis.

## Reproduction and next gates

```powershell
$env:PYTHONPATH='paper2/restart'
& 'C:\Users\tanne\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -B -m unittest paper2.restart.test_dataset_client_validation paper2.restart.test_data_pipeline -v
& 'C:\Users\tanne\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -B paper2/restart/dataset_client_validation.py --data paper2/results/restart_regression_data_v1 --output paper2/results/dataset_client_validation_v1
```

The final scoped test run passed 12/12 tests; the versioned audit passed with
607 systems, eight groups and 392 client-fold records. The first test command
in the repository `.venv` failed because that environment lacks `pypdf`,
which the existing source verifier imports. The already configured bundled
runtime completed both tests and audit; no dependency was installed. The
output directory is deliberately non-overwriting. No training, inference,
original source/result mutation or manuscript edit was performed.

Step 8 can reuse archived regression outputs for a same-fold/seed comparison
of eight implemented methods; accuracy, macro-F1 and macro recall are not
defined for this continuous target. Knowledge-guidance ablations (Step 9)
remain blocked by absent validated independent rules and a regression-
compatible mechanism. The privacy simulations (Step 10) do not implement
secure aggregation or end-to-end DP. Repeated seeds (Step 11) are not
independent geographic replicates; uncertainty must reflect 37 clustered
experiments and only eight held-out groups. Steps 12--15 must retain these
boundaries instead of converting the draft into unsupported KG-FCR claims.
