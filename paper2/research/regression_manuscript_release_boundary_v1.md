# Paper 2 regression manuscript: evidence and release boundary

Date: 2026-10-16. This note applies to the versioned
`paper2/manuscript/PAPER2_REGRESSION_MANUSCRIPT_V4.md` draft. V3 supplies the
text/evidence revision; V4 places five pre-existing audited PNG figures and
binds each to the matching vector PDF. V2 remains a superseded intermediate.
This does not
replace the archived v1 manuscript or authorize new model training.

## What can be supported now

- The source is the Gaudio *et al.* Zenodo dataset, version 2.0.0,
  DOI `10.5281/zenodo.14160885` (37 field experiments). The prepared,
  complete-harvest analysis table contains 607 recorded systems from eight
  geographic holdout groups. The response is seed dry matter in t/ha; there
  are no best-crop labels or counterfactual intervention outcomes.
- The original eight-method, eight-group, five-seed outer comparison has
  320/320 independently verified records. The original inner search has
  1120/1120 verified records. Post-hoc guard, later fixed-public refit and
  matched-noise studies have separate frozen receipts and interpretation.
- The new manuscript uses the already saved post-hoc paired geographic
  sensitivity. Five seeds repeat computations on the same geography and are
  not independent site observations. Its paired table makes no p-value,
  confidence-interval or confirmatory superiority claim.
- The local reproducibility inventory verified 21/21 registered essential
  assets with zero errors. It did not reinstall dependencies, retrain a model,
  repeat test inference, or reproduce the study on another machine.

## Local verification, without model inference

From the repository root in the existing environment:

```powershell
$env:PYTHONPATH='paper2/restart;paper2'
$env:TEMP=(Resolve-Path 'paper2/results').Path
$env:TMP=$env:TEMP
& '.\.venv\Scripts\python.exe' -B -m unittest paper2.restart.test_revise_regression_manuscript paper2.tests.test_intro_bibliography paper2.restart.test_reviewer_sensitivity paper2.restart.test_outer_summary -q
& '.\.venv\Scripts\python.exe' -B -c "from paper2.restart.reproducibility_readiness import audit; r=audit(); print(r['status'],r['verified_asset_count'],r['required_asset_count'],len(r['errors']))"
```

The `results/regression_manuscript_v3/manifest.json` revision manifest records
SHA-256 identities of the archived v1 source, the saved sensitivity analysis,
the BibTeX registry and the V3 draft. The V4 manifest binds the five inserted
figures to their accepted asset manifests and to the V4 output. Do not
rerun the versioned generator over an existing output or change the archived
v1 source to make a revised claim fit. A new manuscript revision should have
a new output and manifest path.

## Not yet a portable or public release

| Boundary | Current state | Required before claiming closure |
| --- | --- | --- |
| Source control | The `paper2/` tree is untracked in the inspected Git checkout; local hashes and attempt journals exist. | Establish a reviewed commit/release history without rewriting frozen evidence. |
| Dependencies | `requirements-runtime-observed.txt` records one functioning Windows environment, not a lock. | Build a versioned, installable lock from approved dependencies and test in a fresh environment. |
| Paths | Historical manifests include absolute checkout paths. | Add a read-only relocation adapter or a clearly specified reconstruction protocol; retain archived hashes. |
| Independent reproduction | Existing-checkout asset audit passes; no clean-room or Linux end-to-end run has occurred. | Test from a separate checkout/environment, validate split and result identities, and record failures as well as passes. |
| Data/code distribution | The dataset record and DOI are public, but the inspected Zenodo page did not expose a clear license in its Rights display. Repository code and derived artifacts have not received a release-license review. | Confirm each upstream and project asset's redistribution terms before uploading data, caches, weights or a code package. Link to the source dataset rather than assuming redistribution rights. |
| Security and privacy | One process simulates client partitions; the server sees individual states. Noise is a utility simulation. | Do not use “privacy-preserving,” secure aggregation or end-to-end DP as findings without separate mechanism, threat-model, accounting and deployment audits. |

The manuscript may state that the source dataset is available through its
versioned DOI and that local code/evidence assets are retained. It must not
state that a public code package, independent reproduction, or production
federated deployment is available until those actions actually occur.
