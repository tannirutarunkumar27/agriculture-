# RR03: Experiment Dependence and Category Hash Audit — Resolution
# Status: RESOLVED_WITH_DOCUMENTED_LIMITATIONS
# Completed: 2026-11-16

## Summary

RR03 examined two concerns raised in the reviewer response: (1) experiment-level
clustering (dependence among the 607 system rows), and (2) category hash
collisions in the `previous_crop` encoding.

Both concerns are now fully documented as disclosed limitations. No scientific
claim has been altered to hide these issues; they are explicitly stated in the
manuscript Limitations section (Section VII).

---

## 1. Experiment-Level Clustering (Dependence)

### Finding
The 607 system rows come from 37 field experiments. All 37 experiments contain
multiple rows — the minimum is 3 and the maximum is 64 (Auzeville_PP_2015).
The data are therefore **clustered**, not independently sampled.

### Group Distribution
| Group              | Systems | Experiments |
|--------------------|--------:|------------:|
| Auzeville          |     354 |          10 |
| Angers             |      75 |          11 |
| Grignon            |      64 |           5 |
| Copenhagen+Taastrup|      52 |           4 |
| Jyndevad           |      48 |           3 |
| Sanmarcoargentano  |       6 |           2 |
| Reading            |       4 |           1 |
| Kassel             |       4 |           1 |

Auzeville accounts for **58.3%** of all systems and **27.0%** of all experiments.

### Implication for Claims
- Conventional significance tests assume independently sampled rows.
  With 37 experiment clusters, effective sample size is far less than 607.
- **Action taken:** The manuscript reports only **descriptive** seed-level
  standard deviations, never geographic confidence intervals or p-values.
  Seed SD is labelled as a computational stability indicator, not a
  statistical confidence measure. This is explicitly stated in Section V
  and Section VII.
- **No change** to model training or evaluation is required; the limitation
  is already correctly disclosed in the evidence record.

---

## 2. Previous-Crop Category Hash Collisions

### Finding (from reviewer_data_audit_v1/analysis.json)
The `previous_crop` field has:
- **13 unique non-missing categories** in 386 non-missing rows (221 missing)
- **7 occupied bins** out of 8 (hash modulo 8 encoding)
- **10 categories fall into colliding bins** (share a bin with ≥1 other category)
- **9 distinct colliding pairs** — categories that cannot be distinguished
  from each other at prediction time by this encoding alone

`crop_season` and `pest_control` have **no collisions** (2 unique values each,
2 bins each).

### Implication for Claims
Hash collisions in `previous_crop` mean that the model **cannot distinguish**
between the 9 colliding category pairs at inference time. For example, if
categories "wheat" and "barley" hash to the same bin, the model treats them
identically for the `previous_crop` feature.

- **Decision-time availability:** `previous_crop` describes the previous
  season's crop. This information is typically available before sowing the
  current season and is a valid pre-sowing feature.  
  **However**, it is NOT certified as universally available across all 37
  experiments and geographic groups. Its missingness rate of 36.4% (221/607)
  means it is absent for more than a third of records.

### Resolution
- **Disclosure:** The 9 colliding pairs and 36.4% missingness are documented
  here and referenced in the manuscript Limitations.
- **Mitigation:** No claim is made that relies on the precise categorical
  identity of `previous_crop` beyond what the encoded hash supports.
- **Ranking gate:** Crop-system ranking remains BLOCKED (R07). The collision
  issue is one additional reason ranking cannot proceed: tied utility values
  arising from hash-indistinguishable `previous_crop` categories would require
  an explicit tie-breaking rule that does not currently exist.
- **Model results unaffected:** The model was trained on the encoded
  representation. Results are valid for the encoded feature space. The
  collision means external interpretation must be cautious about which
  `previous_crop` identity a prediction corresponds to when collision occurs.

---

## 3. RR03 Status After This Document

| Sub-task | Status |
|---|---|
| Experiment clustering documented | ✅ COMPLETE |
| Effective sample size caveat in manuscript | ✅ COMPLETE |
| Seed SD correctly labelled (not CI) | ✅ COMPLETE |
| Hash collision audit completed | ✅ COMPLETE |
| Collision pairs counted and recorded | ✅ COMPLETE (9 pairs) |
| Decision-time availability assessed | ✅ COMPLETE (not fully certified; missingness noted) |
| Ranking gate updated with collision reason | ✅ COMPLETE |

**RR03 is now RESOLVED_WITH_DOCUMENTED_LIMITATIONS.**
No model retraining, re-inference, or new outer-test exposure occurred.
All findings are adverse-result disclosures, not corrections to cached results.

---

## 4. Evidence Binding

| Item | Value |
|---|---|
| Data audit source | `paper2/results/reviewer_data_audit_v1/analysis.json` |
| Source data SHA-256 | `b052a9e2c6895d8df9863b5bbf23c4b32895569f4ce9b9a3ebaffe209cb78aa2` |
| Completion date | 2026-11-16 |
| Reviewer claim matrix | `paper2/research/reviewer_claim_evidence_matrix_v1.md` |
