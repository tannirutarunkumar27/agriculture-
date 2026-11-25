# Ranking experiment reference resolution v1

Date checked: 2026-10-27

## Scope

The frozen Zenodo `references.xlsx` workbook contains 11 experiments with blank
original-publication cells. Nine of those experiments contribute 15 candidate
ranking blocks. This audit searched for experiment-level publications using
site, season, crop pair, cultivar, sowing/harvest dates, and management design.
It does not edit or reinterpret the frozen workbook.

## Outcome

- All 11 blank-reference experiments now have an explicit candidate or negative
  match record in `paper2/configs/ranking_reference_candidates_v1.json`.
- One link, `Grignon_inrae_2009` to DOI `10.2134/agronj2015.0440`, has an exact
  implementation-audit match on site, season, cultivars, and management factors.
  It remains unusable until an independent reviewer verifies the evidence.
- Five records have only partial or aggregate evidence: `Angers_thorigne_2007`,
  `Angers_thorigne_2009`, `Auzeville_cochard_2010`, `Grignon_inrae_2007`, and
  `Grignon_inrae_2008`. None can be treated as a complete original-experiment source.
- Three plausible-looking publications were rejected because the site or season
  is explicitly different: `Angers_thorigne_2006`, `Angers_thorigne_2008`, and
  `Auzeville_PP_2011`.
- `Auzeville_marinette_1_2015` and `Grignon_inrae_2017` still have only
  dataset-level or derived-analysis provenance, not an exact original design
  publication.

## Research safeguards

The new reference gate requires all of the following before a supplemental link
can be used: an exact site/year/design match, full treatment coverage, an
eligible original-source type, an independent reviewer, an ISO review date, a
specific evidence locator, and a written rationale. A related review article,
conference abstract, aggregate analysis, or paper from a neighboring season
cannot clear the gate.

The generated independent-review worksheet is:

`paper2/results/restart_ranking_reference_gate_v1/independent_review_template.csv`

Ranking remains disabled. Even after publication references are resolved, every
candidate block still requires independent randomization, treatment, harvest,
replicate/tie-policy, and restricted-utility approval through the existing
ranking approval gate.

## Principal sources checked

- Frozen source dataset: https://doi.org/10.5281/zenodo.14160885
- Grignon two-year crop-management study: https://doi.org/10.2134/agronj2015.0440
- Naudin et al. pea-wheat N study: https://doi.org/10.1016/j.fcr.2010.06.002
- Auzeville cultivar study: https://doi.org/10.3389/fpls.2021.733705
- Cereal-legume synthesis: https://doi.org/10.1007/s13593-014-0277-7
- Aggregate reproductive-allometry analysis: https://doi.org/10.1111/1365-2664.13979
- Western-France conference paper: https://orgprints.org/id/eprint/19416/
