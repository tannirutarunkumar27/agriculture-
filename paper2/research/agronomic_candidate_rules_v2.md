# Agronomic candidate-rule review v2

## Outcome

Eight independently sourced, unit-compatible candidate rules are registered for
soil pH. They cover all crop taxa in the cereal-legume dataset. The schema audit
passes all eight records, but no rule has a declared domain approval and
agronomic guidance remains disabled.

These are candidate suitability intervals from FAO ECOCROP, not causal yield
effects, treatment recommendations, or validated rules for the five-country
experimental sample. ECOCROP is a generalized species database. Variety,
growth-stage, geography, soil measurement protocol, and intercropping effects
still require agronomic review.

## Candidate intervals

| Dataset taxon | FAO ECOCROP taxon and record | Optimal soil pH |
|---|---|---:|
| barley | [Hordeum vulgare, code 1232](https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=1232) | 6.5–7.5 |
| chickpea | [Cicer arietinum, code 2479](https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=2479) | 6.0–8.5 |
| fababean | [Vicia faba, code 2146](https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=2146) | 6.0–7.0 |
| lentil | [Lens culinaris, code 7209](https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=7209) | 5.5–7.5 |
| lupin | [Lupinus albus, code 7431](https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=7431) | 5.5–6.4 |
| pea | [Pisum sativum, code 1721](https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=1721) | 5.5–7.0 |
| wheat_aestivum | [Triticum aestivum, code 2114](https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=2114) | 6.0–7.0 |
| wheat_turgidum | [Triticum durum, code 10582](https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=10582) | 6.0–7.0 |

The durum source crop view lists *Triticum turgidum* var. *durum* as a synonym;
that taxonomy mapping requires reviewer confirmation. The local metadata defines
`soil_pH` with unit `-`, so the transformation is identity.

## Excluded candidate features

ECOCROP also reports temperature requirements and annual rainfall. They were not
registered because the active pre-sowing regression table contains neither a
source-compatible annual-rainfall feature nor a defined temperature statistic.
Realized in-season climate is intentionally disabled. Converting annual
rainfall into irrigation, daily rainfall, or growing-season water would be a
semantic error.

## Required independent review

For every rule, an agronomist must confirm taxonomy, geographic transfer,
measurement protocol, cultivar/stage scope, water regime, intercropping use, and
whether an ECOCROP optimal interval is appropriate as a soft model prior. The
reviewer and review date must be recorded. Even a declared approval remains
disabled until a separate independent audit authorizes a versioned guidance
study.
