# Agronomic source-transcription audit v1

Date checked: 2026-10-27

Scope: source-record identity and transcription accuracy only. This is not an
agronomic endorsement, causal yield claim, geographic-transfer approval, or
authorization to enable model guidance.

| Candidate taxon | Official FAO ECOCROP record | Registered optimal soil pH | Transcription result |
|---|---|---:|---|
| barley | [Hordeum vulgare, 1232](https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=1232) | 6.5–7.5 | MATCH |
| chickpea | [Cicer arietinum, 2479](https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=2479) | 6.0–8.5 | MATCH |
| fababean | [Vicia faba, 2146](https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=2146) | 6.0–7.0 | MATCH |
| lentil | [Lens culinaris, 7209](https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=7209) | 5.5–7.5 | MATCH |
| lupin | [Lupinus albus, 7431](https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=7431) | 5.5–6.4 | MATCH |
| pea | [Pisum sativum, 1721](https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=1721) | 5.5–7.0 | MATCH |
| wheat_aestivum | [Triticum aestivum, 2114](https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=2114) | 6.0–7.0 | MATCH |
| wheat_turgidum | [Triticum durum, 10582](https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=10582) | 6.0–7.0 | MATCH |

All eight source pages identify the recorded species and expose the registered
values in the Ecology table's `Soil PH` optimal minimum and maximum columns.
The code-level validator separately requires pH bounds within 0–14, exact unit
compatibility, identity transformation, HTTPS FAO host, matching ECOCROP record
identifier and an ISO check date.

Outstanding independent review remains unchanged: taxonomy (especially durum
wheat mapping), soil measurement protocol, geographic transfer, cultivar and
growth-stage scope, water regime, intercropping applicability, and suitability
of each interval as a soft prior rather than a treatment recommendation.
