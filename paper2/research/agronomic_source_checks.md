# Agronomic source checks — 2026-10-12

Outcome: **zero approved training rules**. Source discovery and compatibility
checks completed for current assets; numeric crop thresholds have not been
externally validated for all dataset columns. Baseline training is allowed;
knowledge-guided training is not yet allowed.

| Candidate source / asset | Checked evidence | Decision |
|---|---|---|
| Existing `KG-Hybrid/src/knowledge/knowledge_base.py` | File explicitly states manual four-point bounds, no per-cell document/page/range derivation. Generic FAO/ICAR registry text supplies no actual citation for each bound | Reject as externally validated Paper 2 rule bank; preserve Paper 1 files |
| Active prior weights / matrix validation | Shape, alignment and numeric bounds do not certify institutional derivation | No inference of physiological suitability or label-specific causality |
| Manipur v2025.1 YAML rules and v2025.3 README | Synthetic field data and rule-based advice; YAML includes generic NPK and irrigation amounts without per-rule source pages | Do not validate the generator by treating its own rules as independent empirical truth |
| [FAO ECOCROP](https://www.fao.org/geospatial/data-and-tools/data-portals/ecocrop/) | Official descriptors include crop temperature, annual precipitation, pH and environmental scope | Candidate source only. Retrieve crop-specific records/version and verify taxonomy. Annual precipitation is not daily rainfall; temperature statistic and growth-period meaning must also match |
| [TNAU rainfed maize guidance](https://agritech.tnau.ac.in/agriculture/agri_maize_rainfed_maize.html) | Separate fertilizer prescriptions for rainfed Alfisols and Vertisols, expressed as application rates N:P2O5:K2O per hectare | Not soil-measured N/P/K suitability bounds. Do not equate fertilizer oxides/rates with dataset nutrient columns or transfer Tamil Nadu recommendations unqualified to other regions |
| Canonical crop benchmark | Seven features but no original field/date identifiers or sufficiently verified nutrient measurement/precipitation period | Nutrient and rainfall rules blocked pending original metadata |
| Maharashtra download | CSV has district, soil color and numeric values but no local README/unit/measurement protocol | District labels support partitioning, not proof of observed farms. No humidity column: do not impute fictitious humidity. Nutrient and rainfall rules blocked |
| Manipur v2025.3 | README specifies daily rainfall, Celsius, relative humidity, pH and relative nutrient indices; records simulated | Physical units do not establish field validity. Relative nutrient indices cannot use kg/ha rules; annual rainfall cannot use a daily observation |

## Approval requirements for each future rule

Crop label and scientific name; source URL/DOI, version, document section/page or
table/record ID; original exact numeric bounds and units; measurement type,
temporal aggregation, region, crop variety/stage and rainfed/irrigated scope;
transformation to the dataset unit with justification; soft/hard classification;
regional exceptions; confidence and approval rationale; independence from the
benchmark/generated labels; date checked. Conflicting sources are retained
separately, not silently averaged into purported scientific consensus.

Unknown or out-of-scope rules contribute no bias. Missing features remain
unknown, not unsuitable. No four-point "optimal" interval is invented from a
source that gives only minimum/maximum values. Tuning alpha uses validation only.

Machine-readable status: `agronomic_registry.json`. Empty approved rules is an
intentional evidence gate, not an accidentally missing implementation.

## 2026-10-13 recheck and engineering gate

Reopened the two official FAO/TNAU pages above; annual precipitation and
fertilizer application semantics remain incompatible with unsupported raw
measurement assumptions. No crop-specific numerical rules approved. Local raw
documentation still does not establish canonical/Maharashtra feature contracts.

The implemented module consumes `../configs/knowledge_registry_v1.json`
(versioned empty bank), while the historical `agronomic_registry.json` retains
the source-review ledger. Generated `../results/knowledge_gate_v1/gate_report.json`
records both actual source hashes, ordered crop labels and null/unverified unit
contracts. Nonzero guidance is correctly blocked. Review declarations are not
automatic scientific certification. See `knowledge_module_specification.md`.
