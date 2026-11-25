# CHANGELOG – AgriField-Manipur Dataset

All notable changes to this dataset are documented in this file.  
Versioning reflects dataset evolution and documentation updates.

---

## v2025.3 – Documentation Alignment Update (2026-03-30)

### Changed
- Updated README and documentation to improve clarity and consistency with the associated research work.
- Refined terminology to better reflect the constraint-based and simulated nature of the dataset.
- No changes to dataset structure, schema, or values.

---

## v2025.2 – Publication-Aligned Release (2025-12-21)

### Added
- Expanded dataset schema to support decision-support modeling, including:
  - Crop-wise suitability scoring (`suitability_score`)
  - Binary suitability classification (`crop_suitable`)
  - Rule-based advisory outputs:
    - `irrigation_advice`
    - `fertilizer_advice`
    - `pest_risk`
  - Soil macronutrient indicators (`N`, `P`, `K`)
  - Field-level geographic coordinates (`lat`, `lon`)
- Support for **52 crops** relevant to Manipur valley agriculture.
- Temporal coverage spanning a full annual agricultural cycle (2024-12-01 to 2025-11-30).

### Changed
- Refined hybrid data generation framework combining:
  - Rule-based agronomic constraints  
  - Stochastic variability anchored to institutional baselines  
- Updated dataset structure to a **single standardized CSV** (`AgriField-Manipur_v2.csv`) for ease of reuse.
- Documentation aligned with the associated research work.
- Designated as the primary version intended for citation.

### Consistency Checks
- Performed consistency checks including:
  - Physical and agronomic range enforcement  
  - Cross-feature dependency verification  
  - Removal of missing or invalid records  

---

## v2025.1 – Initial Stable Release (2025-11-26)

- Released initial hybrid dataset representing valley agriculture conditions in Manipur, India.
- Dataset includes daily time-series variables:
  - Temperature (°C)  
  - Humidity (%)  
  - Soil moisture (%)  
  - Soil pH  
  - Rainfall (mm)  
  - Crop suitability parameters  
- Data generation combines real-world references and synthetic simulation:
  - Climate ranges: IMD, NASA POWER  
  - Soil and fertility: ICAR-NEH, Manipur Agriculture Department  
  - Crop parameters: FAO Crop Data Sheets, ICAR agronomy standards  
- Included generator code for reproducibility.
- Licensed under Creative Commons Attribution 4.0 (CC BY 4.0).