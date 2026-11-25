# AgriField-Manipur: A Hybrid Constraint-Based Dataset for Agricultural Decision Support in Data-Scarce Valley Regions (Version 3)

## Overview

The **AgriField-Manipur Dataset (Version 3)** is a simulated field-level agricultural dataset developed to support machine learning and decision-support research in data-scarce valley agricultural regions, with a focus on the Manipur valley in North-East India.

The dataset provides daily time-series observations for simulated agricultural field plots. It is generated using a hybrid framework that combines rule-based agronomic constraints with stochastic variability. Although synthetic, all variables are bounded by agro-climatic, soil, and crop-physiological references, supporting agronomically plausible and internally consistent data.

Version 3 is a documentation-aligned release that ensures consistency between the dataset, its metadata, and the associated research work.

---

## Dataset Contents

Each record represents a daily observation for a simulated field plot and includes:

- **date** – Observation date (YYYY-MM-DD)  
- **field_id** – Unique simulated field identifier  
- **district** – Administrative district (Imphal East, Imphal West, Thoubal, Bishnupur, Kakching)  
- **lat, lon** – Approximate geographic coordinates (district-level centroids)  
- **crop** – Cultivated crop (52 supported crops)

### Environmental Variables
- **temperature (°C)**  
- **humidity (%)**  
- **rainfall (mm)**  

### Soil and Nutrient Indicators
- **soil_moisture (%)**  
- **soil_type**  
- **ph**  
- **N, P, K** – Relative macronutrient indices  

### Decision-Support Outputs
- **suitability_score (0–100)**  
- **crop_suitable**  
- **irrigation_advice**  
- **fertilizer_advice**  
- **pest_risk**

---

## Geographic Coverage

The dataset represents the **Manipur valley**, covering five key agricultural districts:

- Imphal East  
- Imphal West  
- Thoubal  
- Bishnupur  
- Kakching  

---

## Temporal Coverage

- Start date: 2024-12-01  
- End date: 2025-11-30  
- Temporal resolution: Daily  

The dataset spans a full annual agricultural cycle.

---

## Dataset Size

- Total records: 94,900  
- Simulated field plots: 260  
- Crops covered: 52  

---

## Data Generation

The dataset is generated using a hybrid approach that combines:

- Rule-based agronomic constraints  
- Stochastic variability  
- Institutional baseline references  

Key data sources used for grounding include:

- India Meteorological Department (IMD)  
- NASA POWER climate data  
- ICAR-NEH and CAU Imphal agronomic reports  
- FAO crop databases  
- NBSS&LUP soil data  

The framework simulates agronomically consistent conditions under controlled constraints, making it suitable for research in environments where direct field data are limited.

---

## Intended Use

This dataset is intended for:

- Agricultural decision-support research  
- Machine learning model development  
- Smart farming and IoT system prototyping  
- Academic and educational purposes  

---

## Disclaimer

- This dataset is **synthetic but agronomically grounded**  
- It should not be treated as direct field measurements  
- Advisory outputs are rule-based and intended for research use only  

---

## Versioning

- v2025.1 – Initial exploratory release  
- v2025.2 – Publication-aligned dataset version  
- v2025.3 – Documentation refinement and consistency update  

**Note:** The dataset file (`AgriField-Manipur_v2.csv`) remains unchanged from version v2025.2. Version v2025.3 introduces documentation and metadata refinements only.

---

## Citation

Please cite the dataset using the version-specific DOI provided on Zenodo.

---

## License

This dataset is released under the **Creative Commons Attribution 4.0 (CC BY 4.0)** license.