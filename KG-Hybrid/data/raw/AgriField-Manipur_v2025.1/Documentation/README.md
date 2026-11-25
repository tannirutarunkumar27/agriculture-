# AgriField-Manipur: Hybrid Smart Farming Dataset for Valley Agriculture (v2025.1)

## A Hybrid Environmental Agriculture Dataset for Valley-Based Smart Farming in Manipur, India

---

## Overview

The **AgriField-Manipur Dataset (v2025.1)** is a realistic hybrid dataset representing environmental and agronomic conditions of valley-based agriculture in Manipur, India.

It contains **daily time-series field data**, including:

- Temperature
- Humidity
- Soil Moisture
- Soil pH
- Rainfall

The dataset is **synthetically generated at the field level**, but all values and advisory rules are grounded in credible real-world references such as IMD climatology, NASA POWER datasets, ICAR soil reports, and agronomic standards from FAO and ICAR. This ensures the data behaves realistically while remaining reproducible and scalable.

This dataset is intended for:

- Smart farming and IoT platform prototyping
- Machine learning model development (crop advisory, irrigation, soil health, pest risk modeling)
- Predictive analytics and agricultural DSS (Decision Support Systems)
- Research, benchmarking, competitions, and academic training in precision agriculture

For citation and download, see: [Zenodo DOI](https://zenodo.org/records/17730719)

---

## Dataset Motivation

Existing public crop datasets (e.g., Kaggle Crop Dataset) are not region-specific, do not reflect environmental variations in the North-East of India, and lack sensor-style time-series structure, making them unsuitable for:

- Precision agriculture
- Field-based advisory automation
- Temporal trend ML models (LSTM, Time-Series Forecasting)

This dataset fills that gap.

---

## Dataset Structure

| Field             | Type        | Description                                                                 |
| ----------------- | ----------- | --------------------------------------------------------------------------- |
| field_id          | Integer     | Unique field identifier                                                     |
| district          | Categorical | One of 5 districts (Imphal East, Imphal West, Thoubal, Bishnupur, Kakching) |
| date              | YYYY-MM-DD  | Daily time-series data                                                      |
| temperature (°C)  | Float       | Daily average temperature                                                   |
| humidity (%)      | Float       | Daily average relative humidity                                             |
| soil_moisture (%) | Float       | Soil moisture range modeled for valley agriculture                          |
| pH                | Float       | Soil acidity value                                                          |
| rainfall (mm)     | Float       | Daily rainfall (simulated using regional climatology)                       |

---

## Geographic Coverage

| Region Type             | Districts Included                                     |
| ----------------------- | ------------------------------------------------------ |
| Valley Agriculture Belt | Imphal East, Imphal West, Thoubal, Bishnupur, Kakching |

---

## Data Generation Sources

| Reference Area               | Source Type                                                    |
| ---------------------------- | -------------------------------------------------------------- |
| Climate ranges               | India Meteorological Department (IMD), NASA POWER tool         |
| Soil and fertility values    | ICAR-NEH Region Reports, Manipur Agriculture Dept Publications |
| Crop parameter ranges        | FAO Crop Data Sheets, ICAR agronomy standards                  |
| Seasonal variation modelling | Normal distribution + random noise simulation                  |
| Expert Knowledge             | Agricultural literature + domain-verified rules                |

---

## Intended Use

- ML/AI-based crop prediction systems
- Smart irrigation planning
- Precision fertilizer advisory systems
- IoT sensor simulation, benchmarking, and competitions
- Academic and research publication use

---

## Ethical Disclaimer

- This dataset is **synthetic but realistic**.  
- All advisory rules and parameter behaviours are based on verified agricultural references.  
- It should not be used for **real-world farm decisions without expert validation**.

---

## File Contents

AgriField-Manipur_v2025.1/
├── data/
│   ├── ml_dataset.csv
│   ├── final_dataset.csv
│   ├── fields_sample.csv
│   ├── soil_by_district.csv
│   └── config.yaml
├── rules/
│   └── advisory_rules.yaml
├── code/
│   ├── agrifield_dataset_generator_full.py
│   └── visualize_dataset_preview_pdf.ipynb
├── documentation/
│   ├── README.md
│   ├── LICENSE.txt
│   ├── CHANGELOG.md
│   ├── charts/
│   │   ├── temperature_trend.png
│   │   ├── humidity_trend.png
│   │   ├── soil_moisture_distribution.png
│   │   ├── ph_distribution.png
│   │   ├── rainfall_trend.png
│   │   └── correlation_heatmap.png
│   └── AgriField_Charts.pdf
└── metadata.json

---

## License

**Creative Commons Attribution 4.0 (CC-BY 4.0)**

---

## Notes on Uniqueness

This dataset is **unique** compared to existing public datasets because:

- It contains **time-series of multiple environmental and soil parameters** at fine resolution.
- Tailored to **valley-agriculture conditions of Manipur**.
- Includes **generator code and metadata**, enabling reproducibility.
- Fills a **niche gap** in datasets for Northeast India, which are scarce in public domain.

---

## Versions

**v2025.1** – Initial stable release. See [CHANGELOG.md](CHANGELOG.md) for full details.
