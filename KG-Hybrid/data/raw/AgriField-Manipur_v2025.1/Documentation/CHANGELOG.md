# CHANGELOG – AgriField-Manipur Dataset

All notable changes to this dataset are documented in this file.  
This follows [Semantic Versioning](https://semver.org/) principles where possible.

## v2025.1 – Initial Stable Release (2025-11-26)
- Released full hybrid dataset representing valley agriculture conditions in Manipur, India.
- Dataset contains **daily time-series measurements**:
  - Temperature (°C)
  - Humidity (%)
  - Soil moisture (%)
  - Soil pH
  - Rainfall (mm)
  - Crop suitability parameters
- Data generation combines **real-world references** and **synthetic field-level simulation**:
  - Climate ranges: IMD, NASA POWER
  - Soil and fertility: ICAR-NEH, Manipur Agriculture Department
  - Crop parameters: FAO Crop Data Sheets, ICAR agronomy standards
- Included **generator code** for reproducibility and custom simulation.
- Documentation and charts included for easy data exploration and visualization.
- Licensed under **Creative Commons Attribution 4.0 (CC-BY 4.0)**.
