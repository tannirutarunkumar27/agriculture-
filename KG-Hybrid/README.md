# KG-Hybrid: Knowledge-Guided Hybrid Deep Learning Engine

This directory contains the primary research, model architectures, training workflows, datasets, and output figures for the **Knowledge-Guided Hybrid (KG-Hybrid)** Crop Recommendation System.

For full project documentation, performance benchmarks, and quick-start instructions, please refer to the main repository [README.md](../README.md).

## Submodule Organization
- **`data/`**: Raw datasets (`Crop_recommendation.csv`), interim data splits, processed knowledge matrices, and leaf embeddings.
- **`src/models/`**: Implementation of baseline architectures, FT-Transformer, Concatenation Hybrid, and KG-Hybrid variants.
- **`src/knowledge/`**: Domain knowledge matrix extraction and agronomic rule encoders.
- **`src/ablation/`**: Multi-phase ablation study scripts (A1 to A7).
- **`src/robustness/`**: Robustness testing under missing feature scenarios and noise injection.
- **`src/inference/`**: Lightweight early-exit production pipeline and latency benchmarks.
- **`src/utils/`**: IEEE-compliant figure generators and visualization tools.
- **`outputs/`**: Model weight checkpoints (`.zip` format), ablation checkpoints, figures, plots, and evaluation reports.
