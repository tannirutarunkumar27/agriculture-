import numpy as np
import pandas as pd
import os
from pathlib import Path

# Set paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
os.makedirs(DATA_PROCESSED, exist_ok=True)

# Define features and crops (must match label_encoder order)
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

CROPS = [
    'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
    'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
    'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
    'pigeonpeas', 'pomegranate', 'rice', 'watermelon'
]

# Expert knowledge: Suitability score (0 to 1) for each crop-feature pair
# Manually assigned weights with claimed FAO/ICAR guidance. No per-cell
# citation or deterministic agronomic-range-to-weight conversion is retained.
crop_knowledge = {
    'rice':        [0.9, 0.7, 0.8, 0.8, 0.9, 0.9, 0.9],  # Needs high N, pH neutral, high rain
    'wheat':       [0.7, 0.6, 0.6, 0.7, 0.5, 0.7, 0.5],  # Moderate everything
    'maize':       [0.8, 0.8, 0.5, 0.8, 0.6, 0.7, 0.6],
    'chickpea':    [0.5, 0.6, 0.6, 0.6, 0.4, 0.8, 0.4],
    'cotton':      [0.6, 0.7, 0.7, 0.9, 0.4, 0.6, 0.3],
    'jute':        [0.8, 0.5, 0.6, 0.9, 0.9, 0.5, 0.9],
    'coffee':      [0.6, 0.5, 0.6, 0.6, 0.8, 0.6, 0.8],
    'coconut':     [0.5, 0.6, 0.7, 0.9, 0.8, 0.6, 0.8],
    'grapes':      [0.5, 0.5, 0.6, 0.7, 0.5, 0.6, 0.4],
    'mango':       [0.4, 0.5, 0.5, 0.9, 0.7, 0.6, 0.7],
    'papaya':      [0.6, 0.5, 0.6, 0.9, 0.8, 0.6, 0.8],
    'orange':      [0.5, 0.5, 0.6, 0.7, 0.7, 0.6, 0.7],
    'pomegranate': [0.5, 0.5, 0.6, 0.8, 0.5, 0.6, 0.4],
    'banana':      [0.7, 0.6, 0.7, 0.9, 0.8, 0.6, 0.8],
    'apple':       [0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.5],
    'muskmelon':   [0.6, 0.6, 0.6, 0.9, 0.7, 0.7, 0.5],
    'watermelon':  [0.6, 0.6, 0.6, 0.9, 0.7, 0.7, 0.5],
    'blackgram':   [0.5, 0.6, 0.5, 0.7, 0.5, 0.7, 0.5],
    'mothbeans':   [0.5, 0.6, 0.5, 0.7, 0.5, 0.7, 0.4],
    'mungbean':    [0.5, 0.6, 0.5, 0.7, 0.5, 0.7, 0.5],
    'lentil':      [0.5, 0.6, 0.5, 0.6, 0.5, 0.7, 0.4],
    'pigeonpeas':  [0.5, 0.6, 0.5, 0.7, 0.5, 0.7, 0.5],
    'kidneybeans': [0.5, 0.6, 0.5, 0.6, 0.5, 0.7, 0.5]
}

# Build matrix
knowledge_matrix = np.zeros((22, 7))
for i, crop in enumerate(CROPS):
    if crop in crop_knowledge:
        knowledge_matrix[i] = crop_knowledge[crop]
    else:
        knowledge_matrix[i] = [0.5] * 7  # Default moderate

# Save
np.save(DATA_PROCESSED / "knowledge_matrix.npy", knowledge_matrix)

# Also save as CSV for review
df = pd.DataFrame(knowledge_matrix, index=CROPS, columns=FEATURES)
df.to_csv(DATA_PROCESSED / "knowledge_matrix.csv")

print(f"[OK] Knowledge Matrix saved to {DATA_PROCESSED / 'knowledge_matrix.npy'}")
print(f"Shape: {knowledge_matrix.shape}")
print("\nFirst 5 crops:\n", df.head())
