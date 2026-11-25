"""Prediction helpers for the FT-Transformer package."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover - handled at runtime
    torch = None


def predict_ft_transformer(model, features: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Run inference with a trained FTTransformerClassifier."""
    if torch is None:
        raise ImportError("PyTorch is required to use predict_ft_transformer.")

    if isinstance(features, pd.DataFrame):
        features = features.to_numpy(dtype=np.float32)

    inputs = torch.tensor(np.asarray(features, dtype=np.float32))
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        predictions = torch.argmax(outputs, dim=1).numpy()

    return np.asarray(predictions)
