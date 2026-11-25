"""
Authoritative Agronomic Knowledge Matrix Loader & Validator
KrishiSetu Q1 ML Core — KG-Hybrid Framework

Validates strict row/column semantic alignment and numeric integrity,
and prevents silent random initialization fallbacks across training and inference.
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

FEATURES_ORDERED: List[str] = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
NUM_CROPS_EXPECTED: int = 22
NUM_FEATURES_EXPECTED: int = 7


def load_authoritative_knowledge_matrix(
    data_dir: Optional[Path] = None,
    enforce_strict_validation: bool = True
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Loads the active prior and validates its ordering and numeric integrity.
    This does not verify the claimed institutional derivation of each cell.

    Args:
        data_dir: Optional path to directory containing knowledge_matrix.csv/npy and label_encoder.pkl.
                  Defaults to project root 'data/processed'.
        enforce_strict_validation: Whether to perform full sanity and alignment checks.

    Returns:
        matrix: np.ndarray of shape (22, 7) with dtype np.float32
        crop_names: List of 22 crop class names in exact encoded order
        feature_names: List of 7 feature names in exact column order

    Raises:
        FileNotFoundError: If authoritative matrix files or label encoder are missing.
        ValueError: If shape, alignment, NaNs, or values violate agronomic constraints.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found at: {data_dir}")

    # 1. Load label encoder classes to establish authoritative class order
    le_path = data_dir / "label_encoder.pkl"
    if not le_path.exists():
        raise FileNotFoundError(
            f"Authoritative label encoder missing at: {le_path}. "
            "Cannot establish crop class row ordering without label encoder."
        )

    with open(le_path, "rb") as f:
        le_obj = pickle.load(f)

    if isinstance(le_obj, np.ndarray):
        crop_classes = list(le_obj)
    elif hasattr(le_obj, "classes_"):
        crop_classes = list(le_obj.classes_)
    else:
        raise ValueError(f"Unexpected label encoder format in {le_path}: {type(le_obj)}")

    if len(crop_classes) != NUM_CROPS_EXPECTED:
        raise ValueError(
            f"Expected {NUM_CROPS_EXPECTED} crop classes in label encoder, got {len(crop_classes)}."
        )

    # 2. Prefer CSV for semantic column/row header verification, fallback to NPY
    csv_path = data_dir / "knowledge_matrix.csv"
    npy_path = data_dir / "knowledge_matrix.npy"

    if csv_path.exists():
        km_df = pd.read_csv(csv_path, index_col=0)
        
        # Verify shape
        if km_df.shape != (NUM_CROPS_EXPECTED, NUM_FEATURES_EXPECTED):
            raise ValueError(
                f"Knowledge matrix shape mismatch in {csv_path}: expected ({NUM_CROPS_EXPECTED}, {NUM_FEATURES_EXPECTED}), "
                f"got {km_df.shape}"
            )

        # Verify column order
        cols = list(km_df.columns)
        if cols != FEATURES_ORDERED:
            raise ValueError(
                f"Knowledge matrix feature column mismatch:\nExpected: {FEATURES_ORDERED}\nActual:   {cols}"
            )

        # Verify row alignment with label encoder
        rows = list(km_df.index)
        if rows != crop_classes:
            raise ValueError(
                f"Knowledge matrix crop row mismatch with label encoder classes:\n"
                f"Expected: {crop_classes}\nActual:   {rows}"
            )

        matrix = km_df.values.astype(np.float32)

    elif npy_path.exists():
        matrix = np.load(npy_path).astype(np.float32)
        if matrix.shape != (NUM_CROPS_EXPECTED, NUM_FEATURES_EXPECTED):
            raise ValueError(
                f"Knowledge matrix NPY shape mismatch in {npy_path}: expected ({NUM_CROPS_EXPECTED}, {NUM_FEATURES_EXPECTED}), "
                f"got {matrix.shape}"
            )
    else:
        raise FileNotFoundError(
            f"Authoritative agronomic knowledge matrix missing! Neither {csv_path} nor {npy_path} exists. "
            "Silent random initialization is strictly forbidden. Please run build_knowledge_matrix.py first."
        )

    # 3. Numeric Integrity Validation
    if enforce_strict_validation:
        if np.isnan(matrix).any():
            raise ValueError("Knowledge matrix contains NaN values!")
        if np.isinf(matrix).any():
            raise ValueError("Knowledge matrix contains Infinite values!")
        if (matrix < 0.0).any() or (matrix > 1.0).any():
            raise ValueError(
                f"Knowledge matrix values out of agronomic suitability bounds [0.0, 1.0]: "
                f"min={matrix.min():.4f}, max={matrix.max():.4f}"
            )

    return matrix, crop_classes, FEATURES_ORDERED
