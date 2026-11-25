"""
File: data_loader.py

Purpose:
Loads the Crop Recommendation dataset and performs basic inspection.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "Crop_recommendation.csv"


def load_dataset(csv_path: str | Path = DATASET_PATH) -> pd.DataFrame:
    """Load the crop recommendation dataset from a CSV file."""
    dataset_path = Path(csv_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    return pd.read_csv(dataset_path)


def dataset_summary(df):

    print("=" * 60)
    print("DATASET PREVIEW")
    print("=" * 60)
    print(df.head())

    print("\n")

    print("=" * 60)
    print("DATASET SHAPE")
    print("=" * 60)
    print(df.shape)

    print("\n")

    print("=" * 60)
    print("COLUMN NAMES")
    print("=" * 60)
    print(df.columns.tolist())

    print("\n")

    print("=" * 60)
    print("DATA TYPES")
    print("=" * 60)
    print(df.info())

    print("\n")

    print("=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    print(df.isnull().sum())

    print("\n")

    print("=" * 60)
    print("DUPLICATE ROWS")
    print("=" * 60)
    print(df.duplicated().sum())

    print("\n")

    print("=" * 60)
    print("STATISTICAL SUMMARY")
    print("=" * 60)
    print(df.describe())


if __name__ == "__main__":

    dataframe = load_dataset()

    dataset_summary(dataframe)