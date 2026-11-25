"""
File: split.py

Purpose:
Split the raw dataset into
Train / Validation / Test sets.

NOTE:
No preprocessing is performed here.
This script only splits the raw data.
"""

from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split

# =====================================================
# Project Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = RAW_DIR / "Crop_recommendation.csv"


# =====================================================
# Load Dataset
# =====================================================

def load_dataset():
    return pd.read_csv(DATASET_PATH)


# =====================================================
# Split Dataset
# =====================================================

def split_dataset(df):

    # -----------------------------
    # First Split
    # Train = 70%
    # Temp  = 30%
    # -----------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["label"]
    )

    # -----------------------------
    # Second Split
    # Validation = 15%
    # Test = 15%
    # -----------------------------

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"]
    )

    # -----------------------------
    # Save Raw Splits
    # -----------------------------

    train_df.to_csv(
        INTERIM_DIR / "train_raw.csv",
        index=False
    )

    validation_df.to_csv(
        INTERIM_DIR / "validation_raw.csv",
        index=False
    )

    test_df.to_csv(
        INTERIM_DIR / "test_raw.csv",
        index=False
    )

    return train_df, validation_df, test_df


# =====================================================
# Main
# =====================================================

def main():

    print("=" * 60)
    print("DATA SPLITTING")
    print("=" * 60)

    df = load_dataset()

    print(f"\nOriginal Dataset Size : {len(df)}")

    train_df, validation_df, test_df = split_dataset(df)

    print("\nSplitting Completed Successfully!")

    print(f"\nTraining Samples   : {len(train_df)}")
    print(f"Validation Samples : {len(validation_df)}")
    print(f"Test Samples       : {len(test_df)}")

    print("\nFiles Saved:")

    print(INTERIM_DIR / "train_raw.csv")
    print(INTERIM_DIR / "validation_raw.csv")
    print(INTERIM_DIR / "test_raw.csv")


if __name__ == "__main__":
    main()