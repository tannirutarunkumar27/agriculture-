"""
File: dataset.py

Purpose:
Load processed datasets and create PyTorch Dataset
and DataLoader objects for FT-Transformer.
"""

from pathlib import Path

import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "PyTorch is required for FT-Transformer dataset utilities. "
        "Install it using 'pip install torch'."
    ) from exc

from .config import (
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY
)

# =====================================================
# Project Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# =====================================================
# Crop Recommendation Dataset
# =====================================================

class CropDataset(Dataset):
    """
    PyTorch Dataset for Crop Recommendation.
    """

    def __init__(self, csv_file):

        df = pd.read_csv(csv_file)

        self.features = torch.tensor(
            df.drop(columns=["label"]).values,
            dtype=torch.float32
        )

        self.labels = torch.tensor(
            df["label"].values,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


# =====================================================
# DataLoader Function
# =====================================================

def get_dataloaders(batch_size=BATCH_SIZE):
    """
    Create DataLoaders for training,
    validation and testing.

    Returns
    -------
    train_loader
    validation_loader
    test_loader
    """

    train_dataset = CropDataset(
        PROCESSED_DIR / "train.csv"
    )

    validation_dataset = CropDataset(
        PROCESSED_DIR / "validation.csv"
    )

    test_dataset = CropDataset(
        PROCESSED_DIR / "test.csv"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )

    return (
        train_loader,
        validation_loader,
        test_loader
    )


# =====================================================
# Testing
# =====================================================

if __name__ == "__main__":

    train_loader, validation_loader, test_loader = get_dataloaders()

    print("=" * 60)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nTraining Batches   : {len(train_loader)}")
    print(f"Validation Batches : {len(validation_loader)}")
    print(f"Test Batches       : {len(test_loader)}")

    X, y = next(iter(train_loader))

    print("\nBatch Information")
    print(f"Feature Shape : {X.shape}")
    print(f"Label Shape   : {y.shape}")

    print("\nExample Feature Vector")
    print(X[0])

    print("\nExample Label")
    print(y[0])