"""
File: utils.py

Purpose:
Utility functions for the FT-Transformer project.
"""

from __future__ import annotations

import json
import os
import random

import numpy as np
import torch


# =====================================================
# Random Seed
# =====================================================

def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# =====================================================
# Device
# =====================================================

def get_device() -> torch.device:
    """
    Return the available device.

    Returns
    -------
    torch.device
    """

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# =====================================================
# Model Parameters
# =====================================================

def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    int
    """

    return sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )


# =====================================================
# Directory
# =====================================================

def create_directory(path: str) -> None:
    """
    Create directory if it does not exist.
    """

    os.makedirs(path, exist_ok=True)


# =====================================================
# JSON Save
# =====================================================

def save_json(data: dict, file_path: str) -> None:
    """
    Save dictionary to JSON.
    """

    with open(file_path, "w") as file:

        json.dump(
            data,
            file,
            indent=4
        )


# =====================================================
# JSON Load
# =====================================================

def load_json(file_path: str) -> dict:
    """
    Load JSON file.
    """

    with open(file_path, "r") as file:

        return json.load(file)


# =====================================================
# Model Summary
# =====================================================

def print_model_summary(model: torch.nn.Module) -> None:
    """
    Print model information.
    """

    print("\n" + "=" * 60)

    print("MODEL SUMMARY")

    print("=" * 60)

    print(model)

    print("\nTrainable Parameters : {:,}".format(

        count_parameters(model)

    ))

    print("=" * 60)


# =====================================================
# Testing
# =====================================================

if __name__ == "__main__":

    import torch.nn as nn

    set_seed()

    model = nn.Linear(10, 5)

    print_model_summary(model)

    print(f"\nDevice : {get_device()}")

    create_directory("outputs")

    sample = {

        "accuracy": 0.98,

        "loss": 0.05

    }

    save_json(sample, "outputs/sample.json")

    loaded = load_json("outputs/sample.json")

    print("\nLoaded JSON")

    print(loaded)