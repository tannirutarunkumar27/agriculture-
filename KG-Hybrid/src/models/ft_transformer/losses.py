"""
File: loss.py

Purpose:
Create and return the loss function used for
training the FT-Transformer model.
"""

import torch.nn as nn

from .config import LOSS_FUNCTION


# =====================================================
# Loss Function Factory
# =====================================================

def get_loss_function():
    """
    Create and return the configured loss function.

    Returns
    -------
    criterion : nn.Module
        Loss function.
    """

    if LOSS_FUNCTION == "CrossEntropyLoss":

        criterion = nn.CrossEntropyLoss()

    elif LOSS_FUNCTION == "NLLLoss":

        criterion = nn.NLLLoss()

    else:

        raise ValueError(
            f"Unsupported loss function: {LOSS_FUNCTION}"
        )

    return criterion


# =====================================================
# Testing
# =====================================================

if __name__ == "__main__":

    criterion = get_loss_function()

    print("=" * 60)
    print("LOSS FUNCTION TEST")
    print("=" * 60)

    print(f"Loss Function : {criterion}")

    print("\nLoss Function Created Successfully.")