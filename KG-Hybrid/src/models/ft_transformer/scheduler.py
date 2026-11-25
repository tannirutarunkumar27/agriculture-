"""
File: scheduler.py

Purpose:
Create the learning rate scheduler used for
training the FT-Transformer model.
"""

import torch.optim.lr_scheduler as lr_scheduler

from .config import (
    SCHEDULER,
    LR_FACTOR,
    LR_PATIENCE,
    MIN_LR,
    STEP_SIZE,
    STEP_GAMMA,
    COSINE_T_MAX
)


def get_scheduler(optimizer):
    """
    Create and return the configured learning
    rate scheduler.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer

    Returns
    -------
    scheduler
        Learning rate scheduler.
    """

    if SCHEDULER == "ReduceLROnPlateau":

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="max",
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
            min_lr=MIN_LR
        )

    elif SCHEDULER == "StepLR":

        scheduler = lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=STEP_SIZE,
            gamma=STEP_GAMMA
        )

    elif SCHEDULER == "CosineAnnealingLR":

        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=COSINE_T_MAX
        )

    else:

        raise ValueError(
            f"Unsupported scheduler: {SCHEDULER}"
        )

    return scheduler


# =====================================================
# Testing
# =====================================================

if __name__ == "__main__":

    import torch
    import torch.nn as nn
    import torch.optim as optim

    model = nn.Linear(10, 2)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001
    )

    scheduler = get_scheduler(optimizer)

    print("=" * 60)
    print("SCHEDULER TEST")
    print("=" * 60)

    print(type(scheduler))

    print("\nScheduler Created Successfully.")