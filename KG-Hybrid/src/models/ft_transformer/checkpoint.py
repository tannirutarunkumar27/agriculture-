"""
File: checkpoint.py

Purpose:
Research-grade model checkpointing for FT-Transformer.

This module saves and loads:
- Model weights
- Optimizer state
- Scheduler state
- Current epoch
- Best validation score
"""

from pathlib import Path
import torch

from .config import CHECKPOINT_DIR


class ModelCheckpoint:

    def __init__(
        self,
        directory=CHECKPOINT_DIR,
        filename="best_model.pth"
    ):

        self.directory = Path(directory)
        self.directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.filepath = self.directory / filename

        self.best_score = float("-inf")

    # =====================================================
    # Save Checkpoint
    # =====================================================

    def save(
        self,
        model,
        optimizer=None,
        scheduler=None,
        epoch=0,
        best_score=0.0
    ):
        """
        Save complete training checkpoint.

        Parameters
        ----------
        model : nn.Module
        optimizer : torch.optim.Optimizer
        scheduler : torch.optim.lr_scheduler
        epoch : int
        best_score : float
        """

        if best_score > self.best_score:

            self.best_score = best_score

            checkpoint = {

                "epoch": epoch,

                "best_score": best_score,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict()
                    if optimizer is not None
                    else None,

                "scheduler_state_dict":
                    scheduler.state_dict()
                    if scheduler is not None
                    else None
            }

            torch.save(
                checkpoint,
                self.filepath
            )

            print("\n" + "=" * 60)
            print("CHECKPOINT SAVED")
            print("=" * 60)
            print(f"Epoch      : {epoch}")
            print(f"Best Score : {best_score:.4f}")
            print(f"Saved To   : {self.filepath}")

    # =====================================================
    # Load Checkpoint
    # =====================================================

    def load(
        self,
        model,
        optimizer=None,
        scheduler=None,
        device="cpu"
    ):
        """
        Load checkpoint.

        Returns
        -------
        model
        optimizer
        scheduler
        start_epoch
        best_score
        """

        if not self.filepath.exists():

            print("\nNo checkpoint found.")

            return (
                model,
                optimizer,
                scheduler,
                0,
                float("-inf")
            )

        checkpoint = torch.load(
            self.filepath,
            map_location=device
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if (
            optimizer is not None
            and checkpoint["optimizer_state_dict"] is not None
        ):
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        if (
            scheduler is not None
            and checkpoint["scheduler_state_dict"] is not None
        ):
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        start_epoch = checkpoint["epoch"] + 1

        best_score = checkpoint["best_score"]

        print("\n" + "=" * 60)
        print("CHECKPOINT LOADED")
        print("=" * 60)
        print(f"Resume Epoch : {start_epoch}")
        print(f"Best Score   : {best_score:.4f}")

        return (
            model,
            optimizer,
            scheduler,
            start_epoch,
            best_score
        )


# =====================================================
# Testing
# =====================================================

if __name__ == "__main__":

    import torch.nn as nn

    model = nn.Linear(10, 2)

    checkpoint = ModelCheckpoint()

    print("=" * 60)
    print("CHECKPOINT TEST")
    print("=" * 60)

    # Save a dummy checkpoint
    checkpoint.save(
        model=model,
        optimizer=None,
        scheduler=None,
        epoch=5,
        best_score=0.95
    )

    # Load the checkpoint
    (
        model,
        optimizer,
        scheduler,
        epoch,
        best_score
    ) = checkpoint.load(model)

    print("\nCheckpoint Test Completed Successfully.")