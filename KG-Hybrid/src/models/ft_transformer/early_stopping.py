"""
File: early_stopping.py

Purpose:
Early stopping utility for FT-Transformer.

Stops training when the validation metric
does not improve for a specified number
of epochs.
"""

from .config import (
    EARLY_STOPPING_PATIENCE,
    MIN_DELTA
)


class EarlyStopping:
    """
    Early stopping based on validation score.

    Parameters
    ----------
    patience : int
        Number of consecutive epochs without
        improvement before stopping.

    min_delta : float
        Minimum improvement required to reset
        the patience counter.
    """

    def __init__(
        self,
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=MIN_DELTA
    ):

        self.patience = patience
        self.min_delta = min_delta

        self.best_score = float("-inf")

        self.counter = 0

        self.early_stop = False

    # =====================================================
    # Check Validation Score
    # =====================================================

    def __call__(self, score):
        """
        Evaluate validation score.

        Parameters
        ----------
        score : float
            Current validation metric
            (typically validation accuracy).

        Returns
        -------
        bool
            True if training should stop,
            otherwise False.
        """

        # Improvement found
        if score > self.best_score + self.min_delta:

            self.best_score = score

            self.counter = 0

            self.early_stop = False

        # No improvement
        else:

            self.counter += 1

            print(
                f"EarlyStopping: "
                f"{self.counter}/{self.patience}"
            )

            if self.counter >= self.patience:

                self.early_stop = True

        return self.early_stop

    # =====================================================
    # Reset
    # =====================================================

    def reset(self):
        """
        Reset the early stopping state.
        """

        self.best_score = float("-inf")

        self.counter = 0

        self.early_stop = False


# =====================================================
# Testing
# =====================================================

if __name__ == "__main__":

    early_stopping = EarlyStopping(
        patience=3,
        min_delta=0.001
    )

    validation_scores = [

        0.80,
        0.83,
        0.84,
        0.84,
        0.839,
        0.838,
        0.837,
        0.836

    ]

    print("=" * 60)
    print("EARLY STOPPING TEST")
    print("=" * 60)

    for epoch, score in enumerate(validation_scores, start=1):

        print(
            f"Epoch {epoch:02d} "
            f"| Validation Accuracy: {score:.4f}"
        )

        if early_stopping(score):

            print("\nEarly stopping triggered.")

            break

    print("\nTest Completed.")