"""
File: plot_training_history.py

Purpose:
Parse the FT-Transformer training log (trainer_check.txt) and
generate training curve plots:
  - Training vs Validation Loss
  - Training vs Validation Accuracy
  - Training vs Validation F1-Score

Saves all plots to outputs/plots/
"""

import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
# Project Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_FILE     = PROJECT_ROOT / "trainer_check.txt"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Parse Training Log
# =====================================================

def parse_training_log(log_path: Path) -> dict:
    """
    Parse trainer_check.txt and extract per-epoch metrics.

    Returns
    -------
    dict with keys:
        epochs, train_loss, val_loss,
        train_acc, val_acc,
        train_f1, val_f1
    """

    text = log_path.read_text(encoding="utf-16")

    epochs     = []
    train_loss = []
    val_loss   = []
    train_acc  = []
    val_acc    = []
    train_f1   = []
    val_f1     = []

    # Split by epoch blocks
    blocks = re.split(r"Epoch \[(\d+)/\d+\]", text)

    epoch_num = 0
    for i in range(1, len(blocks), 2):
        epoch_num = int(blocks[i])
        block     = blocks[i + 1] if i + 1 < len(blocks) else ""

        # Training metrics
        t_loss = re.search(r"Training Loss\s*:\s*([\d.]+)", block)
        t_acc  = re.search(r"Accuracy\s*:\s*([\d.]+)", block)
        t_f1   = re.search(r"F1 Score\s*:\s*([\d.]+)", block)

        # Validation section
        val_block = re.search(r"Validation.*", block, re.DOTALL)
        v_loss = v_acc = v_f1 = None
        if val_block:
            vsec   = val_block.group(0)
            v_loss = re.search(r"Validation Loss\s*:\s*([\d.]+)", vsec)
            v_accs = re.findall(r"Accuracy\s*:\s*([\d.]+)", vsec)
            v_f1s  = re.findall(r"F1 Score\s*:\s*([\d.]+)", vsec)
            v_acc  = v_accs[0]  if v_accs else None
            v_f1   = v_f1s[0]   if v_f1s  else None

        if t_loss and t_acc and t_f1 and v_loss and v_acc and v_f1:
            epochs.append(epoch_num)
            train_loss.append(float(t_loss.group(1)))
            val_loss.append(float(v_loss.group(1)))
            train_acc.append(float(t_acc.group(1)))
            val_acc.append(float(v_acc))
            train_f1.append(float(t_f1.group(1)))
            val_f1.append(float(v_f1))

    return {
        "epochs":     epochs,
        "train_loss": train_loss,
        "val_loss":   val_loss,
        "train_acc":  train_acc,
        "val_acc":    val_acc,
        "train_f1":   train_f1,
        "val_f1":     val_f1,
    }


# =====================================================
# Plot Helpers
# =====================================================

STYLE = {
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor":   "#16213e",
    "axes.edgecolor":   "#e0e0e0",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#e0e0e0",
    "ytick.color":      "#e0e0e0",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2a2a4a",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
}


def _apply_style(ax):
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_loss(history: dict):
    """Plot training vs validation loss."""

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            history["epochs"], history["train_loss"],
            color="#00d4ff", linewidth=2.5,
            marker="o", markersize=4,
            label="Training Loss"
        )
        ax.plot(
            history["epochs"], history["val_loss"],
            color="#ff6b6b", linewidth=2.5,
            marker="s", markersize=4,
            linestyle="--",
            label="Validation Loss"
        )

        ax.set_title(
            "FT-Transformer — Training vs Validation Loss",
            fontsize=14, fontweight="bold", pad=15
        )
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.legend(fontsize=11)
        _apply_style(ax)

        plt.tight_layout()
        path = OUTPUT_DIR / "ft_transformer_loss_curve.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")


def plot_accuracy(history: dict):
    """Plot training vs validation accuracy."""

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            history["epochs"], history["train_acc"],
            color="#00d4ff", linewidth=2.5,
            marker="o", markersize=4,
            label="Training Accuracy"
        )
        ax.plot(
            history["epochs"], history["val_acc"],
            color="#ff6b6b", linewidth=2.5,
            marker="s", markersize=4,
            linestyle="--",
            label="Validation Accuracy"
        )

        ax.set_title(
            "FT-Transformer — Training vs Validation Accuracy",
            fontsize=14, fontweight="bold", pad=15
        )
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=11)
        _apply_style(ax)

        plt.tight_layout()
        path = OUTPUT_DIR / "ft_transformer_accuracy_curve.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")


def plot_f1(history: dict):
    """Plot training vs validation F1-Score."""

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            history["epochs"], history["train_f1"],
            color="#00d4ff", linewidth=2.5,
            marker="o", markersize=4,
            label="Training F1-Score"
        )
        ax.plot(
            history["epochs"], history["val_f1"],
            color="#ff6b6b", linewidth=2.5,
            marker="s", markersize=4,
            linestyle="--",
            label="Validation F1-Score"
        )

        ax.set_title(
            "FT-Transformer — Training vs Validation F1-Score",
            fontsize=14, fontweight="bold", pad=15
        )
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("F1-Score", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=11)
        _apply_style(ax)

        plt.tight_layout()
        path = OUTPUT_DIR / "ft_transformer_f1_curve.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")


def plot_combined(history: dict):
    """All three metrics in one figure (3 subplots)."""

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            "FT-Transformer Training History",
            fontsize=16, fontweight="bold", y=1.02
        )

        pairs = [
            ("train_loss", "val_loss",   "Loss",     axes[0]),
            ("train_acc",  "val_acc",    "Accuracy", axes[1]),
            ("train_f1",   "val_f1",     "F1-Score", axes[2]),
        ]

        for train_key, val_key, ylabel, ax in pairs:
            ax.plot(
                history["epochs"], history[train_key],
                color="#00d4ff", linewidth=2, label="Train"
            )
            ax.plot(
                history["epochs"], history[val_key],
                color="#ff6b6b", linewidth=2,
                linestyle="--", label="Validation"
            )
            ax.set_title(ylabel, fontsize=12, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=9)
            _apply_style(ax)

        plt.tight_layout()
        path = OUTPUT_DIR / "ft_transformer_training_history.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")


# =====================================================
# Main
# =====================================================

def main():

    print("=" * 60)
    print("PLOTTING FT-TRANSFORMER TRAINING HISTORY")
    print("=" * 60)

    if not LOG_FILE.exists():
        print(f"Log file not found: {LOG_FILE}")
        return

    history = parse_training_log(LOG_FILE)

    print(f"\nEpochs parsed : {len(history['epochs'])}")
    print(f"First epoch   : {history['epochs'][0]}")
    print(f"Last epoch    : {history['epochs'][-1]}")

    print("\nGenerating plots...")

    plot_loss(history)
    plot_accuracy(history)
    plot_f1(history)
    plot_combined(history)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
