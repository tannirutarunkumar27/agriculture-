"""
generate_all_training_history.py
=================================
Generates training history plot strictly from recorded JSON logs in outputs/.
Models without saved per-epoch histories (e.g., Random Forest, XGBoost) are
not fabricated; only recorded neural histories are plotted.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def generate_all_models_training_history():
    history_path = TABLES_DIR / "kg_hybrid_training_history.json"
    if not history_path.exists():
        print(f"[SKIP] Missing {history_path}")
        return

    history = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = np.arange(1, len(history["train_acc"]) + 1)
    train_acc = [v * 100.0 for v in history["train_acc"]]
    val_acc = [v * 100.0 for v in history["val_acc"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    fig.suptitle('Recorded Training Curves (Non-Epoch Tree Models Omitted)', fontsize=13, fontweight='bold')

    axes[0].plot(epochs, train_acc, label='Train Acc (%)', color='#2563EB', lw=2.2)
    axes[0].plot(epochs, val_acc, label='Val Acc (%)', color='#16A34A', lw=2.2)
    axes[0].set_title('KG-Hybrid Accuracy (Recorded)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.4, ls='--')

    axes[1].plot(epochs, history["train_loss"], label='Train Loss', color='#DC2626', lw=2.2)
    axes[1].plot(epochs, history["val_loss"], label='Val Loss', color='#F59E0B', lw=2.2)
    axes[1].set_title('KG-Hybrid Loss (Recorded)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.4, ls='--')

    out = FIGURES_DIR / "all_models_training_history.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print(f"[OK] Saved {out}")


if __name__ == "__main__":
    generate_all_models_training_history()
