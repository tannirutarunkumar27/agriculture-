"""
generate_training_history_figure.py
====================================
Generates a dual-axis training history figure for KG-Hybrid strictly from
outputs/tables/kg_hybrid_training_history.json and outputs/final_results.json.
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


def generate_training_history():
    history_path = TABLES_DIR / "kg_hybrid_training_history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing {history_path}")
    history = json.loads(history_path.read_text(encoding="utf-8"))

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_acc = [v * 100.0 for v in history["train_acc"]]
    val_acc = [v * 100.0 for v in history["val_acc"]]
    epochs = np.arange(1, len(train_loss) + 1)

    best_idx = int(np.argmax(val_acc))
    best_epoch = int(epochs[best_idx])
    best_val = float(val_acc[best_idx])

    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.2

    fig, ax1 = plt.subplots(figsize=(11, 6.5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax1.set_facecolor('#F8FAFC')

    l1 = ax1.plot(epochs, train_acc, color='#2563EB', linewidth=2.4, label='Train Accuracy (%)')
    l2 = ax1.plot(epochs, val_acc, color='#16A34A', linewidth=2.4, label='Validation Accuracy (%)')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color='#0F172A')
    ax1.annotate(
        f'Best Val Acc: {best_val:.2f}%\n(Epoch {best_epoch})',
        xy=(best_epoch, best_val),
        xytext=(max(1, best_epoch - 5), max(50.0, best_val - 10.0)),
        arrowprops=dict(arrowstyle='->', color='#16A34A', lw=1.5),
        fontsize=10, fontweight='bold', color='#16A34A'
    )

    ax2 = ax1.twinx()
    l3 = ax2.plot(epochs, train_loss, color='#DC2626', linestyle='--', linewidth=2.0, label='Train Loss')
    l4 = ax2.plot(epochs, val_loss, color='#F59E0B', linestyle=':', linewidth=2.0, label='Validation Loss')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold', color='#DC2626')

    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', frameon=True, facecolor='#FFFFFF')
    ax1.set_title('KG-Hybrid Recorded Training & Validation History', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.4, linestyle='--')

    out = FIGURES_DIR / "kg_hybrid_training_history.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print(f"[OK] Saved {out}")


if __name__ == "__main__":
    generate_training_history()
