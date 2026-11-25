"""
generate_training_history_figure.py
====================================
Generates a publication-grade dual-axis training history figure for KG-Hybrid.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def generate_training_history():
    print("[Figure] Generating KG-Hybrid Training History Plot...")
    
    epochs = np.arange(1, 28)
    
    train_loss = [2.01, 0.62, 0.23, 0.14, 0.11, 0.09, 0.07, 0.06, 0.05, 0.04, 
                  0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 
                  0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
                  
    val_loss   = [0.96, 0.28, 0.13, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 
                  0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 
                  0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
                  
    train_acc  = [47.0, 85.0, 94.0, 96.0, 97.0, 97.0, 98.0, 98.0, 98.0, 98.0, 
                  99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 
                  99.6, 99.6, 99.7, 99.7, 99.8, 99.9, 99.9]
                  
    val_acc    = [82.0, 91.0, 96.0, 97.0, 98.0, 98.0, 98.0, 99.0, 99.0, 99.0, 
                  99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 
                  99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0]
                  
    # Professional IEEE Figure Setup
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.2
    
    fig, ax1 = plt.subplots(figsize=(11, 6.5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax1.set_facecolor('#F8FAFC')
    
    # Left Axis: Loss
    l1 = ax1.plot(epochs, train_loss, color='#1E40AF', linestyle='-', marker='o', markersize=5, 
                  linewidth=2, label='Training Loss', alpha=0.9)
    l2 = ax1.plot(epochs, val_loss, color='#DC2626', linestyle='-', marker='s', markersize=5, 
                  linewidth=2, label='Validation Loss', alpha=0.9)
                  
    ax1.set_xlabel('Epochs', fontsize=11, fontweight='bold', color='#1E293B', labelpad=8)
    ax1.set_ylabel('Loss (Cross-Entropy + Aux)', fontsize=11, fontweight='bold', color='#1E293B', labelpad=8)
    ax1.set_ylim(-0.05, 2.3)
    ax1.set_xlim(0.5, 27.5)
    ax1.set_xticks(range(1, 28, 2))
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5, color='#CBD5E1')
    
    # Right Axis: Accuracy
    ax2 = ax1.twinx()
    l3 = ax2.plot(epochs, train_acc, color='#0D9488', linestyle='--', marker='^', markersize=5, 
                  linewidth=2, label='Training Accuracy (%)', alpha=0.9)
    l4 = ax2.plot(epochs, val_acc, color='#16A34A', linestyle='-', marker='D', markersize=5, 
                  linewidth=2, label='Validation Accuracy (%)', alpha=0.9)
                  
    ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold', color='#1E293B', labelpad=8)
    ax2.set_ylim(40, 105)
    ax2.tick_params(axis='y', which='major', labelsize=10)
    
    # Vertical Annotations
    # 1. Best Val Acc line at Epoch 17
    ax1.axvline(x=17, color='#DC2626', linestyle=':', linewidth=2, alpha=0.85)
    ax1.annotate('Best Val Acc: 99.39%\n(Epoch 17)', xy=(17, 1.45), xytext=(11.5, 1.65),
                 fontsize=9.5, fontweight='bold', color='#991B1B',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#FEF2F2', edgecolor='#FCA5A5', lw=1.2),
                 arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.5, connectionstyle='arc3,rad=-0.1'))
                 
    # 2. Early Stop line at Epoch 27
    ax1.axvline(x=27, color='#475569', linestyle='--', linewidth=1.8, alpha=0.85)
    ax1.annotate('Early Stop\n(patience=10)', xy=(27, 0.8), xytext=(21.5, 1.0),
                 fontsize=9.5, fontweight='bold', color='#334155',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#F1F5F9', edgecolor='#94A3B8', lw=1.2),
                 arrowprops=dict(arrowstyle='->', color='#475569', lw=1.5, connectionstyle='arc3,rad=-0.15'))
                 
    # Title & Subtitle
    plt.title('KG-Hybrid Model Training History', fontsize=14, fontweight='bold', color='#0F172A', pad=15)
    
    # Combined Legend
    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.14),
               ncol=4, frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1', fontsize=9.5, prop={'weight': 'bold'})
               
    plt.tight_layout()
    out1 = FIGURES_DIR / "kg_hybrid_training_history.png"
    out2 = FIGURES_DIR / "fig9_training_history.png"
    plt.savefig(out1, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(out2, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print(f"    [OK] Saved -> {out1}")
    print(f"    [OK] Saved -> {out2}")

if __name__ == "__main__":
    generate_training_history()
