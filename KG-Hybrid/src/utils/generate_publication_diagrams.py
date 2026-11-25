"""
generate_publication_diagrams.py
===================================
Generates publication-grade, human-designed academic figures:
  1. Agricultural Knowledge Matrix Heatmap (22 Crops x 7 Features)
  2. Knowledge-Guided Fusion Mechanism Architecture Diagram
  3. Clean Human-Designed System Architecture Diagram (KG-Hybrid Framework)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.experiments.publication_evidence import save_figure, copy_artifact
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR    = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# FIGURE 1: AGRICULTURAL KNOWLEDGE MATRIX HEATMAP
# ---------------------------------------------------------
def generate_knowledge_matrix_heatmap():
    print("[1/3] Generating Agricultural Knowledge Matrix Heatmap...")
    
    matrix_path = DATA_PROCESSED / "knowledge_matrix.npy"
    if not matrix_path.exists():
        from src.knowledge.build_knowledge_matrix import crop_knowledge, CROPS, FEATURES
        matrix = np.zeros((22, 7))
        for i, crop in enumerate(CROPS):
            matrix[i] = crop_knowledge.get(crop, [0.5]*7)
    else:
        matrix = np.load(matrix_path)

    CROPS = [
        'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
        'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
        'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
        'pigeonpeas', 'pomegranate', 'rice', 'watermelon'
    ]
    FEATURES = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']

    # Academic Style Setup
    fig, ax = plt.subplots(figsize=(10, 11), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap=cmap, vmin=0.0, vmax=1.0,
                xticklabels=FEATURES, yticklabels=CROPS, ax=ax,
                linewidths=0.8, linecolor='#E5E7EB', annot_kws={"size": 9.5, "weight": "bold"})

    ax.set_title("Agricultural Knowledge Matrix (22 Crops × 7 Features)", fontsize=14, fontweight='bold', pad=18, color='#111827')
    ax.text(0.5, 1.015, "Expert Domain Prior derived from FAO / ICAR Agronomic Guidelines", transform=ax.transAxes,
            ha='center', fontsize=10, color='#4B5563')

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, fontweight='bold', color='#1F2937', rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10, fontweight='bold', color='#1F2937', rotation=0)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Agronomic Suitability Score (0.0 to 1.0)', fontsize=11, fontweight='bold', labelpad=12)

    plt.tight_layout()
    out_path = FIGURES_DIR / "agricultural_knowledge_matrix_heatmap.png"
    save_figure(plt.gcf(), out_path, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print(f"    [OK] Saved -> {out_path}")


# ---------------------------------------------------------
# FIGURE 2: KNOWLEDGE-GUIDED FUSION MECHANISM DIAGRAM
# ---------------------------------------------------------
def generate_fusion_mechanism_diagram():
    print("[2/3] Generating Knowledge-Guided Fusion Mechanism Diagram...")

    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Color Palette (Distinct semantic colors per requirement)
    C_BLUE   = {'bg': '#EFF6FF', 'border': '#2563EB', 'text': '#1E3A8A'}
    C_ORANGE = {'bg': '#FFF7ED', 'border': '#EA580C', 'text': '#7C2D12'}
    C_GREEN  = {'bg': '#ECFDF5', 'border': '#059669', 'text': '#064E3B'}
    C_PURPLE = {'bg': '#FAF5FF', 'border': '#9333EA', 'text': '#581C87'}
    C_DARK   = '#0F172A'

    def draw_node(x, y, w, h, title, subtitle=None, style=C_BLUE, rx=0.8):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.3,rounding_size={rx}",
                                      lw=2, edgecolor=style['border'], facecolor=style['bg'])
        ax.add_patch(rect)
        if title:
            cy = y + h/2 + (0.9 if subtitle else 0)
            ax.text(x + w/2, cy, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color=style['text'])
        if subtitle:
            cy_sub = y + h/2 - (0.9 if title else 0)
            ax.text(x + w/2, cy_sub, subtitle, ha='center', va='center', fontsize=8.5, color='#475569')

    def draw_edge(start, end, label=None, color='#64748B', style='->', rad=0.0):
        connectionstyle = f"arc3,rad={rad}"
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle=style, color=color, lw=2, mutation_scale=14, connectionstyle=connectionstyle))
        if label:
            mx, my = (start[0] + end[0])/2, (start[1] + end[1])/2 + 0.8
            ax.text(mx, my, label, ha='center', va='center', fontsize=8, fontweight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFFFFF', edgecolor=color, lw=1))

    # Title Banner
    ax.text(50, 96.5, "KNOWLEDGE-GUIDED FUSION MECHANISM", ha='center', va='center', fontsize=15, fontweight='bold', color=C_DARK)
    ax.text(50, 94.0, "Adaptive Gated Integration of Decision Embeddings, Context Vectors, & Domain Knowledge Prior",
            ha='center', va='center', fontsize=9.5, color='#475569')

    # Top Inputs
    draw_node(10, 81, 34, 7, "Decision Features (f_xgb)", "Box with 128-dim vector", style=C_BLUE)
    draw_node(56, 81, 34, 7, "Context Features (f_ft)", "Box with 64-dim vector", style=C_BLUE)

    # Concatenation
    draw_node(30, 69, 40, 6.5, "Concatenate [f_xgb ; f_ft]", "Vector Dimension → 192-dim", style=C_BLUE)

    draw_edge((27, 81), (42, 75.5), label="128-dim", color='#2563EB')
    draw_edge((73, 81), (58, 75.5), label="64-dim", color='#2563EB')

    # Three Parallel Paths
    # Path A: Gate Network (Orange)
    draw_node(4, 29, 28, 33, title=None, style=C_ORANGE)
    ax.text(18, 59.5, "a) Gate Network Path", ha='center', va='center', fontsize=10.5, fontweight='bold', color=C_ORANGE['text'])
    
    draw_node(6, 51, 24, 5, "Linear (192 → 128)", style=C_ORANGE)
    draw_node(6, 44, 24, 4.5, "ReLU", style=C_ORANGE)
    draw_node(6, 38, 24, 4.5, "Dropout", style=C_ORANGE)
    draw_node(6, 32, 24, 4.5, "Linear (128 → 192)", style=C_ORANGE)
    draw_node(6, 26, 24, 4.5, "Sigmoid → Gate (g)", style=C_ORANGE)

    # Path B: Forward Feature Representation (Blue)
    draw_node(36, 29, 28, 33, title=None, style=C_BLUE)
    ax.text(50, 59.5, "c) Forward Feature Path", ha='center', va='center', fontsize=10.5, fontweight='bold', color=C_BLUE['text'])
    draw_node(38, 42, 24, 8, "Concatenated Features", "Forward 192-dim Vector\nconcat = [f_xgb ; f_ft]", style=C_BLUE)

    # Path C: Knowledge Bias (Green)
    draw_node(68, 29, 28, 33, title=None, style=C_GREEN)
    ax.text(82, 59.5, "b) Knowledge Attention Path", ha='center', va='center', fontsize=10.5, fontweight='bold', color=C_GREEN['text'])
    
    draw_node(70, 50, 24, 5.5, "Knowledge Matrix K", "(22 × 7)", style=C_GREEN)
    draw_node(70, 41, 24, 5.5, "Encode K & Query concat", "K: 7 → 64; concat: 192 → 64", style=C_GREEN)
    draw_node(70, 32, 24, 6, "Crop Attention & Project", "softmax(qH_Kᵀ) → b_kg (192-d)", style=C_GREEN)

    # Edges from Concatenation to 3 Paths
    draw_edge((38, 69), (18, 62), color='#EA580C')
    draw_edge((50, 69), (50, 62), color='#2563EB')
    draw_edge((62, 69), (82, 62), color='#059669')

    # Path 1 Internal Arrows
    draw_edge((18, 51), (18, 48.5), color='#EA580C')
    draw_edge((18, 44), (18, 42.5), color='#EA580C')
    draw_edge((18, 38), (18, 36.5), color='#EA580C')
    draw_edge((18, 32), (18, 30.5), color='#EA580C')

    # Path 3 Internal Arrows
    draw_edge((82, 50), (82, 46.5), color='#059669')
    draw_edge((82, 41), (82, 38), color='#059669')

    # Fusion Equation Box (Purple)
    draw_node(14, 13, 72, 11, "GATED FUSION OPERATION",
              subtitle=r"$f_{fused} = \mathrm{LayerNorm}(g \odot \mathrm{concat} + \beta \cdot b_{kg}) \quad (\text{where } \beta = 0.1)$",
              style=C_PURPLE)

    # Edges into Fusion Box
    draw_edge((18, 26), (30, 24), label="Gate (g)", color='#EA580C')
    draw_edge((50, 29), (50, 24), label="Forward concat", color='#2563EB')
    draw_edge((82, 29), (70, 24), label="Knowledge Bias (b_kg)", color='#059669')

    # Final Output Box
    draw_node(30, 2, 40, 7, "Fused Features (192-dim)", "Output to Prediction Head MLP", style=C_BLUE)
    draw_edge((50, 13), (50, 9), color='#9333EA')

    plt.tight_layout()
    out_path = FIGURES_DIR / "knowledge_guided_fusion_architecture.png"
    save_figure(plt.gcf(), out_path, dpi=300, bbox_inches='tight', facecolor='#F8FAFC')
    plt.close()
    print(f"    [OK] Saved -> {out_path}")


# ---------------------------------------------------------
# FIGURE 3: HUMAN-DESIGNED SYSTEM ARCHITECTURE DIAGRAM
# ---------------------------------------------------------
def generate_human_system_architecture():
    print("[3/3] Generating Clean Human-Designed System Architecture Diagram...")

    fig, ax = plt.subplots(figsize=(15, 11), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Clean Color Palette
    C_BLUE   = {'bg': '#F0F9FF', 'border': '#0284C7', 'text': '#0369A1'}
    C_INDIGO = {'bg': '#EEF2FF', 'border': '#4F46E5', 'text': '#3730A3'}
    C_GREEN  = {'bg': '#F0FDF4', 'border': '#16A34A', 'text': '#15803D'}
    C_AMBER  = {'bg': '#FFFBEB', 'border': '#D97706', 'text': '#B45309'}
    C_SLATE  = {'bg': '#F8FAFC', 'border': '#475569', 'text': '#1E293B'}

    def draw_card(x, y, w, h, title=None, subtitle=None, items=None, style=C_SLATE, rx=0.5):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.3,rounding_size={rx}",
                                      lw=1.8, edgecolor=style['border'], facecolor=style['bg'])
        ax.add_patch(rect)
        
        curr_y = y + h - 2.0
        if title:
            ax.text(x + w/2, curr_y, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color=style['text'])
            curr_y -= 1.8
        if subtitle:
            ax.text(x + w/2, curr_y, subtitle, ha='center', va='center', fontsize=8.5, color='#475569')
            curr_y -= 1.6
        if items:
            for item in items:
                ax.text(x + w/2, curr_y, item, ha='center', va='center', fontsize=8, color='#334155')
                curr_y -= 1.5

    def draw_arrow(start, end, label=None, color='#64748B', style='->', rad=0.0):
        connectionstyle = f"arc3,rad={rad}"
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle=style, color=color, lw=1.8, mutation_scale=12, connectionstyle=connectionstyle))
        if label:
            mx, my = (start[0] + end[0])/2, (start[1] + end[1])/2 + 0.7
            ax.text(mx, my, label, ha='center', va='center', fontsize=8, fontweight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFFFFF', edgecolor=color, lw=0.8))

    # Header Title
    ax.text(50, 97, "KG-Hybrid End-to-End System Architecture", ha='center', va='center', fontsize=15, fontweight='bold', color='#0F172A')
    ax.text(50, 94.8, "Knowledge-Guided Hybrid Neural Network & Tree Ensemble Pipeline", ha='center', va='center', fontsize=9.5, color='#64748B')

    # 1. Input Layer
    draw_card(22, 86, 56, 6.5, title="1. Input Agricultural Data",
              subtitle="7 Soil & Climate Features: N, P, K, Temperature, Humidity, pH, Rainfall", style=C_SLATE)

    # 2. Parallel Feature Extraction Branches
    # Branch A: XGBoost
    draw_card(4, 52, 43, 30, title=None, style=C_BLUE)
    ax.text(25.5, 79.5, "Branch A: XGBoost Decision Tree Embedder", ha='center', va='center', fontsize=10.5, fontweight='bold', color=C_BLUE['text'])
    
    draw_card(6, 71, 39, 5.5, title="XGBoost Booster Ensemble", subtitle="200 Trees × 22 Classes = 4,400 Trees (Frozen)", style=C_BLUE)
    draw_card(6, 62, 39, 5.5, title="Leaf Index Extraction", subtitle="Extract Active Decision Path Leaf Indices", style=C_BLUE)
    draw_card(6, 53.5, 39, 5.5, title="EmbeddingBag Layer", subtitle="118,800 Unique Leaves → 128-dim Dense Feature (f_xgb)", style=C_BLUE)

    draw_arrow((25.5, 71), (25.5, 67.5), color=C_BLUE['border'])
    draw_arrow((25.5, 62), (25.5, 59.0), color=C_BLUE['border'])

    # Branch B: FT-Transformer
    draw_card(53, 52, 43, 30, title=None, style=C_INDIGO)
    ax.text(74.5, 79.5, "Branch B: Feature Tokenizer Transformer", ha='center', va='center', fontsize=10.5, fontweight='bold', color=C_INDIGO['text'])
    
    draw_card(55, 71, 39, 5.5, title="Feature Tokenizer", subtitle="Project 7 Continuous Features → (7 × 64-dim Embeddings)", style=C_INDIGO)
    draw_card(55, 62, 39, 5.5, title="[CLS] Token Addition", subtitle="Prepend Learnable Class Token → (8 × 64-dim)", style=C_INDIGO)
    draw_card(55, 53.5, 39, 5.5, title="Transformer Encoder", subtitle="4 Layers, 8 Heads → Extract f_ft (64-dim Context)", style=C_INDIGO)

    draw_arrow((74.5, 71), (74.5, 67.5), color=C_INDIGO['border'])
    draw_arrow((74.5, 62), (74.5, 59.0), color=C_INDIGO['border'])

    # Inputs -> Branches
    draw_arrow((50, 86), (25.5, 82), color='#64748B', rad=-0.05)
    draw_arrow((50, 86), (74.5, 82), color='#64748B', rad=0.05)

    # 3. Fusion Module
    draw_card(22, 26, 56, 22, title=None, style=C_AMBER)
    ax.text(50, 45.5, "2. Knowledge-Guided Gated Fusion Module", ha='center', va='center', fontsize=11, fontweight='bold', color=C_AMBER['text'])

    draw_card(24, 38, 52, 5, title="Adaptive Gated Network g = Sigmoid(MLP([f_xgb ; f_ft]))", subtitle="Dynamic Branch Weighting (g ∈ [0, 1]^192)", style=C_AMBER)
    
    draw_card(24, 32, 25, 4.5, title="Knowledge Matrix K (22 × 7)", subtitle="FAO/ICAR Agronomic Prior", style=C_GREEN)
    draw_card(51, 32, 25, 4.5, title="Knowledge Context b_kg", subtitle="Attention(q(concat), encode(K)) → 192-dim", style=C_AMBER)

    draw_card(24, 27, 52, 4, title="f_fused = LayerNorm(g ⊙ concat + 0.1 · b_kg)", subtitle="Forward-active gate → 192-dim dense vector", style=C_AMBER)

    # Branches -> Fusion
    draw_arrow((25.5, 52), (38, 48), label="f_xgb (128-dim)", color=C_BLUE['border'])
    draw_arrow((74.5, 52), (62, 48), label="f_ft (64-dim)", color=C_INDIGO['border'])
    draw_arrow((49, 34.2), (51, 34.2), color=C_GREEN['border'])

    # 4. Production Suite (Side Left)
    draw_card(2, 2, 27, 21, title="Production Suite", subtitle=None, items=[
        "• KNN Imputer (k=5)",
        "• Stage 1 Early Exit (tau=0.90)",
        "Coverage: see measured route audit",
        "• MC Dropout Uncertainty Quantification",
        "• Lightweight Model Option (4.04M params)",
        "• JSON API Diagnostics Schema"
    ], style=C_SLATE)

    # 5. Prediction Head (Bottom Center)
    draw_card(32, 11, 36, 11, title="3. Prediction Head & Output", subtitle=None, items=[
        "MLP Classifier: 192 → 128 → 22 Crops",
        "Softmax Probability Distribution",
        "Output: Recommended Crop",
        "+ Top-3 Probabilities & Confidence Score"
    ], style=C_SLATE)

    # 6. Auxiliary Losses (Side Right)
    draw_card(71, 2, 27, 21, title="Auxiliary Loss Guidance", subtitle=None, items=[
        "• L_CE (Cross-Entropy Loss)",
        "• L_distill (KL Tree/Transformer Distill)",
        "• L_contrast (Supervised Contrastive)",
        "L_total = L_CE + 0.5*L_distill + 0.1*L_contrast"
    ], style=C_SLATE)

    # Arrows to Output
    draw_arrow((50, 26), (50, 22), color=C_AMBER['border'])
    draw_arrow((71, 12), (68, 16.5), color='#64748B', style='<-')

    plt.tight_layout()
    out1 = FIGURES_DIR / "fig1_kg_hybrid_detailed_architecture.png"
    out2 = FIGURES_DIR / "fig1_architecture.png"
    save_figure(plt.gcf(), out1, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    save_figure(plt.gcf(), out2, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print(f"    [OK] Saved -> {out1}")
    print(f"    [OK] Saved -> {out2}")


def main():
    print("==================================================")
    print("GENERATING HUMAN-DESIGNED ACADEMIC FIGURES")
    print("==================================================")
    generate_knowledge_matrix_heatmap()
    generate_fusion_mechanism_diagram()
    generate_human_system_architecture()
    print("==================================================")
    print("[OK] ALL 3 PUBLICATION FIGURES COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    main()
