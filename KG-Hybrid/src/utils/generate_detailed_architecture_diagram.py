"""
generate_detailed_architecture_diagram.py
============================================
Generates a detailed publication-quality system architecture diagram for the KG-Hybrid Framework.
Includes:
  1. Input Agricultural Data Layer (7 features: N, P, K, Temp, Humidity, pH, Rainfall)
  2. Branch A: XGBoost Leaf Embedding (200 Trees / 4400 Total Trees, Leaf Index Extraction, EmbeddingBag 118,800 leaves -> 128-dim / 32-dim)
  3. Branch B: FT-Transformer (Feature Tokenizer 7x64-dim, CLS Token, 4-Layer 8-Head Encoder, Context Feature 64-dim)
  4. Agricultural Knowledge Base (FAO/ICAR Guidelines) -> Anchored Knowledge Matrix (22x7)
  5. Knowledge-Guided Gated Fusion Module (forward-active gate plus sample-conditioned knowledge attention)
  6. Auxiliary Losses (KL Distillation + Supervised Contrastive)
  7. Prediction Head (MLP: 192 -> 128 -> 22 Crops) + Softmax Output & Confidence Score
  8. Production Enhancements (KNN Imputer k=5, Early Exit Stage 1 tau=0.90, MC Dropout Uncertainty 10 passes, JSON API Schema)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.experiments.publication_evidence import save_figure, copy_artifact
FIGURES_DIR  = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def create_diagram():
    fig, ax = plt.subplots(figsize=(16, 12), dpi=300)
    fig.patch.set_facecolor('#080B10')
    ax.set_facecolor('#080B10')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Color Palette (Dark Mode Luxury Theme)
    C_CARD       = '#111622'
    C_BORDER     = '#30363D'
    C_TEXT       = '#F0F6FC'
    C_TEXT_MUTED = '#8B949E'
    
    C_INPUT      = '#38BDF8'  # Sky Blue
    C_BRANCH_A   = '#0EA5E9'  # Ocean Blue
    C_BRANCH_B   = '#6366F1'  # Indigo
    C_KNOWLEDGE  = '#10B981'  # Emerald Green
    C_FUSION     = '#F97316'  # Vivid Orange
    C_HEAD       = '#EC4899'  # Rose / Pink
    C_AUX        = '#A855F7'  # Purple
    C_PROD       = '#06B6D4'  # Cyan

    def draw_box(x, y, w, h, title, subtitle=None, details=None, border_color=C_BORDER, bg_color=C_CARD, text_color=C_TEXT, linestyle='-'):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.0",
                                      linewidth=1.8, edgecolor=border_color, facecolor=bg_color, linestyle=linestyle)
        ax.add_patch(rect)
        
        cy = y + h - 1.8
        if title:
            ax.text(x + w/2, cy, title, ha='center', va='center', fontsize=10.5, fontweight='bold', color=border_color if border_color != C_BORDER else C_TEXT)
            cy -= 1.8

        if subtitle:
            ax.text(x + w/2, cy, subtitle, ha='center', va='center', fontsize=8.5, fontweight='bold', color=text_color)
            cy -= 1.5

        if details:
            for item in details:
                ax.text(x + w/2, cy, item, ha='center', va='center', fontsize=8.0, color=C_TEXT_MUTED)
                cy -= 1.4

    def draw_arrow(start, end, color=C_BORDER, label=None, style="->", rad=0.0):
        connectionstyle = f"arc3,rad={rad}"
        arrowprops = dict(arrowstyle=style, color=color, lw=1.8, mutation_scale=12, connectionstyle=connectionstyle)
        ax.annotate('', xy=end, xytext=start, arrowprops=arrowprops)
        if label:
            mx = (start[0] + end[0]) / 2
            my = (start[1] + end[1]) / 2 + 1.0
            ax.text(mx, my, label, ha='center', va='center', fontsize=7.5, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#080B10', edgecolor='none', alpha=0.8))

    # --- TITLE BANNER ---
    ax.text(50, 97.5, "KG-HYBRID SYSTEM ARCHITECTURE FRAMEWORK", ha='center', va='center', fontsize=16, fontweight='bold', color=C_TEXT)
    ax.text(50, 95.5, "Knowledge-Guided Hybrid Neural Network & Tree Ensemble for Precision Crop Recommendation", ha='center', va='center', fontsize=9.5, color=C_TEXT_MUTED)

    # --- 1. TOP: INPUT AGRICULTURAL DATA LAYER ---
    draw_box(25, 87, 50, 6.5, "INPUT AGRICULTURAL DATA LAYER", 
             subtitle="Raw Soil & Climate Factors (7 Continuous Features)", 
             details=["[ N (Nitrogen), P (Phosphorus), K (Potassium), Temperature, Humidity, pH, Rainfall ]"],
             border_color=C_INPUT)

    # --- 2. MIDDLE BRANCHES (LEFT: BRANCH A, RIGHT: BRANCH B) ---
    
    # Left Branch A Box Container
    draw_box(4, 52, 42, 31, "BRANCH A: XGBoost Leaf Embedding", 
             subtitle="Decision Tree Feature Representation", border_color=C_BRANCH_A)
    
    draw_box(6, 73, 38, 5.5, "XGBoost Booster Ensemble", 
             subtitle="200 Trees per Class * 22 Classes = 4,400 Trees", details=["[ Trained & Frozen Estimator ]"], border_color=C_BRANCH_A)
    
    draw_box(6, 64, 38, 5.5, "Leaf Index Extraction", 
             subtitle="dmatrix -> booster.predict(pred_leaf=True)", details=["Output Shape: (Batch, 4,400 Leaves)"], border_color=C_BRANCH_A)
    
    draw_box(6, 54, 38, 6.5, "EmbeddingBag Layer", 
             subtitle="118,800 Unique Leaf Embeddings -> 128-dim", details=["(Lightweight Variant: 32-dim) | Mode: 'mean' + LayerNorm"], border_color=C_BRANCH_A)

    # Internal Arrows Branch A
    draw_arrow((25, 87), (25, 78.5), color=C_INPUT)
    draw_arrow((25, 73), (25, 69.5), color=C_BRANCH_A)
    draw_arrow((25, 64), (25, 60.5), color=C_BRANCH_A)

    # Right Branch B Box Container
    draw_box(54, 52, 42, 31, "BRANCH B: FT-Transformer Backbone", 
             subtitle="Contextual Feature Representation", border_color=C_BRANCH_B)
    
    draw_box(56, 73, 38, 5.5, "Feature Tokenizer", 
             subtitle="Continuous Feature Projection (7 * 64-dim)", details=["Numerical Feature Embedding Layers"], border_color=C_BRANCH_B)
    
    draw_box(56, 64, 38, 5.5, "CLS Token Addition & Positional Encoding", 
             subtitle="Prepend Learnable [CLS] Token (Shape: 8 * 64-dim)", border_color=C_BRANCH_B)
    
    draw_box(56, 54, 38, 6.5, "Transformer Encoder Stack", 
             subtitle="4 Transformer Layers | 8 Multi-Head Attention", details=["LayerNorm + Multi-Head Self-Attention + FFN -> Extract f_ft (64-dim)"], border_color=C_BRANCH_B)

    # Internal Arrows Branch B
    draw_arrow((25, 87), (75, 78.5), color=C_INPUT, rad=-0.1)
    draw_arrow((75, 73), (75, 69.5), color=C_BRANCH_B)
    draw_arrow((75, 64), (75, 60.5), color=C_BRANCH_B)

    # --- 3. SIDE BOX 1: AGRICULTURAL KNOWLEDGE BASE ---
    draw_box(2, 32, 22, 14, "AGRICULTURAL KNOWLEDGE BASE", 
             subtitle="FAO / ICAR Domain Guidelines", details=["Expert Crop Suitability Scores", "Ranges for N, P, K, Temp, Hum, pH, Rain", "Normalized [0, 1] Agronomy Matrix"], 
             border_color=C_KNOWLEDGE, linestyle='--')

    # --- 4. FUSION MODULE ---
    draw_box(28, 28, 44, 21, "KNOWLEDGE-GUIDED GATED FUSION MODULE", 
             subtitle="Core Integration & Feature Fusion Layer", border_color=C_FUSION)

    draw_box(30, 40, 40, 5.5, "Adaptive Gated Network", 
             subtitle="Linear(192->128) -> ReLU -> Dropout -> Linear(128->192) -> Sigmoid", details=["Dynamic Routing Gate: g in [0, 1]^192"], border_color=C_FUSION)

    draw_box(30, 34, 19, 4.5, "Anchored Knowledge Matrix", 
             subtitle="K: (22 Crops * 7 Features)", details=["K0 + λ||K-K0||² during training"], border_color=C_KNOWLEDGE)

    draw_box(51, 34, 19, 4.5, "Input-Conditioned Attention", 
             subtitle="q(concat) × encode(K) → softmax", details=["context → k_bias (64 → 192)"], border_color=C_FUSION)

    draw_box(30, 29.5, 40, 3.8, "Fusion Formula & LayerNorm", 
             subtitle="f_fused = LayerNorm(g ⊙ concat + 0.1 · k_bias)", details=["Forward-active gate; output is a 192-dim feature vector"], border_color=C_FUSION)

    # Arrows into Fusion Module
    draw_arrow((25, 52), (38, 49), color=C_BRANCH_A, label="f_xgb (128-dim)")
    draw_arrow((75, 52), (62, 49), color=C_BRANCH_B, label="f_ft (64-dim)")
    draw_arrow((24, 39), (30, 36.2), color=C_KNOWLEDGE, label="Agronomic Prior", rad=0.0)

    # --- 5. SIDE BOX 2: AUXILIARY LOSSES ---
    draw_box(76, 32, 22, 14, "AUXILIARY LOSS FUNCTIONS", 
             subtitle="Joint Fine-Tuning Guidance", details=["1. Cross-Entropy Loss (L_CE)", "2. KL Distillation Loss (L_distill)", "3. Supervised Contrastive Loss (L_contrast)", "4. Knowledge Anchor (λ_K ||K-K0||²)"], 
             border_color=C_AUX, linestyle='--')

    draw_arrow((76, 39), (72, 34), color=C_AUX, label="Gradient Guidance", rad=0.0)

    # --- 6. PREDICTION HEAD & OUTPUT ---
    draw_box(28, 14, 44, 10, "PREDICTION HEAD (MLP)", 
             subtitle="Linear(192 -> 128) -> ReLU -> Dropout(0.1) -> Linear(128 -> 22)", 
             details=["Softmax Layer -> 22 Crop Probability Distribution"], border_color=C_HEAD)

    draw_box(28, 2, 44, 9, "FINAL MODEL OUTPUT", 
             subtitle="Recommended Crop + Top-3 Crop Distribution + Confidence Score", 
             details=["Output: Crop Class (e.g. Rice, Coffee, Chickpea) | Confidence (0 - 1)"], border_color=C_HEAD)

    # Arrow from Fusion to Prediction Head
    draw_arrow((50, 28), (50, 24), color=C_FUSION)
    draw_arrow((50, 14), (50, 11), color=C_HEAD)

    # --- 7. SIDE BOX 3: PRODUCTION PIPELINE ENHANCEMENTS ---
    draw_box(74, 2, 24, 23, "PRODUCTION PIPELINE ENHANCEMENTS", 
             subtitle="Real-World Deployment Suite", details=[
                 "1. KNN Imputer (k=5):", "   Recovers up to 60% missing inputs",
                 "2. Stage 1 Early Exit (tau=0.90):", "   Coverage: outputs/tables/early_exit_route_audit.csv",
                 "3. MC Dropout Uncertainty:", "   10 passes | std > 0.15 safety alert",
                 "4. Lightweight Model Option:", "   32-dim embed | 4.04M params",
                 "5. Structured JSON API Output:", "   Crop, confidence, latency, health"
             ], border_color=C_PROD, linestyle='-')

    draw_arrow((74, 13), (72, 6.5), color=C_PROD, label="Runtime Wrapper", rad=0.0)

    # Save outputs
    out1 = FIGURES_DIR / "fig1_kg_hybrid_detailed_architecture.png"
    out2 = FIGURES_DIR / "fig1_architecture.png"
    
    save_figure(plt.gcf(), out1, dpi=300, bbox_inches='tight', facecolor='#080B10')
    save_figure(plt.gcf(), out2, dpi=300, bbox_inches='tight', facecolor='#080B10')
    plt.close()

    print(f"[OK] Detailed System Architecture saved -> {out1}")
    print(f"[OK] Updated main architecture figure -> {out2}")

if __name__ == "__main__":
    create_diagram()
