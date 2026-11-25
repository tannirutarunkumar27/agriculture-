"""
File: final_report.py

Purpose:
Generate a comprehensive plain-text summary report of the
entire KG-Hybrid Crop Recommendation experiment.

Saves to outputs/reports/final_report.txt
"""

from pathlib import Path
from datetime import datetime

# =====================================================
# Project Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR   = PROJECT_ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH  = REPORT_DIR / "final_report.txt"


# =====================================================
# Report Data
# =====================================================

DATASET_INFO = {
    "Name":            "Crop Recommendation Dataset",
    "Source":          "data/raw/Crop_recommendation.csv",
    "Total Samples":   2200,
    "Features":        7,
    "Classes":         22,
    "Missing Values":  0,
    "Duplicates":      0,
    "Balance":         "Perfect — 100 samples per class",
}

FEATURE_INFO = [
    ("N",           "Nitrogen content in soil (integer)"),
    ("P",           "Phosphorous content in soil (integer)"),
    ("K",           "Potassium content in soil (integer)"),
    ("temperature", "Ambient temperature in Celsius (float)"),
    ("humidity",    "Relative humidity in % (float)"),
    ("ph",          "Soil pH value (float)"),
    ("rainfall",    "Annual rainfall in mm (float)"),
    ("label",       "Recommended crop name (target)"),
]

CROP_NAMES = [
    "apple", "banana", "blackgram", "chickpea", "coconut",
    "coffee", "cotton", "grapes", "jute", "kidneybeans",
    "lentil", "maize", "mango", "mothbeans", "mungbean",
    "muskmelon", "orange", "papaya", "pigeonpeas",
    "pomegranate", "rice", "watermelon",
]

SPLIT_INFO = {
    "Strategy":   "Stratified split (preserves class proportions)",
    "Train":      "70%  →  1540 samples",
    "Validation": "15%  →  330 samples",
    "Test":       "15%  →  330 samples",
    "Seed":       42,
}

PREPROCESSING = [
    "1. LabelEncoder   — fit on y_train only → integers 0–21",
    "2. StandardScaler — fit on X_train only → mean≈0, std≈1",
    "3. No data leakage — val/test scaled using train statistics",
    "4. Artifacts saved: scaler.pkl, label_encoder.pkl",
]

MODEL_RESULTS = [
    ("Logistic Regression", 0.9788, 0.9795, 0.9788, 0.9787),
    ("Decision Tree",       0.9818, 0.9829, 0.9818, 0.9818),
    ("FT-Transformer",      0.9788, 0.9814, 0.9788, 0.9787),
    ("XGBoost",             0.9939, 0.9943, 0.9939, 0.9939),
    ("Random Forest",       0.9970, 0.9972, 0.9970, 0.9970),
]

FT_CONFIG = {
    "Architecture":        "Feature Tokenizer → CLS Token → 4× Transformer Encoder → MLP",
    "NUM_FEATURES":        7,
    "NUM_CLASSES":         22,
    "EMBEDDING_DIM":       64,
    "NUM_HEADS":           8,
    "NUM_LAYERS":          4,
    "FF_DIM":              256,
    "DROPOUT":             0.1,
    "EPOCHS_PLANNED":      100,
    "EPOCHS_RAN":          27,
    "EARLY_STOP_PATIENCE": 10,
    "BATCH_SIZE":          32,
    "OPTIMIZER":           "AdamW (lr=1e-3, wd=1e-5)",
    "SCHEDULER":           "ReduceLROnPlateau (factor=0.5, patience=5)",
    "BEST_VAL_ACCURACY":   "99.39% (epoch 17)",
    "FINAL_TEST_ACCURACY": "97.88%",
}

OUTPUT_FILES = {
    "outputs/figures/": [
        "class_distribution.png",
        "correlation_heatmap.png",
        "pairplot.png",
        "N/P/K/temperature/humidity/ph/rainfall — _histogram.png (7 files)",
        "N/P/K/temperature/humidity/ph/rainfall — _boxplot.png (7 files)",
        "decision_tree_confusion_matrix.png",
        "logistic_regression_confusion_matrix.png",
        "random_forest_confusion_matrix.png",
        "xgboost_confusion_matrix.png",
        "ft_transformer_confusion_matrix.png",
    ],
    "outputs/tables/": [
        "model_comparison.csv",
        "model_comparison_clean.csv",
        "decision_tree_metrics.json + _report.txt + _report.csv",
        "logistic_regression_metrics.json + _report.txt + _report.csv",
        "random_forest_metrics.json + _report.txt + _report.csv",
        "xgboost_metrics.json + _report.txt + _report.csv",
        "ft_transformer_metrics.json + _report.txt + _report.csv",
    ],
    "outputs/models/": [
        "best_model.pth  (FT-Transformer checkpoint, 2.6 MB)",
    ],
    "outputs/plots/": [
        "ft_transformer_loss_curve.png",
        "ft_transformer_accuracy_curve.png",
        "ft_transformer_f1_curve.png",
        "ft_transformer_training_history.png",
        "model_comparison_bar_chart.png",
        "model_accuracy_comparison.png",
        "model_comparison_radar.png",
    ],
    "data/processed/": [
        "train.csv (1540 rows)",
        "validation.csv (330 rows)",
        "test.csv (330 rows)",
        "scaler.pkl",
        "label_encoder.pkl",
    ],
}


# =====================================================
# Report Generator
# =====================================================

SEP  = "=" * 70
SEP2 = "-" * 70


def build_report() -> str:
    lines = []

    def h1(title):
        lines.append(f"\n{SEP}")
        lines.append(f"  {title}")
        lines.append(SEP)

    def h2(title):
        lines.append(f"\n{SEP2}")
        lines.append(f"  {title}")
        lines.append(SEP2)

    def kv(key, val, indent=4):
        lines.append(f"{' ' * indent}{key:<28}: {val}")

    def li(items, indent=4):
        for item in items:
            lines.append(f"{' ' * indent}• {item}")

    # ─── Header ───────────────────────────────────────────────────────
    lines.append(SEP)
    lines.append("  KG-HYBRID CROP RECOMMENDATION — FINAL REPORT")
    lines.append(f"  Generated : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    lines.append(SEP)

    # ─── 1. Dataset ───────────────────────────────────────────────────
    h1("1. DATASET INFORMATION")
    for k, v in DATASET_INFO.items():
        kv(k, v)

    h2("1.1 Features")
    lines.append(f"    {'Column':<16} Description")
    lines.append(f"    {'-'*16} {'-'*40}")
    for col, desc in FEATURE_INFO:
        lines.append(f"    {col:<16} {desc}")

    h2("1.2 Crop Classes (22 total)")
    for i, crop in enumerate(CROP_NAMES, 1):
        lines.append(f"    {i:>2}. {crop}")

    # ─── 2. Data Split ────────────────────────────────────────────────
    h1("2. DATA SPLITTING")
    for k, v in SPLIT_INFO.items():
        kv(k, v)

    # ─── 3. Preprocessing ─────────────────────────────────────────────
    h1("3. PREPROCESSING PIPELINE")
    li(PREPROCESSING)

    # ─── 4. Models ────────────────────────────────────────────────────
    h1("4. MODEL RESULTS (Test Set — 330 samples)")

    lines.append(f"\n    {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    lines.append(f"    {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for name, acc, prec, rec, f1 in MODEL_RESULTS:
        lines.append(
            f"    {name:<25} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}"
        )

    best = max(MODEL_RESULTS, key=lambda x: x[1])
    lines.append(f"\n    Best Model : {best[0]}  (Accuracy = {best[1]:.4f})")

    # ─── 5. FT-Transformer ────────────────────────────────────────────
    h1("5. FT-TRANSFORMER CONFIGURATION")
    for k, v in FT_CONFIG.items():
        kv(k, v)

    h2("5.1 Architecture Flow")
    lines.append("""
    Input (batch × 7 features)
            ↓
    Feature Tokenizer  →  (batch × 7 × 64)
            ↓
    [CLS] Token        →  (batch × 8 × 64)
            ↓
    Transformer ×4         [8-head self-attention]
            ↓
    CLS Embedding      →  (batch × 64)
            ↓
    Linear(64→128) → GELU → Dropout(0.1)
            ↓
    Linear(128→22)
            ↓
    Crop Prediction (22 classes)
    """)

    h2("5.2 Training Summary")
    lines.append("    Epoch   Train Loss   Train Acc   Val Loss    Val Acc")
    lines.append("    " + "-" * 55)
    training_log = [
        (1,  2.0078, 0.4708, 0.9634, 0.8242),
        (2,  0.6172, 0.8539, 0.2818, 0.9121),
        (3,  0.2309, 0.9370, 0.1257, 0.9576),
        (5,  0.0979, 0.9695, 0.0526, 0.9848),
        (10, 0.0568, 0.9818, 0.0308, 0.9909),
        (17, "—",    "—",    "—",    0.9939),
        (27, 0.0160, 0.9948, 0.0190, 0.9939),
    ]
    for row in training_log:
        epoch = row[0]
        if row[1] == "—":
            lines.append(f"    {epoch:<7} {'(best ckpt saved)':>50}")
        else:
            lines.append(
                f"    {epoch:<7} {row[1]:<13.4f} {row[2]:<12.4f}"
                f"{row[3]:<12.4f} {row[4]:.4f}"
            )
    lines.append("    Early stopping triggered at epoch 27 (patience=10)")

    # ─── 6. Output Files ──────────────────────────────────────────────
    h1("6. OUTPUT FILES GENERATED")
    for folder, files in OUTPUT_FILES.items():
        lines.append(f"\n    {folder}")
        li(files, indent=6)

    # ─── 7. Conclusion ────────────────────────────────────────────────
    h1("7. CONCLUSION")
    lines.append("""
    All five models were trained and evaluated on the same 330-sample
    test set (15% stratified split, seed=42).

    Key findings:
      • The dataset is perfectly balanced (100 samples/class) which
        makes accuracy a reliable metric.
      • All models achieved > 97.8% accuracy, confirming the dataset
        is highly separable with the 7 soil/climate features.
      • Random Forest achieved the highest accuracy (99.70%) followed
        by XGBoost (99.39%).
      • FT-Transformer (98.79% checkpoint val, 97.88% test) shows
        deep learning is competitive but not superior on small tabular
        data — consistent with published FT-Transformer findings.
      • Logistic Regression achieved 97.88%, showing strong linear
        separability in the scaled feature space.
    """)

    lines.append(SEP)
    lines.append("  END OF REPORT")
    lines.append(SEP)

    return "\n".join(lines)


# =====================================================
# Main
# =====================================================

def main():

    print("=" * 60)
    print("GENERATING FINAL REPORT")
    print("=" * 60)

    report = build_report()

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nReport saved: {REPORT_PATH}")
    print(f"Size        : {REPORT_PATH.stat().st_size} bytes")

    # Print to console too
    print("\n" + report)


if __name__ == "__main__":
    main()
