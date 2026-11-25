"""
File: config.py

Purpose:
Central configuration file for the FT-Transformer.

All model, training, checkpoint, logging,
and experiment hyperparameters are defined here.
"""

from pathlib import Path
import random
import numpy as np
import torch

# =====================================================
# Project Root
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# =====================================================
# Experiment Information
# =====================================================

EXPERIMENT_NAME = "FT_Transformer_Baseline"

RANDOM_SEED = 42

# =====================================================
# Device Configuration
# =====================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# =====================================================
# Dataset Configuration
# =====================================================

NUM_FEATURES = 7

NUM_CLASSES = 22

NUM_WORKERS = 0

PIN_MEMORY = False

# =====================================================
# Model Configuration
# =====================================================

EMBEDDING_DIM = 64

NUM_HEADS = 8

NUM_LAYERS = 4

FF_DIM = 256

DROPOUT = 0.1

class FTConfig:
    NUM_FEATURES = NUM_FEATURES
    NUM_CLASSES = NUM_CLASSES
    EMBEDDING_DIM = EMBEDDING_DIM
    NUM_HEADS = NUM_HEADS
    NUM_LAYERS = NUM_LAYERS
    FF_DIM = FF_DIM
    DROPOUT = DROPOUT

# =====================================================
# Training Configuration
# =====================================================

EPOCHS = 100

BATCH_SIZE = 32

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

LOSS_FUNCTION = "CrossEntropyLoss"

# =====================================================
# Optimizer Configuration
# =====================================================

OPTIMIZER = "AdamW"

BETAS = (0.9, 0.999)

EPS = 1e-8

# =====================================================
# Scheduler Configuration
# =====================================================

SCHEDULER = "ReduceLROnPlateau"

LR_FACTOR = 0.5

LR_PATIENCE = 5

MIN_LR = 1e-6

STEP_SIZE = 10

STEP_GAMMA = 0.1

COSINE_T_MAX = 50

# =====================================================
# Early Stopping Configuration
# =====================================================

EARLY_STOPPING_PATIENCE = 10

MIN_DELTA = 1e-4

# =====================================================
# Output Directories
# =====================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR = OUTPUT_DIR / "models"

REPORT_DIR = OUTPUT_DIR / "reports"

PLOT_DIR = OUTPUT_DIR / "plots"

# Create directories automatically
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Checkpoint Configuration
# =====================================================

CHECKPOINT_NAME = "best_model.pth"

CHECKPOINT_DIR = MODEL_DIR

MODEL_SAVE_PATH = MODEL_DIR / CHECKPOINT_NAME

SAVE_BEST_ONLY = True

RESUME_TRAINING = False

# =====================================================
# Preprocessing Files
# =====================================================

SCALER_PATH = PROJECT_ROOT / "data" / "processed" / "scaler.pkl"

LABEL_ENCODER_PATH = PROJECT_ROOT / "data" / "processed" / "label_encoder.pkl"

# =====================================================
# Logging Configuration
# =====================================================

PRINT_EVERY = 1

SAVE_LOGS = True

# =====================================================
# Reproducibility
# =====================================================

torch.manual_seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

random.seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False

# =====================================================
# Configuration Summary
# =====================================================

if __name__ == "__main__":

    print("=" * 60)
    print("FT-TRANSFORMER CONFIGURATION")
    print("=" * 60)

    print(f"Experiment        : {EXPERIMENT_NAME}")
    print(f"Device            : {DEVICE}")
    print(f"Features          : {NUM_FEATURES}")
    print(f"Classes           : {NUM_CLASSES}")

    print(f"\nEpochs            : {EPOCHS}")
    print(f"Batch Size        : {BATCH_SIZE}")
    print(f"Learning Rate     : {LEARNING_RATE}")

    print(f"\nModel Path        : {MODEL_SAVE_PATH}")
    print(f"Scaler Path       : {SCALER_PATH}")
    print(f"Label Encoder     : {LABEL_ENCODER_PATH}")

    print("=" * 60)