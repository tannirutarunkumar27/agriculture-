"""
Seed Controller & Environmental Reproducibility Manager (Stage 7 - Multi-Seed)
KrishiSetu Q1 ML Core — Knowledge-Guided Hybrid Crop Recommendation

Provides centralized deterministic seeding across:
  - Python standard random module
  - NumPy random generator
  - PyTorch CPU generator
  - PyTorch CUDA GPU generators & CuDNN deterministic backends
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import platform
import random
import numpy as np
import torch
import xgboost as xgb
import sklearn
from typing import Dict, List, Any


PRESET_RESEARCH_SEEDS: List[int] = [42, 123, 2024, 3407, 9999]


def set_all_seeds(seed: int = 42) -> None:
    """
    Sets global random seed for Python, NumPy, PyTorch CPU/GPU, and CuDNN.

    Args:
        seed: Integer random seed.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # PyTorch deterministic flags
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_preset_seeds() -> List[int]:
    """Returns the predetermined 5-seed list for Q1 statistical evaluation."""
    return list(PRESET_RESEARCH_SEEDS)


def get_execution_environment_metadata() -> Dict[str, Any]:
    """Collects system, hardware, and library versions for audit logs."""
    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "xgboost_version": xgb.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "preset_seeds": get_preset_seeds()
    }
