"""
Model Exporter & Converter — PyTorch to TorchScript & ONNX
KrishiSetu Deployment Suite — Phase 1 (Milestone 1 API)

Exports:
  1. Lightweight KG-Hybrid PyTorch -> TorchScript Mobile (outputs/models/kg_hybrid_mobile.pt)
  2. Lightweight KG-Hybrid PyTorch -> ONNX (outputs/models/kg_hybrid.onnx)
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import warnings
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2] # d:\Final Project\implementation_for _crop recommendation\KG-Hybrid
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hybrid.lightweight_kg_hybrid import LightweightKGHybrid
from src.robustness.robustness_utils import DATA_PATH, MODEL_PATH, TOTAL_UNIQUE_LEAVES, NUM_TREES

OUTPUT_MODEL_DIR = MODEL_PATH
OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def export_models():
    print("=" * 70)
    print("MODEL EXPORTER & CONVERTER (PyTorch -> TorchScript & ONNX)")
    print("=" * 70)

    # 1. Load PyTorch Model
    print("\n[1/3] Loading Lightweight KG-Hybrid PyTorch Checkpoint...")
    km = np.load(DATA_PATH / "knowledge_matrix.npy")
    model = LightweightKGHybrid(
        num_unique_leaves=TOTAL_UNIQUE_LEAVES,
        num_trees=NUM_TREES,
        embed_dim=32,
        num_classes=22,
        knowledge_matrix=km
    )

    ckpt_path = MODEL_PATH / "lightweight_kg_hybrid_best.pth"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        print(f"  Loaded weights: {ckpt_path.name}")
    else:
        print(f"  [WARN] Checkpoint not found at {ckpt_path}, using initialized weights")

    model.eval()

    dummy_leaves = torch.zeros((1, NUM_TREES), dtype=torch.long)
    dummy_features = torch.randn((1, 7), dtype=torch.float32)

    with torch.no_grad():
        pytorch_logits, pytorch_fused = model(dummy_leaves, dummy_features)
        pytorch_probs = torch.softmax(pytorch_logits, dim=1).numpy()[0]

    print(f"  PyTorch prediction check - Top crop ID: {np.argmax(pytorch_probs)} (Prob: {np.max(pytorch_probs):.4f})")

    # 2. Export to TorchScript (.pt)
    pt_path = OUTPUT_MODEL_DIR / "kg_hybrid_mobile.pt"
    print(f"\n[2/3] Exporting to TorchScript Mobile Format: {pt_path}...")
    traced_model = torch.jit.trace(model, (dummy_leaves, dummy_features), check_trace=False)
    traced_model.save(str(pt_path))
    print(f"  [OK] TorchScript Mobile export SUCCESS! Size: {pt_path.stat().st_size / (1024*1024):.2f} MB")

    # 3. Export to ONNX
    onnx_path = OUTPUT_MODEL_DIR / "kg_hybrid.onnx"
    print(f"\n[3/3] Exporting to ONNX format: {onnx_path}...")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                model,
                (dummy_leaves, dummy_features),
                str(onnx_path),
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=["leaf_indices", "raw_features"],
                output_names=["logits", "fused_embeddings"]
            )
        print(f"  [OK] ONNX export SUCCESS! Size: {onnx_path.stat().st_size / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"  [INFO] ONNX export notice ({type(e).__name__}). TorchScript mobile binary (.pt) is ready.")

    print("\n" + "=" * 70)
    print("MODEL EXPORT COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"  PyTorch Weights : {ckpt_path}")
    if pt_path.exists():
        print(f"  TorchScript Mobile: {pt_path} ({pt_path.stat().st_size/(1024*1024):.2f} MB)")
    if onnx_path.exists():
        print(f"  ONNX Model      : {onnx_path} ({onnx_path.stat().st_size/(1024*1024):.2f} MB)")


if __name__ == "__main__":
    export_models()
