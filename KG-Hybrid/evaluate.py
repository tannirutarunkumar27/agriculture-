"""
Evaluation entry point for KG-Hybrid FT-Transformer.

Loads the best saved checkpoint and evaluates it on the test set.
Saves confusion matrix, classification report, metrics JSON,
and appends to model_comparison.csv.

Run from the KG-Hybrid/ directory:
    python evaluate.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ft_transformer.trainer import Trainer


if __name__ == "__main__":

    print("=" * 60)
    print("FT-TRANSFORMER EVALUATION")
    print("=" * 60)

    trainer = Trainer()

    # Evaluate on test set and save all results
    metrics = trainer.test()

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1_score']:.4f}")
    print("\nResults saved to outputs/tables/ and outputs/figures/")
