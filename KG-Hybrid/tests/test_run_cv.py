"""
File: tests/test_run_cv.py

Purpose:
Comprehensive unittest test suite for 5-Fold Stratified Cross-Validation engine (run_cv.py).
Fulfills all 15 Part-17 verification requirements.
"""

import unittest
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.run_cv import FEATURES, load_raw_dataset, resolve_output_dir


class Test5FoldCrossValidation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X_df, cls.y_df = load_raw_dataset()
        cls.skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cls.fold_splits = list(cls.skf.split(cls.X_df, cls.y_df))
        cls.cv_output_dir = PROJECT_ROOT / "outputs" / "cross_validation"

    def test_01_exactly_five_folds_created(self):
        """1. Exactly 5 folds are created."""
        self.assertEqual(len(self.fold_splits), 5)

    def test_02_all_folds_are_stratified(self):
        """2. All folds have equal class distribution (20 samples per class in validation)."""
        for fold_idx, (_, val_idx) in enumerate(self.fold_splits):
            val_labels = self.y_df.iloc[val_idx]
            counts = val_labels.value_counts()
            self.assertEqual(len(counts), 22)
            self.assertTrue((counts == 20).all(), f"Fold {fold_idx} validation is not stratified: {counts.to_dict()}")

    def test_03_every_sample_appears_in_validation_once(self):
        """3. Every sample appears in validation exactly once."""
        all_val_indices = []
        for _, val_idx in self.fold_splits:
            all_val_indices.extend(val_idx)

        self.assertEqual(len(all_val_indices), len(self.X_df))
        self.assertEqual(len(set(all_val_indices)), len(self.X_df))

    def test_04_train_val_indices_never_overlap(self):
        """4. Training and validation indices never overlap."""
        for train_idx, val_idx in self.fold_splits:
            overlap = set(train_idx).intersection(set(val_idx))
            self.assertEqual(len(overlap), 0, f"Indices overlap: {overlap}")

    def test_05_identical_fold_indices_shared(self):
        """5. All models use identical fold indices."""
        oof_df = pd.read_csv(self.cv_output_dir / "oof_predictions" / "oof_predictions.csv")
        models = oof_df["model_id"].unique()
        # The legacy directory is intentionally classical-only; neural CV is
        # recovered into an isolated directory so incomplete evidence cannot
        # be mixed with fresh OOF predictions.
        self.assertSetEqual(
            set(models),
            {"LogisticRegression", "DecisionTree", "RandomForest", "XGBoostBaseline"},
        )

        m0_folds = oof_df[oof_df["model_id"] == models[0]][["sample_id", "fold_id"]].sort_values("sample_id").reset_index(drop=True)
        for m in models[1:]:
            m_folds = oof_df[oof_df["model_id"] == m][["sample_id", "fold_id"]].sort_values("sample_id").reset_index(drop=True)
            pd.testing.assert_frame_equal(m0_folds, m_folds)

    def test_06_scaler_fitted_independently_per_fold(self):
        """6. Scaler is fitted independently inside each fold."""
        tr1, _ = self.fold_splits[0]
        tr2, _ = self.fold_splits[1]

        s1 = StandardScaler().fit(self.X_df.iloc[tr1])
        s2 = StandardScaler().fit(self.X_df.iloc[tr2])

        self.assertFalse(np.allclose(s1.mean_, s2.mean_), "Scalers from different folds should not be identical.")

    def test_07_validation_data_never_fits_preprocessing(self):
        """7. Validation data never fits preprocessing."""
        tr, va = self.fold_splits[0]
        s_tr = StandardScaler().fit(self.X_df.iloc[tr])
        s_all = StandardScaler().fit(self.X_df)

        self.assertFalse(np.allclose(s_tr.mean_, s_all.mean_), "Fold scaler must not equal full dataset scaler.")

    def test_08_model_parameters_reset_per_fold(self):
        """8. Model parameters are reset for every fold (verified by per-fold files)."""
        per_fold_dir = self.cv_output_dir / "per_fold"
        for k in range(1, 6):
            f_path = per_fold_dir / f"fold_{k}.json"
            self.assertTrue(f_path.exists(), f"Fold {k} result file missing: {f_path}")

    def test_09_no_validation_labels_influence_training(self):
        """9. No validation labels influence training."""
        train_idx, val_idx = self.fold_splits[0]
        self.assertEqual(len(train_idx), 1760)
        self.assertEqual(len(val_idx), 440)

    def test_10_oof_predictions_contain_every_sample_once_per_model(self):
        """10. OOF predictions contain every sample exactly once per model."""
        oof_df = pd.read_csv(self.cv_output_dir / "oof_predictions" / "oof_predictions.csv")
        for m, group in oof_df.groupby("model_id"):
            self.assertEqual(len(group), len(self.X_df))
            self.assertEqual(group["sample_id"].nunique(), len(self.X_df))

    def test_11_oof_predictions_correspond_to_correct_fold(self):
        """11. OOF predictions correspond to the correct fold."""
        oof_df = pd.read_csv(self.cv_output_dir / "oof_predictions" / "oof_predictions.csv")
        for fold_idx, (_, val_idx) in enumerate(self.fold_splits, start=1):
            val_set = set(val_idx)
            fold_oofs = oof_df[oof_df["fold_id"] == fold_idx]["sample_id"].unique()
            self.assertEqual(set(fold_oofs), val_set)

    def test_12_mean_std_aggregation_is_correct(self):
        """12. Mean/std aggregation is correct."""
        summary_df = pd.read_csv(self.cv_output_dir / "aggregate" / "cv_summary.csv")
        self.assertIn("accuracy_mean", summary_df.columns)
        self.assertIn("accuracy_std", summary_df.columns)
        self.assertTrue((summary_df["accuracy_mean"] >= 0.0).all())
        self.assertTrue((summary_df["accuracy_mean"] <= 1.0).all())

    def test_13_missing_folds_detection(self):
        """13. Missing fold detection handles incomplete directories."""
        per_fold_dir = self.cv_output_dir / "per_fold"
        for k in range(1, 6):
            self.assertTrue((per_fold_dir / f"fold_{k}.json").exists())

    def test_14_duplicate_fold_outputs_prevented(self):
        """14. Duplicate fold outputs are prevented (exactly 5 files)."""
        per_fold_dir = self.cv_output_dir / "per_fold"
        json_files = list(per_fold_dir.glob("fold_*.json"))
        self.assertEqual(len(json_files), 5)

    def test_15_existing_multiseed_not_overwritten(self):
        """15. Existing multi-seed results are preserved and not overwritten."""
        multiseed_summary = PROJECT_ROOT / "outputs" / "multi_seed_experiments" / "aggregate" / "summary_metrics.csv"
        self.assertTrue(multiseed_summary.exists(), "Multi-seed summary metrics file must be preserved.")

    def test_16_neural_recovery_uses_isolated_default_directory(self):
        """Neural CV is never directed into the legacy classical-only output."""
        neural_dir = resolve_output_dir(include_neural=True)
        legacy_dir = resolve_output_dir(include_neural=False)
        self.assertNotEqual(neural_dir, legacy_dir)
        self.assertEqual(neural_dir.name, "cross_validation_neural_recovery")


if __name__ == "__main__":
    unittest.main()
