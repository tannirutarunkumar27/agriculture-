"""Regression checks for the reviewer-identified functional and evidence failures."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from unittest.mock import Mock

from src.models.hybrid.knowledge_fusion import KnowledgeGuidedFusion
from src.experiments.run_knowledge_comparison import compact_leaf_vocabulary, encode_leaves
from src.experiments.publication_evidence import load_evidence, copy_artifact
from src.experiments.cv_evidence import validate_oof_frame
from src.experiments.validate_knowledge_comparison import holm_adjust
from src.inference.production_pipeline import ProductionCropRecommender
from src.robustness.robustness_utils import FEATURES

ROOT = Path(__file__).resolve().parents[1]


class TestReviewerFixes(unittest.TestCase):
    def test_artifact_copy_replaces_existing_complete_file(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            destination = Path(directory) / "destination.png"
            source.write_bytes(b"new complete artifact")
            destination.write_bytes(b"old artifact")
            copy_artifact(source, destination)
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertEqual(len(list(Path(directory).iterdir())), 2)

    def test_holm_adjustment_accounts_for_comparison_family(self):
        adjusted = holm_adjust([.01, .04, .03, np.nan])
        np.testing.assert_allclose(adjusted[:3], [.03, .06, .06])
        self.assertTrue(np.isnan(adjusted[3]))

    def test_sensor_inputs_are_scaled_once_before_inference(self):
        recommender = ProductionCropRecommender.__new__(ProductionCropRecommender)
        recommender.scaler = StandardScaler().fit(np.vstack((np.arange(7), np.arange(7) + 10)))
        recommender.xgb_clf = Mock()
        recommender.xgb_clf.predict_proba.return_value = np.eye(22)[[0]]
        recommender.early_exit_threshold = .9
        recommender.model_version = "test"
        recommender.region_context = "test"
        raw = np.arange(7, dtype=float) + 3
        expected = recommender.scaler.transform(raw.reshape(1, -1))
        recommender.predict_single(dict(zip(FEATURES, raw)))
        actual = recommender.xgb_clf.predict_proba.call_args.args[0].to_numpy()
        np.testing.assert_allclose(actual, expected)
        recommender.predict_single(dict(zip(FEATURES, expected[0])), input_is_scaled=True)
        actual = recommender.xgb_clf.predict_proba.call_args.args[0].to_numpy()
        np.testing.assert_allclose(actual, expected)

    def test_invalid_sensor_value_is_rejected(self):
        recommender = ProductionCropRecommender.__new__(ProductionCropRecommender)
        for value in ("invalid", float("inf")):
            with self.assertRaises(ValueError):
                recommender.predict_single({"N": value})

    def test_fixed_prior_does_not_update(self):
        model = KnowledgeGuidedFusion(knowledge_matrix=np.full((22, 7), .5), knowledge_policy="fixed")
        before = model.knowledge_matrix.clone()
        optimizer = torch.optim.Adam(model.parameters())
        model(torch.randn(3, 128), torch.randn(3, 64))[0].square().sum().backward()
        optimizer.step()
        self.assertTrue(torch.equal(before, model.knowledge_matrix))
        self.assertEqual(model.knowledge_anchor_loss().item(), 0)

    def test_anchor_pulls_matrix_toward_prior(self):
        model = KnowledgeGuidedFusion(knowledge_matrix=np.full((22, 7), .5))
        with torch.no_grad():
            model.knowledge_matrix.add_(.2)
        before = model.knowledge_anchor_loss().item()
        optimizer = torch.optim.SGD([model.knowledge_matrix], lr=1)
        model.knowledge_anchor_loss().backward()
        optimizer.step()
        self.assertLess(model.knowledge_anchor_loss().item(), before)
        self.assertGreater(model.knowledge_drift_metrics()["mean_abs_change"], 0)

    def test_legacy_checkpoint_does_not_relabel_learned_matrix_as_prior(self):
        prior = np.full((22, 7), .5)
        original = KnowledgeGuidedFusion(knowledge_matrix=prior)
        state = original.state_dict()
        state.pop("knowledge_prior")
        state["knowledge_matrix"] = torch.full((22, 7), .7)
        loaded = KnowledgeGuidedFusion(knowledge_matrix=prior)
        loaded.load_state_dict(state, strict=True)
        self.assertAlmostEqual(loaded.knowledge_drift_metrics()["mean_abs_change"], .2, places=6)
        self.assertAlmostEqual(loaded.knowledge_prior.mean().item(), .5, places=6)

    def test_invalid_prior_is_rejected(self):
        for value in (np.nan, np.inf, -1, 2):
            with self.assertRaises(ValueError):
                KnowledgeGuidedFusion(knowledge_matrix=np.full((22, 7), value))

    def test_compact_mapping_covers_every_tree_leaf(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(80, 3)).astype(np.float32)
        labels = (X[:, 0] > 0).astype(int)
        model = xgb.XGBClassifier(n_estimators=3, max_depth=2, n_jobs=1).fit(X, labels)
        lookup, size = compact_leaf_vocabulary(model.get_booster())
        mapped = encode_leaves(model.apply(X), lookup)
        self.assertTrue((mapped >= 0).all())
        self.assertTrue((mapped < size).all())
        self.assertEqual(len(np.unique(lookup[lookup >= 0])), size)
        self.assertEqual(sorted(lookup[lookup >= 0]), list(range(size)))
        with self.assertRaises(ValueError):
            encode_leaves(np.full_like(model.apply(X), lookup.shape[1] + 1), lookup)

    def test_reliability_bins_and_ece_use_same_saved_predictions(self):
        labels, results = load_evidence()
        for result in results:
            bins = result["bin_data"]
            self.assertEqual(sum(b["sample_count"] for b in bins), len(labels))
            recomputed = sum(b["sample_count"] * abs(b["accuracy"] - b["confidence"]) for b in bins) / len(labels)
            self.assertAlmostEqual(result["ece"], recomputed, places=5)

    def test_oof_labels_and_fold_assignment_cannot_be_faked(self):
        raw = pd.read_csv(ROOT / "data/raw/Crop_recommendation.csv")
        frame = pd.read_csv(ROOT / "outputs/cross_validation_neural_recovery/oof_predictions/oof_predictions.csv")
        models = list(frame.model_id.unique())
        self.assertTrue(validate_oof_frame(frame, raw, models)["true_labels_match_raw_dataset"])
        wrong_label = frame.copy()
        wrong_label.loc[0, "true_label"] = (wrong_label.loc[0, "true_label"] + 1) % 22
        with self.assertRaisesRegex(ValueError, "labels"):
            validate_oof_frame(wrong_label, raw, models)
        wrong_fold = frame.copy()
        wrong_fold.loc[0, "fold_id"] = wrong_fold.loc[0, "fold_id"] % 5 + 1
        with self.assertRaisesRegex(ValueError, "fold"):
            validate_oof_frame(wrong_fold, raw, models)


if __name__ == "__main__":
    unittest.main()
