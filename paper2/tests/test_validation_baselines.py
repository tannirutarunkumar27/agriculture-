import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from paper2.centralized import run_reference, serializable_parameters
from paper2.data import file_hash, load_dataset, make_partition
from paper2.federated import train_client, make_model
from paper2.metrics import probability_metrics
from paper2.train import evaluate, run


class TestValidationBaselines(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        frame,schema,source = load_dataset("canonical")
        partition = make_partition(frame,schema,file_hash(source),"canonical",clients=3)
        self.partition = self.root/"partition.json"
        self.partition.write_text(json.dumps(partition),encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_neural_validation_only_never_evaluates_test(self):
        output = self.root/"pilot"
        with patch("paper2.train.evaluate", wraps=evaluate) as evaluation:
            result = run(self.partition,output,"fedavg",{"rounds":2,"hidden":8,"batch_size":128},evaluate_test=False)
            self.assertEqual(evaluation.call_count,3)  # two rounds + selected validation, no test call
        self.assertIsNone(result["test"])
        self.assertFalse((output/"test_predictions.npz").exists())
        self.assertFalse(json.loads((output/"manifest.json").read_text())["test_evaluation_enabled"])

    def test_conventional_validation_only(self):
        candidates = {"logistic_regression":{"max_iter":1000},
                      "random_forest":{"n_estimators":5},
                      "xgboost":{"n_estimators":5,"max_depth":2}}
        for kind,parameters in candidates.items():
            with self.subTest(kind=kind):
                output = self.root/kind
                result = run_reference(self.partition,output,kind,parameters,evaluate_test=False)
                self.assertEqual(result["validation"]["samples"],330)
                self.assertIsNone(result["test"])
                self.assertFalse((output/"test_predictions.npz").exists())
                self.assertTrue((output/"model.joblib").exists())

    def test_probability_validation(self):
        for values in [np.array([[0.2,0.2]]),np.array([[np.nan,0.5]]),np.array([[-1,2]])]:
            with self.assertRaises(ValueError):
                probability_metrics(np.array([0]),values,["a","b"])

    def test_probability_metrics_perfect(self):
        result = probability_metrics(np.array([0,1]),np.array([[1.,0.],[0.,1.]]),["a","b"])
        self.assertEqual(result["macro_f1"],1.0)
        self.assertEqual(result["confusion_matrix"],[[1,0],[0,1]])

    def test_nan_library_default_manifest_is_valid_json(self):
        parameters = serializable_parameters({"missing":float("nan"), "max_depth":3})
        json.dumps(parameters,allow_nan=False)
        self.assertEqual(parameters["missing"],{"nonfinite_library_default":"nan"})

    def test_float32_probability_rounding_is_not_a_warning(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = probability_metrics(np.array([0]),np.array([[0.1,0.2,0.7]],dtype=np.float32),["a","b","c"])
        self.assertTrue(np.isfinite(result["nll"]))

    def test_invalid_label_index_rejected(self):
        model = make_model(2,2)
        with self.assertRaises(ValueError):
            train_client(model,torch.ones(2,2),torch.tensor([0,2]),"a",1,"schema",42)

    def test_invalid_label_dtype_rejected(self):
        model = make_model(2,2)
        with self.assertRaises(ValueError):
            train_client(model,torch.ones(2,2),torch.zeros(2),"a",1,"schema",42)
