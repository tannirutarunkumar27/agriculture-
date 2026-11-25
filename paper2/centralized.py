"""Matched LR/RF/XGBoost centralized references; validation-only by default."""
import importlib.metadata
import json
import math
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from paper2.data import file_hash, load_dataset, prepare_arrays
from paper2.metrics import probability_metrics
from paper2.train import write_json


def serializable_parameters(parameters):
    """Record library defaults such as XGBoost's NaN missing-value sentinel.

    This does not allow NaN in prediction tensors, training loss or metrics.
    """
    result = {}
    for key,value in parameters.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value,float) and not math.isfinite(value):
            value = {"nonfinite_library_default":str(value)}
        result[key] = value
    return result


def make_estimator(kind, parameters, seed):
    if kind == "logistic_regression":
        if set(parameters) - {"C", "max_iter"}:
            raise ValueError("Unsupported LogisticRegression parameters")
        return LogisticRegression(C=parameters.get("C", 1.0), max_iter=parameters.get("max_iter",2000),
                                  solver="lbfgs", random_state=seed)
    if kind == "random_forest":
        if set(parameters) - {"n_estimators", "max_depth", "min_samples_leaf"}:
            raise ValueError("Unsupported RandomForest parameters")
        return RandomForestClassifier(n_estimators=parameters.get("n_estimators",200),
                                      max_depth=parameters.get("max_depth"),
                                      min_samples_leaf=parameters.get("min_samples_leaf",1),
                                      n_jobs=1, random_state=seed)
    if kind == "xgboost":
        if set(parameters) - {"n_estimators", "max_depth", "learning_rate"}:
            raise ValueError("Unsupported XGBoost parameters")
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=parameters.get("n_estimators",200),
                             max_depth=parameters.get("max_depth",4), learning_rate=parameters.get("learning_rate",0.1),
                             n_jobs=1, random_state=seed, tree_method="hist", objective="multi:softprob",
                             eval_metric="mlogloss", subsample=1.0, colsample_bytree=1.0)
    raise ValueError("Unknown centralized model")


def run_reference(partition_path, output, kind, parameters=None, seed=42, evaluate_test=False):
    if type(seed) is not int or seed < 0 or type(evaluate_test) is not bool:
        raise ValueError("Invalid seed/test-evaluation policy")
    partition_path, output = Path(partition_path), Path(output)
    if output.exists():
        raise FileExistsError(output)
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    frame, schema, source = load_dataset(partition["dataset"])
    if file_hash(source) != partition["source_sha256"]:
        raise ValueError("Source data changed since partition creation")
    x, y, preprocessing = prepare_arrays(frame, schema, partition)
    train, validation, test = (partition["splits"][name] for name in ["train","validation","test"])
    parameters = parameters or {}
    estimator = make_estimator(kind, parameters, seed)
    output.mkdir(parents=True)
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(x[train], y[train])
    seconds = time.perf_counter()-started
    if not np.array_equal(estimator.classes_, np.arange(len(partition["classes"]))):
        raise ValueError("Estimator class ordering differs from global schema")
    validation_probability = estimator.predict_proba(x[validation])
    validation_metrics = probability_metrics(y[validation], validation_probability, partition["classes"])
    test_metrics = None
    if evaluate_test:
        probability = estimator.predict_proba(x[test])
        test_metrics = probability_metrics(y[test], probability, partition["classes"])
        np.savez_compressed(output/"test_predictions.npz", row_ids=np.array(test), labels=y[test], probabilities=probability)
    joblib.dump(estimator, output/"model.joblib")
    result = {"model":kind, "parameters":parameters, "seed":seed, "validation":validation_metrics,
              "test":test_metrics, "test_evaluated":evaluate_test, "fit_seconds":seconds}
    write_json(output/"summary.json", result)
    write_json(output/"manifest.json", {
        "partition_sha256":file_hash(partition_path), "source_sha256":file_hash(source),
        "code_hashes":{p.name:file_hash(p) for p in [Path(__file__), Path(__file__).with_name("data.py"),
                                                 Path(__file__).with_name("metrics.py")]},
        "preprocessing":preprocessing, "dependencies":{p:importlib.metadata.version(p) for p in
                                                           ["numpy","pandas","scikit-learn","joblib"]},
        "xgboost_version":importlib.metadata.version("xgboost") if kind == "xgboost" else None,
        "privacy":"NONE: centralized public-data reference", "selection":"validation only",
        "parameters":serializable_parameters(estimator.get_params()), "model_sha256":file_hash(output/"model.joblib")})
    print(f"{partition['dataset']} {kind}: validation macro-F1={validation_metrics['macro_f1']:.4f}", flush=True)
    return result
