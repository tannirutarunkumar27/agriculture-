"""Fixed-budget outer refit and fail-closed, once-only inference primitives."""
import os
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from paper2.federated import aggregate, clone_state
from paper2.restart.regression_training import make_regressor, train_regression_client
from paper2.restart.validation_search import atomic_checkpoint, atomic_json, read, sha, identity, metrics

NEURAL = ("centralized", "fedavg", "fedprox", "fednova")


def fit(folder, run, settings, fold, x, y, token, stop_after=None):
    """Only training tensors accepted. stop_after supports synthetic resume tests."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if x.dtype != np.float32 or y.dtype != np.float32 or x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(y):
        raise ValueError("Invalid training schema")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite training data")
    if len(fold["train"]) != len(y) or len(set(fold["train"])) != len(y):
        raise ValueError("Invalid training IDs")
    algorithm = run["algorithm"]
    model_path = folder / ("model.pt" if algorithm in NEURAL else "model.joblib")
    receipt = folder / "fit.json"
    if receipt.exists():
        saved = read(receipt)
        if saved["identity"] != token or sha(model_path) != saved["model_sha256"]:
            raise ValueError("Fit identity/model changed")
        return saved
    if algorithm not in NEURAL:
        candidate = run["configuration"]
        if candidate["model"] != algorithm:
            raise ValueError("Reference identity mismatch")
        if algorithm == "training_mean":
            model = {"training_mean": float(y.astype(np.float64).mean())}
        elif algorithm == "ridge":
            model = Ridge(alpha=candidate["alpha"]).fit(x, y)
        elif algorithm == "random_forest":
            model = RandomForestRegressor(n_estimators=candidate["trees"], min_samples_leaf=candidate["min_samples_leaf"], random_state=run["seed"], n_jobs=1).fit(x, y)
        elif algorithm == "xgboost":
            model = XGBRegressor(n_estimators=candidate["rounds"], max_depth=candidate["max_depth"], learning_rate=candidate["learning_rate"], random_state=run["seed"], n_jobs=1, objective="reg:squarederror").fit(x, y)
        else:
            raise ValueError("Unknown algorithm")
        joblib.dump(model, folder / "model.tmp")
        os.replace(folder / "model.tmp", model_path)
        history = []
    else:
        if type(run["rounds"]) is not int or not 1 <= run["rounds"] <= 80:
            raise ValueError("Invalid fixed rounds")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(run["seed"])
        model = make_regressor(x.shape[1], settings["hidden"])
        done, history = 0, []
        checkpoint = folder / "checkpoint.pt"
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if saved["identity"] != token or not 0 <= saved["done"] <= run["rounds"]:
                raise ValueError("Checkpoint identity changed")
            done, history = saved["done"], saved["history"]
            if len(history) != done:
                raise ValueError("Checkpoint history mismatch")
            model.load_state_dict(saved["state"])
        lookup = {row: i for i, row in enumerate(fold["train"])}
        clients = {"centralized": fold["train"]} if algorithm == "centralized" else fold["clients"]
        membership = [row for rows in clients.values() for row in rows]
        if not clients or any(not rows for rows in clients.values()) or len(membership) != len(set(membership)) or set(membership) != set(lookup):
            raise ValueError("Clients must partition training rows")
        tx, ty = torch.from_numpy(x), torch.from_numpy(y)
        schema = identity({"preprocessing": fold["preprocessing_identity"], "hidden": settings["hidden"], "target": "scalar_regression"})
        for round_number in range(done + 1, run["rounds"] + 1):
            updates = []
            for number, (client, rows) in enumerate(sorted(clients.items())):
                ids = [lookup[row] for row in rows]
                updates.append(train_regression_client(model, tx[ids], ty[ids], client, round_number, schema,
                    run["seed"] + 1009 * round_number + number, learning_rate=run["learning_rate"],
                    batch_size=settings["batch_size"], mu=settings["mu"] if algorithm == "fedprox" else 0.))
            state = updates[0].state if algorithm == "centralized" else aggregate(clone_state(model), updates,
                algorithm, round_number, schema, {name: len(rows) for name, rows in clients.items()})
            model.load_state_dict(state)
            history.append({"round": round_number, "local_steps": sum(u.steps for u in updates)})
            atomic_checkpoint(checkpoint, {"identity": token, "done": round_number, "state": clone_state(model), "history": history})
            if stop_after == round_number and round_number < run["rounds"]:
                return None
        atomic_checkpoint(model_path, clone_state(model))
    result = {"identity": token, "model": model_path.name, "model_sha256": sha(model_path), "history": history}
    atomic_json(receipt, result)
    return result


def evaluate_once(folder, run, settings, x, y, ids, token):
    folder = Path(folder)
    fit_receipt = read(folder / "fit.json")
    if fit_receipt["identity"] != token or sha(folder / fit_receipt["model"]) != fit_receipt["model_sha256"]:
        raise ValueError("Unverified fitted model")
    completion = folder / "completion.json"
    if completion.exists():
        return verify_evaluation(folder, token)
    started = folder / "inference_started.json"
    if started.exists():
        raise RuntimeError("Ambiguous prior inference; automatic repeated test inference forbidden")
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(ids) != len(y) or not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid evaluation arrays")
    # Caller must hold the study writer lock. Persist intent before model load/inference.
    atomic_json(started, {"identity": token, "model_sha256": fit_receipt["model_sha256"]})
    if run["algorithm"] in NEURAL:
        model = make_regressor(x.shape[1], settings["hidden"])
        model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
        model.eval()
        with torch.no_grad():
            prediction = model(torch.from_numpy(x))[:, 0].numpy()
    else:
        model = joblib.load(folder / "model.joblib")  # Only locally produced hash-verified asset.
        prediction = np.full(y.shape, model["training_mean"]) if run["algorithm"] == "training_mean" else model.predict(x)
    result = {"identity": token, "run": run, "test": metrics(y, prediction)}
    with (folder / "test_cache.tmp").open("wb") as stream:
        np.savez_compressed(stream, y=y, prediction=prediction, ids=ids)
    os.replace(folder / "test_cache.tmp", folder / "test_cache.npz")
    atomic_json(folder / "result.json", result)
    atomic_json(completion, {"identity": token, "assets": {name: sha(folder / name) for name in
        ("fit.json", fit_receipt["model"], "test_cache.npz", "result.json", "inference_started.json")}})
    return verify_evaluation(folder, token)


def verify_evaluation(folder, token):
    folder = Path(folder)
    receipt = read(folder / "completion.json")
    if receipt["identity"] != token or any(sha(folder / name) != value for name, value in receipt["assets"].items()):
        raise ValueError("Evaluation assets changed")
    result = read(folder / "result.json")
    with np.load(folder / "test_cache.npz", allow_pickle=False) as cache:
        measured = metrics(cache["y"], cache["prediction"])
    if result["identity"] != token or measured != result["test"]:
        raise ValueError("Evaluation metrics changed")
    return result
