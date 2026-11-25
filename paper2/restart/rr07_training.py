"""Fixed-RR07 outer refit and once-only test inference for the frozen neural study."""
from __future__ import annotations

import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from paper2.federated import clone_state
from paper2.restart.matched_neural_training import (aggregate_round, fixed_step_examples,
                                                    make_regressor, train_client, validate_run)
from paper2.restart.validation_search import atomic_checkpoint, atomic_json, metrics, read, sha


def training_run(run):
    required = {"id", "fold", "seed", "variant", "algorithm", "aggregation_objective", "outer_test_inference",
                "optimizer", "schedule", "hidden", "learning_rate", "weight_decay", "rounds", "local_steps"}
    core_required = required - {"seed"}
    if set(run) == required:
        if type(run["seed"]) is not int:
            raise ValueError("Unexpected RR07 run schema")
        value = {key: item for key, item in run.items() if key != "seed"}
    elif set(run) == core_required:
        value = dict(run)
    else:
        raise ValueError("Unexpected RR07 run schema")
    validate_run(value, fedprox_mu=0.01)
    return value


def _state(state, features, hidden):
    shapes = {"0.weight": (hidden, features), "0.bias": (hidden,), "2.weight": (1, hidden), "2.bias": (1,)}
    if not isinstance(state, dict) or set(state) != set(shapes):
        raise ValueError("RR07 model state schema changed")
    for name, shape in shapes.items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.device.type != "cpu" or tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError("Invalid RR07 model state")


def _groups(fold, rows):
    if fold.get("kind") != "outer_refit" or not fold.get("test") or fold.get("validation") or set(fold["train"]) & set(fold["test"]):
        raise ValueError("Outer refit/test isolation changed")
    lookup = {row: index for index, row in enumerate(fold["train"])}
    if len(lookup) != rows:
        raise ValueError("Outer training identity changed")
    groups = {name: [lookup[row] for row in values] for name, values in fold["clients"].items()}
    if not groups or any(not values for values in groups.values()) or sorted(index for values in groups.values() for index in values) != list(range(rows)):
        raise ValueError("Outer clients must partition training rows")
    return groups


def _checkpoint(saved, run, token, features, complete=False):
    required = {"identity", "done", "state", "history", "central_optimizer_state"}
    if set(saved) != required or saved["identity"] != token or type(saved["done"]) is not int:
        raise ValueError("RR07 checkpoint identity/schema changed")
    done, history = saved["done"], saved["history"]
    if not 1 <= done <= run["rounds"] or (complete and done != run["rounds"]) or [item["round"] for item in history] != list(range(1, done + 1)):
        raise ValueError("RR07 checkpoint progress changed")
    for item in history:
        if item["client_updates"] < 1 or item["optimizer_steps"] < 1 or item["examples_processed"] < 1 or not math.isfinite(item["elapsed_seconds"]) or item["elapsed_seconds"] < 0:
            raise ValueError("RR07 work history changed")
    _state(saved["state"], features, run["hidden"])
    if run["algorithm"] == "centralized" and run["optimizer"] == "adamw" and not isinstance(saved["central_optimizer_state"], dict):
        raise ValueError("RR07 centralized AdamW state missing")
    if run["algorithm"] != "centralized" and saved["central_optimizer_state"] is not None:
        raise ValueError("RR07 federated optimizer state must reset per round")
    return saved


def fit(folder, run, settings, fold, arrays, token, stop_after=None):
    """Refit exclusively on an outer fold's training arrays; it never accepts test arrays."""
    seed = run.get("seed")
    run = training_run(run)
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    if set(arrays) != {"x_train", "y_train"}:
        raise ValueError("RR07 fit may only receive training arrays")
    x, y = arrays["x_train"], arrays["y_train"]
    if x.dtype != np.float32 or y.dtype != np.float32 or x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Finite float32 outer training arrays required")
    groups = _groups(fold, len(y))
    receipt = folder / "fit.json"
    if receipt.exists():
        return verify_fit(folder, run, token, x.shape[1])
    if type(settings.get("seed")) is not int or type(settings.get("batch_size")) is not int or settings["batch_size"] < 1 or not isinstance(settings.get("fedprox_mu"), (int, float)):
        raise ValueError("Invalid frozen RR07 settings")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(settings["seed"] + seed)
    model = make_regressor(x.shape[1], run["hidden"])
    checkpoint = folder / "checkpoint.pt"
    done, history, central_optimizer_state = 0, [], None
    if checkpoint.exists():
        saved = _checkpoint(torch.load(checkpoint, map_location="cpu", weights_only=True), run, token, x.shape[1])
        model.load_state_dict(saved["state"]); done, history, central_optimizer_state = saved["done"], saved["history"], saved["central_optimizer_state"]
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)
    for round_number in range(done + 1, run["rounds"] + 1):
        started, base, updates = time.perf_counter(), clone_state(model), []
        federated_examples = sum(fixed_step_examples(len(ids), settings["batch_size"], run["local_steps"]) for ids in groups.values())
        if run["algorithm"] == "centralized":
            updates.append(train_client(model, tx, ty, "central", round_number, fold["preprocessing_identity"],
                settings["seed"] + seed + 1009 * round_number, run, batch_size=settings["batch_size"],
                fedprox_mu=settings["fedprox_mu"], steps=run["local_steps"] * len(groups), optimizer_state=central_optimizer_state,
                example_budget=federated_examples, device=settings.get("device", "cpu")))
            expected, expected_steps = {"central": len(y)}, run["local_steps"] * len(groups)
            central_optimizer_state = updates[0].optimizer_state
        else:
            expected, expected_steps = {name: len(ids) for name, ids in groups.items()}, None
            for number, name in enumerate(sorted(groups)):
                ids = groups[name]
                updates.append(train_client(model, tx[ids], ty[ids], name, round_number, fold["preprocessing_identity"],
                    settings["seed"] + seed + 1009 * round_number + number, run, batch_size=settings["batch_size"],
                    fedprox_mu=settings["fedprox_mu"], device=settings.get("device", "cpu")))
        model.load_state_dict(aggregate_round(base, updates, run, round_number, fold["preprocessing_identity"], expected, expected_steps))
        if sum(update.steps for update in updates) != run["local_steps"] * len(groups) or sum(update.examples_processed for update in updates) != federated_examples:
            raise ValueError("RR07 matched training work changed")
        history.append({"round": round_number, "client_updates": len(updates), "optimizer_steps": sum(update.steps for update in updates),
                        "examples_processed": sum(update.examples_processed for update in updates), "elapsed_seconds": time.perf_counter() - started})
        atomic_checkpoint(checkpoint, {"identity": token, "done": round_number, "state": clone_state(model), "history": history,
                                       "central_optimizer_state": central_optimizer_state})
        if stop_after == round_number and round_number < run["rounds"]:
            return None
    record = {"identity": token, "run": run, "checkpoint_sha256": sha(checkpoint), "rounds": run["rounds"], "outer_test_inference": False}
    atomic_json(receipt, record)
    return verify_fit(folder, run, token, x.shape[1])


def verify_fit(folder, run, token, features):
    run = training_run(run); folder = Path(folder)
    saved = _checkpoint(torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True), run, token, features, True)
    record = read(folder / "fit.json")
    expected = {"identity": token, "run": run, "checkpoint_sha256": sha(folder / "checkpoint.pt"), "rounds": run["rounds"], "outer_test_inference": False}
    if record != expected or saved["done"] != run["rounds"]:
        raise ValueError("RR07 fit receipt changed")
    return record


def evaluate_once(folder, run, x, y, ids, token):
    """Perform exactly one outer-test prediction after a verified fixed-budget fit."""
    run = training_run(run); folder = Path(folder)
    fit_record = verify_fit(folder, run, token, x.shape[1])
    completion, started = folder / "completion.json", folder / "inference_started.json"
    if completion.exists():
        return verify_completed(folder, run, token, x.shape[1])
    if started.exists():
        raise RuntimeError("Ambiguous prior RR07 test inference; automatic retry forbidden")
    if x.dtype != np.float32 or y.dtype != np.float32 or x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(ids) != len(y) or not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Finite float32 RR07 test arrays required")
    atomic_json(started, {"identity": token, "fit_sha256": sha(folder / "fit.json"), "checkpoint_sha256": fit_record["checkpoint_sha256"]})
    model = make_regressor(x.shape[1], run["hidden"])
    model.load_state_dict(torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)["state"]); model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(x))[:, 0].cpu().numpy()
    with (folder / "test_cache.tmp").open("wb") as stream:
        np.savez_compressed(stream, y=y, prediction=prediction, ids=ids)
    os.replace(folder / "test_cache.tmp", folder / "test_cache.npz")
    atomic_json(folder / "result.json", {"identity": token, "run": run, "test": metrics(y, prediction), "outer_test_inference": True})
    atomic_json(completion, {"identity": token, "assets": {name: sha(folder / name) for name in
                ("checkpoint.pt", "fit.json", "inference_started.json", "test_cache.npz", "result.json")}})
    return verify_completed(folder, run, token, x.shape[1])


def verify_completed(folder, run, token, features):
    run = training_run(run); folder = Path(folder)
    receipt = read(folder / "completion.json")
    if receipt.get("identity") != token or set(receipt.get("assets", {})) != {"checkpoint.pt", "fit.json", "inference_started.json", "test_cache.npz", "result.json"} or any(sha(folder / name) != digest for name, digest in receipt["assets"].items()):
        raise ValueError("RR07 completion receipt changed")
    verify_fit(folder, run, token, features)
    with np.load(folder / "test_cache.npz", allow_pickle=False) as cached:
        measured = metrics(cached["y"], cached["prediction"])
    result = read(folder / "result.json")
    expected = {"identity": token, "run": run, "test": measured, "outer_test_inference": True}
    if result != expected:
        raise ValueError("RR07 result/cache mismatch")
    return result
