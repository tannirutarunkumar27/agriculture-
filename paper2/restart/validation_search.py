"""Frozen bounded validation-only regression search. Outer test data never loaded."""
from __future__ import annotations
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import sys
import time

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from paper2.federated import aggregate, clone_state
from paper2.restart.regression_training import make_regressor, train_regression_client

ROOT = Path(__file__).resolve().parents[2]
CODE = [Path(__file__), Path(__file__).with_name("run_validation_search.py"),
        Path(__file__).with_name("regression_training.py"), ROOT/"paper2/federated.py",
        ROOT/"paper2/configs/restart_validation_search_v1.json"]
PACKAGES = ("torch", "numpy", "pandas", "scikit-learn", "joblib", "xgboost")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _replace_portably(temporary, path):
    """Prefer atomic replacement, with a bounded Windows-safe fallback."""
    temporary, path = Path(temporary), Path(path)
    for delay in (0.0, 0.02, 0.05, 0.1, 0.2):
        if delay:
            time.sleep(delay)
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            pass
    shutil.copyfile(temporary, path)
    temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    _replace_portably(temporary, path)


def atomic_checkpoint(path, value):
    temporary = Path(path).with_suffix(".tmp")
    torch.save(value, temporary)
    _replace_portably(temporary, path)


@contextmanager
def writer_lock(folder):
    path = Path(folder)/"writer.lock"
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another search writer holds this output") from exc
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if y.ndim != 1 or not len(y) or prediction.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Finite aligned regression validation arrays required")
    error = prediction.astype(np.float64)-y.astype(np.float64)
    return {"samples": len(y), "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(abs(error))), "negative_predictions": int((prediction < 0).sum())}


def runtime():
    return {"python": platform.python_version(), "dependencies": {name: importlib.metadata.version(name) for name in PACKAGES}}


def validate_config(config):
    template = read(ROOT/"paper2/configs/restart_validation_search_v1.json")
    if set(config) != set(template):
        raise ValueError("Unexpected search configuration fields")
    for key in ("seed", "rounds", "hidden", "batch_size"):
        if type(config[key]) is not int or config[key] < (0 if key == "seed" else 1):
            raise ValueError("Invalid integer configuration")
    if not config["learning_rates"] or len(set(config["learning_rates"])) != len(config["learning_rates"]) or any(type(v) not in (int, float) or not np.isfinite(v) or v <= 0 for v in config["learning_rates"]):
        raise ValueError("Invalid learning-rate grid")
    for key in ("test_inference", "knowledge_enabled", "ranking_enabled", "privacy_enabled"):
        if config[key] is not False:
            raise ValueError("This search supports validation-only baselines")
    if config["algorithms"] != ["centralized", "fedavg", "fedprox", "fednova"]:
        raise ValueError("Matched four-algorithm neural search required")
    if not np.isfinite(config["mu"]) or config["mu"] < 0 or config["participation"] != 1 or config["local_epochs"] != 1:
        raise ValueError("Unsupported optimizer/participation variant")
    # Reference configurations and method/selection meanings are not arbitrary overrides.
    for key in set(template)-{"version", "seed", "rounds", "hidden", "batch_size", "learning_rates", "mu"}:
        if config[key] != template[key]:
            raise ValueError("Unsupported protocol modification: "+key)


def make_plan(config, data_manifest):
    plan = []
    folds = sorted(name for name in data_manifest["assets"] if "_inner_" in name)
    for fold in folds:
        for algorithm in config["algorithms"]:
            for index, lr in enumerate(config["learning_rates"]):
                plan.append({"id": f"{fold}_{algorithm}_lr{index}", "fold": fold, "algorithm": algorithm,
                             "candidate": index, "learning_rate": lr})
        for index, reference in enumerate(config["references"]):
            plan.append({"id": f"{fold}_reference_{index}", "fold": fold, "algorithm": reference["model"],
                         "candidate": index, "reference": reference})
    return plan


def freeze(config_path, data, verification, output):
    config_path, data, output = Path(config_path).resolve(), Path(data).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if output == data or data in output.parents or output in data.parents:
        raise ValueError("Freeze must be separate from data")
    config, prepared, verified = read(config_path), read(data/"manifest.json"), read(verification)
    validate_config(config)
    if verified["status"] != "PASS" or verified["manifest_sha256"] != sha(data/"manifest.json"):
        raise ValueError("Verified preparation binding changed")
    for name, value in prepared["sources"].items():
        if sha(ROOT/"KG-Hybrid/data/raw"/name) != value["sha256"]:
            raise ValueError("Source changed")
    for fold, assets in prepared["assets"].items():
        for name, value in assets.items():
            if sha(data/"folds"/fold/name) != value:
                raise ValueError("Prepared fold changed")
    plan = make_plan(config, prepared)
    output.mkdir(parents=True)
    atomic_json(output/"config.json", config)
    atomic_json(output/"plan.json", plan)
    (output/"source_snapshot").mkdir()
    for source in CODE:
        shutil.copyfile(source, output/"source_snapshot"/source.name)
    manifest = {"config_sha256": sha(output/"config.json"), "plan_sha256": sha(output/"plan.json"),
                "data": str(data), "data_manifest_sha256": sha(data/"manifest.json"),
                "verification_sha256": sha(verification), "runtime": runtime(),
                "code_sha256": {p.name: sha(p) for p in CODE}, "planned_runs": len(plan),
                "scope": "validation-only per-outer-fold tuning; no outer-test array loading",
                "previous_smoke_exposure": "seed123 one inner fold; development data, no fresh external confirmation"}
    atomic_json(output/"manifest.json", manifest)
    return {"status": "FROZEN", "planned_runs": len(plan), "package": str(output)}


def verify_frozen(frozen):
    frozen = Path(frozen)
    manifest = read(frozen/"manifest.json")
    for name, field in (("config.json", "config_sha256"), ("plan.json", "plan_sha256")):
        if sha(frozen/name) != manifest[field]:
            raise ValueError("Frozen protocol/plan changed")
    if runtime() != manifest["runtime"]:
        raise ValueError("Runtime differs from frozen search")
    for source in CODE:
        if sha(source) != manifest["code_sha256"][source.name] or sha(frozen/"source_snapshot"/source.name) != sha(source):
            raise ValueError("Search code changed")
    data = Path(manifest["data"])
    if sha(data/"manifest.json") != manifest["data_manifest_sha256"]:
        raise ValueError("Prepared data changed")
    prepared = read(data/"manifest.json")
    if sha(data/"regression_rows.csv") != prepared["rows_sha256"]:
        raise ValueError("Prepared row table changed")
    for fold, assets in prepared["assets"].items():
        for name, expected in assets.items():
            if sha(data/"folds"/fold/name) != expected:
                raise ValueError("Prepared fold identity changed")
    for name, value in prepared["sources"].items():
        if sha(ROOT/"KG-Hybrid/data/raw"/name) != value["sha256"]:
            raise ValueError("Raw source changed")
    config, plan = read(frozen/"config.json"), read(frozen/"plan.json")
    validate_config(config)
    if plan != make_plan(config, prepared):
        raise ValueError("Plan does not match protocol")
    return manifest, config, plan


def load_inner(data, prepared, run):
    folder = Path(data)/"folds"/run["fold"]
    for name, value in prepared["assets"][run["fold"]].items():
        if sha(folder/name) != value:
            raise ValueError("Inner asset changed")
    fold = read(folder/"partition.json")
    if fold["kind"] != "inner_selection":
        raise ValueError("Outer fold cannot enter search")
    with np.load(folder/"arrays.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"x_train", "y_train", "ids_train", "x_validation", "y_validation", "ids_validation"}:
            raise ValueError("Outer test exposed or unexpected array fields")
        arrays = {name: archive[name].copy() for name in archive.files}
    for name in ("train", "validation"):
        if arrays["x_"+name].ndim != 2 or arrays["y_"+name].shape != (len(arrays["x_"+name]),) or not np.isfinite(arrays["x_"+name]).all() or not np.isfinite(arrays["y_"+name]).all():
            raise ValueError("Malformed regression arrays")
        np.testing.assert_array_equal(arrays["ids_"+name], fold[name])
    return fold, arrays


def verify_completed(folder, run_identity):
    receipt = read(folder/"completion.json")
    if receipt["identity"] != run_identity or any(sha(folder/name) != value for name, value in receipt["assets"].items()):
        raise ValueError("Completed run identity/artifacts changed")
    result = read(folder/"result.json")
    with np.load(folder/"validation_cache.npz", allow_pickle=False) as cached:
        measured = metrics(cached["y"], cached["prediction"])
    if measured != result["validation"] or result["identity"] != run_identity:
        raise ValueError("Cached validation metrics differ")
    return result


def train_candidate(folder, run, run_identity, config, fold, arrays):
    x, y = arrays["x_train"], arrays["y_train"]
    vx, vy = arrays["x_validation"], arrays["y_validation"]
    checkpoint = folder/"checkpoint.pt"
    model_path = "model.pt" if run["algorithm"] in config["algorithms"] else "model.joblib"
    if "reference" in run:
        candidate = run["reference"]
        if run["algorithm"] == "training_mean":
            model = {"training_mean": float(y.astype(np.float64).mean())}
            prediction = np.full(vy.shape, model["training_mean"], dtype=np.float64)
        else:
            if run["algorithm"] == "ridge":
                model = Ridge(alpha=candidate["alpha"])
            elif run["algorithm"] == "random_forest":
                model = RandomForestRegressor(n_estimators=candidate["trees"], min_samples_leaf=candidate["min_samples_leaf"], random_state=config["seed"], n_jobs=1)
            else:
                model = XGBRegressor(n_estimators=candidate["rounds"], max_depth=candidate["max_depth"], learning_rate=candidate["learning_rate"], random_state=config["seed"], n_jobs=1, objective="reg:squarederror")
            model.fit(x, y)
            prediction = model.predict(vx)
        joblib.dump(model, folder/model_path)
        history, best_round, steps = [], None, None
    else:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(config["seed"])
        model = make_regressor(x.shape[1], config["hidden"])
        history, done, best, best_state, best_round = [], 0, float("inf"), None, None
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if saved["identity"] != run_identity or not 0 <= saved["done"] <= config["rounds"]:
                raise ValueError("Incompatible round checkpoint")
            model.load_state_dict(saved["state"])
            history, done, best, best_state, best_round = saved["history"], saved["done"], saved["best"], saved["best_state"], saved["best_round"]
        tx, ty, tvx = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(vx)
        lookup = {row: i for i, row in enumerate(fold["train"])}
        clients = {"centralized": fold["train"]} if run["algorithm"] == "centralized" else fold["clients"]
        schema = identity({"preprocessing": fold["preprocessing_identity"], "hidden": config["hidden"], "target": "scalar_regression"})
        for round_number in range(done+1, config["rounds"]+1):
            updates = []
            for number, (client, rows) in enumerate(sorted(clients.items())):
                ids = [lookup[row] for row in rows]
                updates.append(train_regression_client(model, tx[ids], ty[ids], client, round_number, schema,
                    config["seed"]+1009*round_number+number, learning_rate=run["learning_rate"], batch_size=config["batch_size"],
                    mu=config["mu"] if run["algorithm"] == "fedprox" else 0.))
            state = updates[0].state if run["algorithm"] == "centralized" else aggregate(clone_state(model), updates,
                run["algorithm"], round_number, schema, {name: len(rows) for name, rows in clients.items()})
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                measured = metrics(vy, model(tvx)[:, 0].numpy())
            history.append({"round": round_number, "validation": measured, "local_steps": sum(u.steps for u in updates)})
            if measured["rmse"] < best:
                best, best_state, best_round = measured["rmse"], clone_state(model), round_number
            atomic_checkpoint(checkpoint, {"identity": run_identity, "done": round_number, "state": clone_state(model),
                "history": history, "best": best, "best_state": best_state, "best_round": best_round})
        model.load_state_dict(best_state)
        with torch.no_grad():
            prediction = model(tvx)[:, 0].numpy()
        torch.save(best_state, folder/model_path)
        steps = sum(row["local_steps"] for row in history)
    result = {"id": run["id"], "fold": run["fold"], "algorithm": run["algorithm"], "candidate": run["candidate"],
              "identity": run_identity, "validation": metrics(vy, prediction), "best_round": best_round,
              "total_local_steps": steps, "history": history, "outer_test_inference": False}
    with (folder/"validation_cache.tmp").open("wb") as stream:
        np.savez_compressed(stream, y=vy, prediction=prediction)
    _replace_portably(folder/"validation_cache.tmp", folder/"validation_cache.npz")
    atomic_json(folder/"result.json", result)
    receipt = {"identity": run_identity, "assets": {name: sha(folder/name) for name in ("result.json", "validation_cache.npz", model_path)}}
    atomic_json(folder/"completion.json", receipt)
    return verify_completed(folder, run_identity)


def selections(results, config, plan):
    if len(results) != len(plan) or {r["id"] for r in results} != {r["id"] for r in plan}:
        raise ValueError("Incomplete search: no partial/favorable selection permitted")
    by_id = {r["id"]: r for r in results}
    selected = {}
    outers = sorted({run["fold"].split("_inner_")[0] for run in plan})
    for outer in outers:
        folds = sorted({run["fold"] for run in plan if run["fold"].startswith(outer+"_inner_")})
        lr_scores = [float(np.mean([by_id[f"{fold}_{algorithm}_lr{index}"]["validation"]["rmse"]
                     for fold in folds for algorithm in config["algorithms"]])) for index in range(len(config["learning_rates"]))]
        winner = min(range(len(lr_scores)), key=lambda index: lr_scores[index])
        chosen = {"common_learning_rate": config["learning_rates"][winner], "lr_validation_mean_rmse": lr_scores,
                  "outer_refit_rounds": {algorithm: int(statistics.median([by_id[f"{fold}_{algorithm}_lr{winner}"]["best_round"] for fold in folds]))
                                         for algorithm in config["algorithms"]}, "references": {}}
        for model in sorted({candidate["model"] for candidate in config["references"]}):
            indices = [index for index, candidate in enumerate(config["references"]) if candidate["model"] == model]
            means = {index: float(np.mean([by_id[f"{fold}_reference_{index}"]["validation"]["rmse"] for fold in folds])) for index in indices}
            best = min(indices, key=lambda index: means[index])
            chosen["references"][model] = {"configuration": config["references"][best], "mean_validation_rmse": means[best]}
        selected[outer] = chosen
    return selected


def execute(frozen, output, max_runs=1, resume=False, retry_failed=False, dry_run=True):
    if type(max_runs) is not int or max_runs < 1:
        raise ValueError("Positive execution cap required")
    frozen, output = Path(frozen).resolve(), Path(output).resolve()
    manifest, config, plan = verify_frozen(frozen)
    data = Path(manifest["data"])
    if any(output == path or path in output.parents or output in path.parents for path in (frozen, data)):
        raise ValueError("Execution output must be separate from frozen/data")
    if dry_run:
        return {"status": "DRY_RUN_NO_TRAINING", "planned_runs": len(plan), "max_new_runs": max_runs}
    binding = identity(manifest)
    if output.exists():
        if not resume or read(output/"binding.json")["identity"] != binding:
            raise ValueError("Existing output needs compatible explicit resume")
    else:
        output.mkdir(parents=True)
        atomic_json(output/"binding.json", {"identity": binding})
    completed, executed, failed = [], 0, []
    prepared = read(data/"manifest.json")
    with writer_lock(output):
        for run in plan:
            folder = output/"runs"/run["id"]
            run_identity = identity({"study": binding, "run": run})
            if (folder/"completion.json").exists():
                completed.append(verify_completed(folder, run_identity))
                continue
            if executed >= max_runs:
                continue
            if (folder/"failure.json").exists() and not retry_failed:
                raise ValueError("Failed run requires inspection and explicit retry: "+run["id"])
            folder.mkdir(parents=True, exist_ok=True)
            attempts = read(folder/"attempts.json") if (folder/"attempts.json").exists() else []
            attempts.append({"attempt": len(attempts)+1, "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running"})
            atomic_json(folder/"attempts.json", attempts)
            try:
                fold, arrays = load_inner(data, prepared, run)
                print("Executing validation candidate:", run["id"], flush=True)
                completed.append(train_candidate(folder, run, run_identity, config, fold, arrays))
                executed += 1
                attempts[-1]["status"] = "complete"
                atomic_json(folder/"attempts.json", attempts)
            except Exception as exc:
                failure = {"run": run, "attempt": len(attempts), "error_type": type(exc).__name__, "error": str(exc)}
                atomic_json(folder/f"failure_attempt_{len(attempts)}.json", failure)
                atomic_json(folder/"failure.json", failure)
                attempts[-1]["status"] = "failed"
                atomic_json(folder/"attempts.json", attempts)
                failed.append(run["id"])
                atomic_json(output/"progress.json", {"status": "FAILED_STOPPED", "completed": len(completed), "failed": failed, "planned": len(plan)})
                raise
            atomic_json(output/"progress.json", {"status": "partial", "completed": len(completed), "planned": len(plan), "failed": []})
        status = {"status": "complete" if len(completed) == len(plan) else "partial", "completed": len(completed),
                  "planned": len(plan), "pending": len(plan)-len(completed), "executed_this_invocation": executed, "failed": [], "outer_test_inference": False}
        if len(completed) == len(plan):
            atomic_json(output/"selected_configurations.json", selections(completed, config, plan))
        atomic_json(output/"progress.json", status)
    return status
