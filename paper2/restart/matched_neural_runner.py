"""Identity-bound, resumable RR06 inner-only execution runner."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil
import statistics
import time
import traceback
import os

import numpy as np
import torch

from paper2.federated import clone_state
from paper2.restart import matched_neural_protocol as protocol
from paper2.restart.control_convergence import load_inner
from paper2.restart.matched_neural_training import aggregate_round, fixed_step_examples, make_regressor, train_client
from paper2.restart.validation_search import ROOT, atomic_checkpoint, atomic_json, identity, read, runtime, sha, writer_lock


RESULTS = ROOT / "paper2/results"
PROTOCOL = RESULTS / "restart_matched_neural_protocol_v6"
CODE = (
    Path(__file__),
    Path(__file__).with_name("matched_neural_training.py"),
    Path(__file__).with_name("matched_neural_protocol.py"),
    Path(__file__).with_name("control_convergence.py"),
    Path(__file__).with_name("validation_search.py"),
    ROOT / "paper2/federated.py",
)


def _safe_output(output, protected=()):
    output = Path(output).resolve()
    if RESULTS not in output.parents or output.exists():
        raise ValueError("Output must be a new Paper 2 results directory")
    for item in protected:
        item = Path(item).resolve()
        if output == item or item in output.parents or output in item.parents:
            raise ValueError("Output overlaps a protected package")
    return output


def verify_protocol(package=PROTOCOL, data_override=None):
    package = Path(package).resolve()
    verified = protocol.verify(package, data_override)
    manifest = read(package / "manifest.json")
    plan, settings = read(package / "plan.json"), read(package / "config.json")
    if verified != {"status": "PASS_FROZEN_MATCHED_NEURAL_PROTOCOL", "planned_runs": 504, "training_started": False}:
        raise ValueError("Unexpected RR06 protocol verification")
    if len(plan) != 504 or manifest["planned_runs"] != 504 or settings["outer_test_inference"] is not False:
        raise ValueError("RR06 protocol scope differs")
    data = Path(data_override).resolve() if data_override is not None else Path(manifest["data"]).resolve()
    return manifest, settings, plan, data


def freeze(output, package=PROTOCOL):
    package = Path(package).resolve()
    protocol_manifest, _, plan, data = verify_protocol(package)
    output = _safe_output(output, (package, data))
    output.mkdir(parents=True); (output / "source_snapshot").mkdir()
    for source in CODE:
        shutil.copyfile(source, output / "source_snapshot" / source.name)
    atomic_json(output / "manifest.json", {
        "status": "FROZEN_MATCHED_NEURAL_EXECUTION_NO_TRAINING",
        "protocol": str(package), "protocol_manifest_sha256": sha(package / "manifest.json"),
        "protocol_identity": identity(protocol_manifest), "runtime": runtime(),
        "source_sha256": {source.name: sha(source) for source in CODE},
        "planned_runs": len(plan), "outer_test_inference": False,
    })
    return {"status": "FROZEN_MATCHED_NEURAL_EXECUTION_NO_TRAINING", "planned_runs": len(plan)}


def verify_execution(package, protocol_override=None, data_override=None):
    package = Path(package).resolve(); manifest = read(package / "manifest.json")
    if manifest.get("status") != "FROZEN_MATCHED_NEURAL_EXECUTION_NO_TRAINING" or manifest.get("runtime") != runtime():
        raise ValueError("Execution package status/runtime changed")
    protocol_package = Path(protocol_override).resolve() if protocol_override is not None else Path(manifest["protocol"]).resolve()
    protocol_manifest, settings, plan, data = verify_protocol(protocol_package, data_override)
    if sha(protocol_package / "manifest.json") != manifest["protocol_manifest_sha256"] or identity(protocol_manifest) != manifest["protocol_identity"]:
        raise ValueError("Execution protocol binding changed")
    if set(manifest["source_sha256"]) != {source.name for source in CODE}:
        raise ValueError("Execution source inventory changed")
    for source in CODE:
        expected = manifest["source_sha256"][source.name]
        if sha(source) != expected or sha(package / "source_snapshot" / source.name) != expected:
            raise ValueError("Execution source changed")
    if manifest["planned_runs"] != len(plan) or manifest["outer_test_inference"] is not False:
        raise ValueError("Execution scope changed")
    return manifest, settings, plan, data


def validation_metrics(y, prediction):
    y, prediction = np.asarray(y, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if y.ndim != 1 or prediction.shape != y.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Finite aligned validation arrays required")
    error = prediction - y
    return {"samples": len(y), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))), "negative_predictions": int((prediction < 0).sum())}


def _check_state(state, features, hidden):
    shapes = {"0.weight": (hidden, features), "0.bias": (hidden,), "2.weight": (1, hidden), "2.bias": (1,)}
    if not isinstance(state, dict) or set(state) != set(shapes):
        raise ValueError("Model state schema changed")
    for name, shape in shapes.items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.device.type != "cpu" or tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError("Invalid model state tensor")


def validate_checkpoint(saved, run, token, features, complete=False):
    required = {"identity", "done", "state", "history", "best", "best_round", "best_state", "central_optimizer_state"}
    if set(saved) != required or saved["identity"] != token or type(saved["done"]) is not int:
        raise ValueError("Checkpoint identity/schema changed")
    done, history = saved["done"], saved["history"]
    if not 1 <= done <= run["rounds"] or (complete and done != run["rounds"]) or [row["round"] for row in history] != list(range(1, done + 1)):
        raise ValueError("Checkpoint progress changed")
    for row in history:
        metric = row["validation"]
        if type(metric["samples"]) is not int or metric["samples"] < 1 or any(not math.isfinite(metric[name]) or metric[name] < 0 for name in ("rmse", "mae")):
            raise ValueError("Invalid validation history")
        if type(row["client_updates"]) is not int or row["client_updates"] < 1 or type(row["optimizer_steps"]) is not int or row["optimizer_steps"] < 1 or type(row["examples_processed"]) is not int or row["examples_processed"] < 1 or not math.isfinite(row["elapsed_seconds"]) or row["elapsed_seconds"] < 0:
            raise ValueError("Invalid measured work history")
    best = min(history, key=lambda row: row["validation"]["rmse"])
    if saved["best"] != best["validation"]["rmse"] or saved["best_round"] != best["round"]:
        raise ValueError("Best-round first-tie rule changed")
    _check_state(saved["state"], features, run["hidden"]); _check_state(saved["best_state"], features, run["hidden"])
    if run["algorithm"] == "centralized" and run["optimizer"] == "adamw" and not isinstance(saved["central_optimizer_state"], dict):
        raise ValueError("Persistent centralized AdamW state missing")
    if run["algorithm"] != "centralized" and saved["central_optimizer_state"] is not None:
        raise ValueError("Federated optimizer state must reset each client round")
    return saved


def _fold_groups(fold, train_rows):
    train, validation, test = set(fold["train"]), set(fold["validation"]), set(fold["test"])
    if fold["kind"] != "inner_selection" or train & validation or train & test or validation & test:
        raise ValueError("Inner-only fold isolation changed")
    lookup = {row: index for index, row in enumerate(fold["train"])}
    if len(lookup) != train_rows:
        raise ValueError("Training row identity changed")
    groups = {name: [lookup[row] for row in rows] for name, rows in fold["clients"].items()}
    if any(not rows for rows in groups.values()) or sorted(index for rows in groups.values() for index in rows) != list(range(train_rows)):
        raise ValueError("Client membership must partition inner training rows")
    return groups


def candidate(folder, run, settings, fold, arrays, token, stop_after=None):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    x, y = arrays["x_train"], arrays["y_train"]
    xv, yv = arrays["x_validation"], arrays["y_validation"]
    if x.dtype != np.float32 or y.dtype != np.float32 or xv.dtype != np.float32 or yv.dtype != np.float32:
        raise ValueError("Prepared float32 arrays required")
    groups = _fold_groups(fold, len(y))
    if (folder / "completion.json").exists():
        return completed(folder, run, token, x.shape[1])
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(settings["seed"])
    model = make_regressor(x.shape[1], run["hidden"])
    done, history, best, best_round, best_state, central_optimizer_state = 0, [], float("inf"), None, None, None
    checkpoint = folder / "checkpoint.pt"
    if checkpoint.exists():
        saved = validate_checkpoint(torch.load(checkpoint, map_location="cpu", weights_only=True), run, token, x.shape[1])
        model.load_state_dict(saved["state"])
        done, history, best, best_round, best_state, central_optimizer_state = saved["done"], saved["history"], saved["best"], saved["best_round"], saved["best_state"], saved["central_optimizer_state"]
    tx, ty, tvx = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(xv)
    for round_number in range(done + 1, run["rounds"] + 1):
        started = time.perf_counter(); base = clone_state(model); updates = []
        federated_examples = sum(fixed_step_examples(len(ids), settings["batch_size"], run["local_steps"]) for ids in groups.values())
        if run["algorithm"] == "centralized":
            central_steps = run["local_steps"] * len(groups)
            updates.append(train_client(model, tx, ty, "central", round_number, fold["preprocessing_identity"],
                settings["seed"] + 1009 * round_number, run, batch_size=settings["batch_size"], fedprox_mu=settings["fedprox_mu"],
                steps=central_steps, optimizer_state=central_optimizer_state, example_budget=federated_examples,
                device=settings.get("device", "cpu")))
            expected = {"central": len(y)}
            central_optimizer_state = updates[0].optimizer_state
        else:
            expected = {name: len(ids) for name, ids in groups.items()}
            for number, name in enumerate(sorted(groups)):
                ids = groups[name]
                updates.append(train_client(model, tx[ids], ty[ids], name, round_number, fold["preprocessing_identity"],
                    settings["seed"] + 1009 * round_number + number, run, batch_size=settings["batch_size"],
                    fedprox_mu=settings["fedprox_mu"], device=settings.get("device", "cpu")))
        expected_steps = run["local_steps"] * len(groups) if run["algorithm"] == "centralized" else None
        model.load_state_dict(aggregate_round(base, updates, run, round_number, fold["preprocessing_identity"], expected, expected_steps))
        if sum(update.steps for update in updates) != run["local_steps"] * len(groups) or sum(update.examples_processed for update in updates) != federated_examples:
            raise ValueError("Matched optimizer-step/example work differs")
        model.eval()
        with torch.no_grad(): prediction = model(tvx)[:, 0].cpu().numpy()
        measured = validation_metrics(yv, prediction)
        history.append({"round": round_number, "validation": measured, "client_updates": len(updates),
                        "optimizer_steps": sum(update.steps for update in updates),
                        "examples_processed": sum(update.examples_processed for update in updates),
                        "elapsed_seconds": time.perf_counter() - started})
        if measured["rmse"] < best:
            best, best_round, best_state = measured["rmse"], round_number, clone_state(model)
        atomic_checkpoint(checkpoint, {"identity": token, "done": round_number, "state": clone_state(model),
                          "history": history, "best": best, "best_round": best_round, "best_state": best_state,
                          "central_optimizer_state": central_optimizer_state})
        if stop_after == round_number and round_number < run["rounds"]:
            return None
    result = {"identity": token, "run": run, "best_validation_rmse": best, "best_round": best_round,
              "rounds": run["rounds"], "total_elapsed_seconds": sum(row["elapsed_seconds"] for row in history),
              "outer_test_inference": False}
    atomic_json(folder / "result.json", result)
    atomic_json(folder / "completion.json", {"identity": token,
                "assets": {name: sha(folder / name) for name in ("result.json", "checkpoint.pt")}})
    return completed(folder, run, token, x.shape[1])


def completed(folder, run, token, features):
    folder = Path(folder); receipt = read(folder / "completion.json")
    if receipt.get("identity") != token or set(receipt.get("assets", {})) != {"result.json", "checkpoint.pt"}:
        raise ValueError("Completion receipt changed")
    if any(sha(folder / name) != digest for name, digest in receipt["assets"].items()):
        raise ValueError("Completed asset changed")
    saved = validate_checkpoint(torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True), run, token, features, True)
    result = read(folder / "result.json")
    if result != {"identity": token, "run": run, "best_validation_rmse": saved["best"], "best_round": saved["best_round"],
                  "rounds": run["rounds"], "total_elapsed_seconds": sum(row["elapsed_seconds"] for row in saved["history"]),
                  "outer_test_inference": False}:
        raise ValueError("Result/checkpoint mismatch")
    return result


def reconcile_cached_attempt(folder, attempts_path, attempts):
    """Commit a recovered journal after a valid completion receipt is verified."""
    if not attempts:
        raise ValueError("Completed run has no attempt journal")
    if attempts[-1]["status"] == "STARTED":
        attempts[-1].update(status="CACHE_VERIFIED_AFTER_INTERRUPTION",
                            completion_sha256=sha(Path(folder) / "completion.json"))
        atomic_json(attempts_path, attempts)
    elif attempts[-1]["status"] not in {"COMPLETE", "CACHE_VERIFIED_AFTER_INTERRUPTION"}:
        raise ValueError("Completed run attempt journal is inconsistent")
    elif attempts[-1].get("completion_sha256") != sha(Path(folder) / "completion.json"):
        raise ValueError("Completed run journal hash differs")
    return attempts


def select_complete(plan, results):
    if set(results) != {run["id"] for run in plan}:
        raise ValueError("All 504 candidates are required before selection")
    selections = []
    for outer in sorted({run["fold"].split("_inner_")[0] for run in plan}):
        for algorithm, objective in sorted({(run["algorithm"], run["aggregation_objective"]) for run in plan}):
            rows = [run for run in plan if run["fold"].split("_inner_")[0] == outer and run["algorithm"] == algorithm and run["aggregation_objective"] == objective]
            if not rows: continue
            options = []
            for optimizer in ("sgd", "adamw"):
                matched = [run for run in rows if run["optimizer"] == optimizer]
                if not matched: continue
                if len(matched) != 7 or len({run["fold"] for run in matched}) != 7:
                    raise ValueError("Seven inner groups per optimizer candidate required")
                options.append({"optimizer": optimizer, "hidden": matched[0]["hidden"], "schedule": matched[0]["schedule"],
                    "equal_inner_group_mean_best_rmse": statistics.mean(results[run["id"]]["best_validation_rmse"] for run in matched),
                    "best_rounds": [results[run["id"]]["best_round"] for run in matched]})
            chosen = min(options, key=lambda option: option["equal_inner_group_mean_best_rmse"])
            selections.append({"outer_fold": outer, "algorithm": algorithm, "aggregation_objective": objective,
                               "selected": chosen, "refit_rounds": int(statistics.median(chosen["best_rounds"])),
                               "round_cap_hits": sum(value == next(run["rounds"] for run in rows if run["optimizer"] == chosen["optimizer"]) for value in chosen["best_rounds"])})
    return {"status": "COMPLETE_INNER_SELECTION", "candidates": len(results), "selections": selections,
            "outer_test_inference": False, "independent_confirmation": False}


def candidate_worker(folder, run, settings, fold, arrays, token, device="cpu"):
    """Top-level spawn-safe worker; every process owns a disjoint run folder."""
    settings = dict(settings, device=device)
    return candidate(folder, run, settings, fold, arrays, token)


def execute(package, output, cap=1, resume=False, dry_run=True, retry_failed=False, workers=1,
            protocol_override=None, data_override=None):
    if type(cap) is not int or cap < 1: raise ValueError("Positive integer cap required")
    if type(workers) is not int or not 1 <= workers <= 8: raise ValueError("Workers must be an integer from 1 to 8")
    package = Path(package).resolve(); manifest, settings, plan, data = verify_execution(package, protocol_override, data_override)
    output = Path(output).resolve()
    if RESULTS not in output.parents or output in (package, data) or package in output.parents or data in output.parents:
        raise ValueError("Execution output overlaps a protected path")
    device_policy = os.environ.get("PAPER2_DEVICE", "cpu").strip().lower()
    if device_policy not in {"cpu", "cuda"}:
        raise ValueError("PAPER2_DEVICE must be cpu or cuda")
    if device_policy == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PAPER2_DEVICE=cuda requested but CUDA is unavailable")
    if device_policy == "cuda" and torch.cuda.device_count() < workers:
        raise RuntimeError(f"Requested {workers} GPU workers but only {torch.cuda.device_count()} GPU(s) are available")
    if dry_run: return {"status": "DRY_RUN_NO_TRAINING", "planned": len(plan), "cap": cap, "workers": workers, "device": device_policy}
    if output.exists() and not resume: raise ValueError("Explicit resume required")
    output.mkdir(parents=True, exist_ok=True); binding = identity(manifest); results = {}; executed = 0
    with writer_lock(output):
        binding_path = output / "binding.json"
        if binding_path.exists():
            if read(binding_path) != {"identity": binding}: raise ValueError("Execution binding changed")
        elif resume or any(item.name != "writer.lock" for item in output.iterdir()):
            raise ValueError("Unbound output cannot be resumed")
        else: atomic_json(binding_path, {"identity": binding})
        tasks = []
        for run in plan:
            folder = output / "runs" / run["id"]; token = identity({"binding": binding, "run": run})
            attempts_path = folder / "attempts.json"; attempts = read(attempts_path) if attempts_path.exists() else []
            if (folder / "completion.json").exists():
                fold, arrays = load_inner(data, run["fold"])
                results[run["id"]] = completed(folder, run, token, arrays["x_train"].shape[1])
                reconcile_cached_attempt(folder, attempts_path, attempts)
                continue
            if len(tasks) >= cap: continue
            if attempts and attempts[-1]["status"] == "FAILED" and not retry_failed: raise RuntimeError("Inspect failure and explicitly request retry")
            folder.mkdir(parents=True, exist_ok=True)
            attempts.append({"attempt": len(attempts) + 1, "status": "STARTED", "identity": token,
                             "started_utc": datetime.now(timezone.utc).isoformat()}); atomic_json(attempts_path, attempts)
            try:
                fold, arrays = load_inner(data, run["fold"])
                tasks.append({"run": run, "folder": folder, "token": token, "fold": fold, "arrays": arrays,
                              "attempts": attempts, "attempts_path": attempts_path})
            except Exception:
                attempts[-1].update(status="FAILED", traceback=traceback.format_exc()); atomic_json(attempts_path, attempts); raise
        failures = []
        if workers == 1:
            completed_tasks = []
            for task in tasks:
                try:
                    value = candidate_worker(task["folder"], task["run"], settings, task["fold"], task["arrays"], task["token"],
                                             "cuda:0" if device_policy == "cuda" else "cpu")
                    completed_tasks.append((task, value, None))
                except Exception as exc:
                    completed_tasks.append((task, None, (exc, traceback.format_exc())))
        elif tasks:
            completed_tasks = []
            with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
                futures = {pool.submit(candidate_worker, task["folder"], task["run"], settings,
                                       task["fold"], task["arrays"], task["token"],
                                       f"cuda:{index % workers}" if device_policy == "cuda" else "cpu"): task
                            for index, task in enumerate(tasks)}
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        completed_tasks.append((task, future.result(), None))
                    except Exception as exc:
                        completed_tasks.append((task, None, (exc, traceback.format_exc())))
        else:
            completed_tasks = []
        for task, value, failure in completed_tasks:
            attempts, attempts_path = task["attempts"], task["attempts_path"]
            if failure is None:
                results[task["run"]["id"]] = completed(task["folder"], task["run"], task["token"], task["arrays"]["x_train"].shape[1])
                if value != results[task["run"]["id"]]: raise ValueError("Worker/parent completion result differs")
                attempts[-1].update(status="COMPLETE", completion_sha256=sha(task["folder"] / "completion.json")); atomic_json(attempts_path, attempts); executed += 1
            else:
                exc, trace = failure; attempts[-1].update(status="FAILED", traceback=trace); atomic_json(attempts_path, attempts); failures.append(exc)
        if failures:
            raise RuntimeError(f"{len(failures)} parallel candidate(s) failed; inspect attempt journals") from failures[0]
        if len(results) == len(plan): atomic_json(output / "selection.json", select_complete(plan, results))
        progress = {"status": "COMPLETE" if len(results) == len(plan) else "PARTIAL", "completed": len(results),
                    "planned": len(plan), "pending": len(plan) - len(results), "executed": executed,
                    "selection_available": len(results) == len(plan), "outer_test_inference": False,
                    "workers": workers, "device": device_policy}
        atomic_json(output / "progress.json", progress); return progress


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--package", type=Path, default=RESULTS / "restart_matched_neural_execution_v1")
    parser.add_argument("--freeze", action="store_true"); parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--retry-failed", action="store_true"); parser.add_argument("--cap", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--protocol-override", type=Path); parser.add_argument("--data-override", type=Path)
    args = parser.parse_args()
    print(freeze(args.output, args.protocol) if args.freeze else execute(args.package, args.output, args.cap, args.resume, not args.execute, args.retry_failed, args.workers, args.protocol_override, args.data_override))
