"""Identity-bound RR07 outer evaluation; device-declared, resumable, and test-once."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback

import numpy as np
import torch

from paper2.restart.rr07_execution_gate import verify_execution
from paper2.restart.rr07_training import evaluate_once, fit, verify_completed
from paper2.restart.validation_search import ROOT, atomic_json, identity, read, sha, writer_lock


RESULTS = ROOT / "paper2/results"


def _fold_assets(data, fold_name):
    manifest = read(data / "manifest.json")
    if fold_name not in manifest.get("assets", {}):
        raise ValueError("RR07 fold missing from frozen data manifest")
    folder = data / "folds" / fold_name
    for name, digest in manifest["assets"][fold_name].items():
        if sha(folder / name) != digest:
            raise ValueError("RR07 fold asset changed")
    fold = read(folder / "partition.json")
    if fold.get("kind") != "outer_refit" or fold.get("validation") or set(fold["train"]) & set(fold["test"]):
        raise ValueError("RR07 outer partition isolation changed")
    if read(folder / "preprocessing.json").get("identity") != fold.get("preprocessing_identity"):
        raise ValueError("RR07 preprocessing binding changed")
    return folder, fold


def load_training(data, run):
    """Read only outer-training arrays; test arrays remain unopened."""
    folder, fold = _fold_assets(Path(data), run["fold"])
    with np.load(folder / "arrays.npz", allow_pickle=False) as saved:
        expected = {"x_train", "y_train", "ids_train", "x_test", "y_test", "ids_test"}
        if set(saved.files) != expected:
            raise ValueError("Unexpected RR07 array schema")
        arrays = {"x_train": saved["x_train"].copy(), "y_train": saved["y_train"].copy(), "ids_train": saved["ids_train"].copy()}
    np.testing.assert_array_equal(arrays["ids_train"], fold["train"])
    if arrays["x_train"].dtype != np.float32 or arrays["y_train"].dtype != np.float32 or arrays["x_train"].ndim != 2 or arrays["y_train"].ndim != 1 or len(arrays["x_train"]) != len(arrays["y_train"]) or not len(arrays["y_train"]) or not np.isfinite(arrays["x_train"]).all() or not np.isfinite(arrays["y_train"]).all():
        raise ValueError("Invalid RR07 outer-training arrays")
    return fold, {"x_train": arrays["x_train"], "y_train": arrays["y_train"]}


def load_test(data, run):
    """Read test arrays only after a corresponding fit receipt exists."""
    folder, fold = _fold_assets(Path(data), run["fold"])
    with np.load(folder / "arrays.npz", allow_pickle=False) as saved:
        arrays = {"x_test": saved["x_test"].copy(), "y_test": saved["y_test"].copy(), "ids_test": saved["ids_test"].copy()}
    np.testing.assert_array_equal(arrays["ids_test"], fold["test"])
    if arrays["x_test"].dtype != np.float32 or arrays["y_test"].dtype != np.float32 or arrays["x_test"].ndim != 2 or arrays["y_test"].ndim != 1 or len(arrays["x_test"]) != len(arrays["y_test"]) or not len(arrays["y_test"]) or not np.isfinite(arrays["x_test"]).all() or not np.isfinite(arrays["y_test"]).all():
        raise ValueError("Invalid RR07 outer-test arrays")
    return fold, arrays


def dry_run(package):
    manifest, plan, data = verify_execution(package)
    return {"status": "DRY_RUN_NO_OUTER_INFERENCE", "planned_runs": len(plan),
            "outer_test_inference": False, "data": str(data),
            "remote_gpu_required": True, "package_identity": identity(manifest)}


def execute(package, output, cap=1, resume=False, retry_failed=False, allow_outer_inference=False):
    if allow_outer_inference is not True:
        raise PermissionError("RR07 outer inference requires explicit authorization after RR06 finalization")
    if type(cap) is not int or cap < 1:
        raise ValueError("Positive RR07 execution cap required")
    device = os.environ.get("PAPER2_DEVICE", "cpu").strip().lower()
    if device not in {"cpu", "cuda"}:
        raise ValueError("PAPER2_DEVICE must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PAPER2_DEVICE=cuda requested but CUDA is unavailable")
    package = Path(package).resolve(); manifest, plan, data = verify_execution(package)
    output = Path(output).resolve()
    protected = (package, Path(manifest["plan_package"]).resolve(), data)
    if RESULTS not in output.parents or any(output == path or path in output.parents or output in path.parents for path in protected):
        raise ValueError("RR07 output must be a separate Paper 2 results directory")
    if output.exists() and not resume:
        raise ValueError("Existing RR07 output requires explicit resume")
    output.mkdir(parents=True, exist_ok=True)
    binding = identity(manifest); completed, executed = [], 0
    settings = {"seed": 42, "batch_size": 32, "fedprox_mu": 0.01, "device": "cuda:0" if device == "cuda" else "cpu"}
    with writer_lock(output):
        binding_path = output / "binding.json"
        if binding_path.exists():
            if read(binding_path) != {"identity": binding}:
                raise ValueError("RR07 output binding changed")
        elif resume or any(item.name != "writer.lock" for item in output.iterdir()):
            raise ValueError("Unbound RR07 output cannot be resumed")
        else:
            atomic_json(binding_path, {"identity": binding})
        for run in plan:
            folder = output / "runs" / run["id"]; token = identity({"binding": binding, "run": run})
            attempts_path = folder / "attempts.json"; attempts = read(attempts_path) if attempts_path.exists() else []
            if (folder / "completion.json").exists():
                _, train = load_training(data, run)
                completed.append(verify_completed(folder, run, token, train["x_train"].shape[1]))
                if attempts and attempts[-1].get("status") == "STARTED":
                    attempts[-1].update(status="CACHE_VERIFIED_AFTER_INTERRUPTION", completion_sha256=sha(folder / "completion.json")); atomic_json(attempts_path, attempts)
                continue
            if executed >= cap:
                continue
            if attempts and attempts[-1].get("status") == "FAILED" and not retry_failed:
                raise RuntimeError("Inspect failed RR07 candidate and explicitly request retry")
            if (folder / "inference_started.json").exists():
                raise RuntimeError("Ambiguous RR07 test inference cannot be automatically retried")
            folder.mkdir(parents=True, exist_ok=True)
            attempts.append({"attempt": len(attempts) + 1, "status": "STARTED", "identity": token,
                             "started_utc": datetime.now(timezone.utc).isoformat()}); atomic_json(attempts_path, attempts)
            try:
                fold, train = load_training(data, run)
                fit(folder, run, settings, fold, train, token)
                test_fold, test = load_test(data, run)
                if test_fold["preprocessing_identity"] != fold["preprocessing_identity"]:
                    raise ValueError("RR07 train/test fold binding changed")
                atomic_json(folder / "partition_context.json", {"fold": run["fold"], "test_group": fold.get("test_group"),
                            "ids_test": test_fold["test"], "support": test_fold.get("support", {}).get("test", []),
                            "preprocessing_identity": fold["preprocessing_identity"]})
                completed.append(evaluate_once(folder, run, test["x_test"], test["y_test"], test["ids_test"], token))
                attempts[-1].update(status="COMPLETE", completion_sha256=sha(folder / "completion.json")); atomic_json(attempts_path, attempts); executed += 1
            except Exception as exc:
                attempts[-1].update(status="FAILED", error_type=type(exc).__name__, traceback=traceback.format_exc()); atomic_json(attempts_path, attempts)
                atomic_json(output / "progress.json", {"status": "FAILED_STOPPED", "completed": len(completed), "planned": len(plan), "failed": run["id"]})
                raise
        status = {"status": "COMPLETE" if len(completed) == len(plan) else "PARTIAL", "completed": len(completed),
                  "planned": len(plan), "pending": len(plan) - len(completed), "executed_this_invocation": executed,
                  "outer_test_inference": bool(executed), "device": device,
                  "hardware_qualification": "CPU_ONLY_NOT_GPU_EVIDENCE" if device == "cpu" else "CUDA_DECLARED_BY_EXECUTION"}
        atomic_json(output / "progress.json", status)
        return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-outer-inference", action="store_true")
    parser.add_argument("--cap", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.execute:
        if args.output is None: parser.error("--output is required with --execute")
        result = execute(args.package, args.output, args.cap, args.resume, args.retry_failed, args.allow_outer_inference)
    else:
        result = dry_run(args.package)
    print(json.dumps(result, indent=2))
