"""Read-only index of completed Paper 2 regression experiments.

Never loads model checkpoints or performs inference. Existing result receipts
are checked before an immutable, stage-labeled CSV is written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

STAGES = {
    "restart_validation_search_v1": (1120, "inner_validation"),
    "restart_outer_study_v1": (320, "outer_test_primary"),
    "restart_guard_study_v1": (320, "outer_test_post_hoc_guard"),
    "restart_control_search_v1": (224, "inner_validation_fixed_public"),
    "restart_public_refit_study_v1": (40, "outer_test_fixed_public_development"),
    "restart_matched_noise_study_v1": (160, "outer_test_matched_noise_development"),
    "restart_privacy_simulation_v3": (200, "outer_test_privacy_simulation_development"),
}

FIELDS = (
    "stage", "experiment_id", "method", "variant", "dataset_version", "dataset_doi",
    "fold", "outer_group", "validation_group", "seed", "client_count",
    "partition_strategy", "learning_rate", "local_epochs", "batch_size",
    "rounds_planned", "round_selected", "target_scale", "clip_norm",
    "noise_multiplier", "evaluation_split", "evaluation_samples", "rmse_t_ha",
    "mae_t_ha", "r2", "accuracy", "macro_f1", "training_time_s_measured",
    "communication_bytes_measured", "checkpoint_path", "result_path",
    "result_sha256", "completion_sha256", "receipt_assets_verified",
    "source_scope",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_run(run_dir: Path) -> tuple[dict, str, str]:
    result_path = run_dir / "result.json"
    completion_path = run_dir / "completion.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    receipt = json.loads(completion_path.read_text(encoding="utf-8"))
    if result.get("identity") != receipt.get("identity"):
        raise ValueError(f"result/receipt identity mismatch: {run_dir}")
    assets = receipt.get("assets")
    if not isinstance(assets, dict) or "result.json" not in assets:
        raise ValueError(f"missing receipt result asset: {run_dir}")
    for name, expected in assets.items():
        if Path(name).name != name or sha(run_dir / name) != expected:
            raise ValueError(f"receipt asset mismatch: {run_dir / name}")
    return result, sha(result_path), sha(completion_path)


def _fold_info(partitions: dict, fold: str) -> dict:
    if fold not in partitions:
        raise ValueError(f"unknown fold: {fold}")
    return partitions[fold]


def _best_validation(history: list[dict]) -> tuple[dict, int]:
    if not history:
        raise ValueError("empty control-search history")
    row = min(history, key=lambda r: r["validation"]["rmse"])
    return row["validation"], row["round"]


def record(stage_dir: str, result: dict, run_dir: Path, result_hash: str,
           completion_hash: str, partitions: dict, search_config: dict) -> dict:
    row = {field: "" for field in FIELDS}
    run = result.get("run", result)
    fold = run["fold"]
    info = _fold_info(partitions, fold)
    method = run.get("algorithm", run.get("method", ""))
    variant = run.get("variant", "")
    if isinstance(variant, dict):
        variant = variant.get("name", "")
    stage = STAGES[stage_dir][1]
    metric = result.get("test")
    split = "outer_test"
    selected_round = ""
    if stage == "inner_validation":
        metric = result["validation"]
        split = "inner_validation"
        selected_round = result.get("best_round") or ""
        if method in ("centralized", "fedavg", "fedprox", "fednova"):
            learning_rate = search_config["learning_rates"][result["candidate"]]
            rounds = search_config["rounds"]
            batch_size = search_config["batch_size"]
            local_epochs = search_config["local_epochs"]
        else:
            reference = search_config["references"][result["candidate"]]
            if reference["model"] != method:
                raise ValueError(f"reference config mismatch: {run_dir}")
            learning_rate = reference.get("learning_rate", "")
            rounds = reference.get("rounds", reference.get("trees", ""))
            batch_size = local_epochs = ""
    elif stage == "inner_validation_fixed_public":
        method = "fixed_public_equal_client_control"
        metric, selected_round = _best_validation(result["history"])
        split = "inner_validation"
        learning_rate = run["learning_rate"]
        rounds = run["rounds"]
        batch_size = 32
        local_epochs = "four_fixed_local_steps"
    else:
        learning_rate = run.get("learning_rate", "")
        rounds = run.get("rounds", "")
        batch_size = 32 if method in ("centralized", "fedavg", "fedprox", "fednova") else ""
        local_epochs = 1 if method in ("fedavg", "fedprox", "fednova") and stage in (
            "outer_test_primary", "outer_test_post_hoc_guard") else ""
    if metric is None:
        raise ValueError(f"missing evaluation metric: {run_dir}")
    expected_n = len(info["validation"] if split == "inner_validation" else info["test"])
    if metric["samples"] != expected_n:
        raise ValueError(f"evaluation group size mismatch: {run_dir}")
    is_federated = method in ("fedavg", "fedprox", "fednova") or "public" in method or stage in (
        "outer_test_matched_noise_development", "outer_test_privacy_simulation_development")
    if stage == "outer_test_post_hoc_guard":
        is_federated = method in ("fedavg", "fedprox", "fednova")
    row.update({
        "stage": stage, "experiment_id": run["id"], "method": method,
        "variant": variant or ("zero_support_guard" if stage == "outer_test_post_hoc_guard" else ""),
        "dataset_version": "Gaudio et al. Zenodo v2.0.0",
        "dataset_doi": "10.5281/zenodo.14160885", "fold": fold,
        "outer_group": info["test_group"], "validation_group": info.get("validation_group") or "",
        "seed": run.get("seed", search_config["seed"] if stage == "inner_validation" else ""),
        "client_count": len(info["clients"]) if is_federated else "",
        "partition_strategy": "nested_leave_geographic_group_out",
        "learning_rate": learning_rate, "local_epochs": local_epochs,
        "batch_size": batch_size, "rounds_planned": rounds,
        "round_selected": selected_round, "target_scale": run.get("target_scale", ""),
        "clip_norm": run.get("clip_norm", run.get("variant", {}).get("clip_norm", "")
                                    if isinstance(run.get("variant"), dict) else ""),
        "noise_multiplier": run.get("noise_multiplier", run.get("variant", {}).get("noise_multiplier", "")
                                              if isinstance(run.get("variant"), dict) else ""),
        "evaluation_split": split, "evaluation_samples": metric["samples"],
        "rmse_t_ha": metric["rmse"], "mae_t_ha": metric["mae"],
        "r2": metric.get("r2", ""),
        "checkpoint_path": str(run_dir / "model.pt") if (run_dir / "model.pt").exists() else "",
        "result_path": str(run_dir / "result.json"), "result_sha256": result_hash,
        "completion_sha256": completion_hash, "receipt_assets_verified": True,
        "source_scope": "post_hoc_development" if "post_hoc" in stage or "development" in stage
                        else "primary_frozen" if stage == "outer_test_primary" else "training_validation",
    })
    return row


def build(results_root: Path) -> tuple[list[dict], dict]:
    data_root = results_root / "restart_regression_data_v1/folds"
    partitions = {p.parent.name: json.loads(p.read_text(encoding="utf-8"))
                  for p in data_root.glob("*/partition.json")}
    search_config = json.loads((results_root / "restart_validation_protocol_v1/config.json").read_text(encoding="utf-8"))
    rows = []
    for stage_dir, (expected, _) in STAGES.items():
        run_dirs = sorted((results_root / stage_dir / "runs").iterdir())
        if len(run_dirs) != expected or any(not p.is_dir() for p in run_dirs):
            raise ValueError(f"unexpected run count in {stage_dir}: {len(run_dirs)} vs {expected}")
        for run_dir in run_dirs:
            result, result_hash, receipt_hash = verify_run(run_dir)
            row = record(stage_dir, result, run_dir, result_hash, receipt_hash, partitions, search_config)
            if row["experiment_id"] != run_dir.name:
                raise ValueError(f"run-directory identity mismatch: {run_dir}")
            rows.append(row)
    keys = [(r["stage"], r["experiment_id"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate stage/run identity")
    summary = {"status": "PASS_COMPLETE_INDEX", "rows": len(rows),
               "by_stage": dict(Counter(r["stage"] for r in rows)),
               "evaluation_splits": dict(Counter(r["evaluation_split"] for r in rows)),
               "all_receipt_assets_verified": True,
               "not_measured_fields": ["accuracy", "macro_f1", "training_time_s_measured",
                                       "communication_bytes_measured"],
               "scope": "Run-level index of seven active regression stages, not an independent scientific replication; historical classification and smokes excluded.",
               "outer_test_exposure": "Only primary outer stage was frozen before first test inference; later stages are development/post-hoc evidence."}
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    rows, audit = build(args.results_root)
    if args.csv.exists() or args.audit.exists():
        raise FileExistsError("Refusing to overwrite an existing index")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    audit["csv_sha256"] = sha(args.csv)
    with args.audit.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
