"""CPU public-data simulation runner with validation selection and resumable rounds."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import torch

from paper2.data import load_dataset, file_hash, object_hash, prepare_arrays
from paper2.federated import aggregate, clone_state, make_model, state_bytes, train_client

DEFAULTS = {"seed": 42, "rounds": 3, "local_epochs": 1, "batch_size": 32,
            "learning_rate": 0.05, "mu": 0.01, "hidden": 32,
            "participation": 1.0, "threads": 1}


def checked_config(config):
    if set(config) - set(DEFAULTS):
        raise ValueError(f"Unknown configuration keys: {set(config)-set(DEFAULTS)}")
    result = DEFAULTS | config
    for key in ["seed", "rounds", "local_epochs", "batch_size", "hidden", "threads"]:
        if type(result[key]) is not int or result[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid {key}")
    for key in ["learning_rate", "mu", "participation"]:
        if not isinstance(result[key], (int,float)) or isinstance(result[key], bool) or not math.isfinite(result[key]):
            raise ValueError(f"Invalid {key}")
    if result["learning_rate"] <= 0 or result["mu"] < 0 or not 0 < result["participation"] <= 1:
        raise ValueError("Invalid learning rate, proximal coefficient or participation")
    return result


def evaluate(model, x, y, classes, detailed=False):
    if not len(y):
        raise ValueError("Empty evaluation split")
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(batch) for batch in x.split(1024)])
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite evaluation logits")
        probability = logits.softmax(dim=1).numpy()
        loss = float(torch.nn.functional.cross_entropy(logits, y))
    prediction = probability.argmax(axis=1)
    labels = list(range(len(classes)))
    precision, recall, f1, _ = precision_recall_fscore_support(y.numpy(), prediction,
                                                            labels=labels, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y.numpy(), prediction, labels=labels,
                                              average="weighted", zero_division=0)[2]
    result = {"samples": len(y), "accuracy": float(accuracy_score(y.numpy(), prediction)),
              "macro_precision": float(precision), "macro_recall": float(recall),
              "macro_f1": float(f1), "weighted_f1": float(weighted), "nll": loss}
    if detailed:
        result["per_class"] = classification_report(y.numpy(), prediction, labels=labels,
                                                   target_names=classes, output_dict=True, zero_division=0)
        result["confusion_matrix"] = confusion_matrix(y.numpy(), prediction, labels=labels).tolist()
    return result, probability


def _replace_portably(temporary, path):
    # Windows sandbox/preview handles can transiently deny an otherwise valid
    # replace.  Preserve atomic replacement where it is available and use a
    # portable overwrite only after bounded retries.
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


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    _replace_portably(temporary, path)


def run(partition_path, output, algorithm="fedavg", config=None, resume=False, evaluate_test=True):
    if algorithm not in {"centralized", "fedavg", "fedprox", "fednova"}:
        raise ValueError("Unknown algorithm")
    config = checked_config(config or {})
    if type(evaluate_test) is not bool:
        raise ValueError("evaluate_test must be explicitly boolean")
    partition_path, output = Path(partition_path).resolve(), Path(output).resolve()
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    frame, schema, source = load_dataset(partition["dataset"])
    if file_hash(source) != partition["source_sha256"]:
        raise ValueError("Source data changed since partition creation")
    x_array, y_array, preprocessing = prepare_arrays(frame, schema, partition)
    x, y = torch.from_numpy(x_array), torch.from_numpy(y_array)
    train, validation, test = (partition["splits"][key] for key in ["train", "validation", "test"])
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config["seed"])
    model = make_model(x.shape[1], len(partition["classes"]), config["hidden"])
    code_files = [Path(__file__), Path(__file__).with_name("data.py"), Path(__file__).with_name("federated.py")]
    code_hashes = {path.name: file_hash(path) for path in code_files}
    # Rounds may be extended during a diagnostic resume; all other settings locked.
    identity = object_hash({"config": {k:v for k,v in config.items() if k != "rounds"},
                            "algorithm": algorithm, "partition": partition, "code_hashes": code_hashes,
                            "evaluate_test": evaluate_test})
    schema_hash = object_hash({"model": "mlp-v1", "hidden": config["hidden"], "preprocessing": preprocessing})
    records, completed = [], 0
    best_state, best_nll, best_round = clone_state(model), float("inf"), 0
    checkpoint_path = output / "checkpoint.pt"
    if resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["identity"] != identity:
            raise ValueError("Resume config, partition, source, or implementation identity changed")
        completed = checkpoint["completed_round"]
        if config["rounds"] <= completed:
            raise ValueError("Resume target rounds must exceed saved completed round")
        model.load_state_dict(checkpoint["model"])
        records, best_state = checkpoint["records"], checkpoint["best_state"]
        best_nll, best_round = checkpoint["best_nll"], checkpoint["best_round"]
    elif output.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {output}")
    else:
        output.mkdir(parents=True)
    git_result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, check=False)
    manifest = {"config": config, "algorithm": algorithm, "identity": identity,
                "partition_path": str(partition_path), "partition_sha256": file_hash(partition_path),
                "source_sha256": file_hash(source), "schema_hash": schema_hash, "code_hashes": code_hashes,
                "git_commit": git_result.stdout.strip() if git_result.returncode == 0 else None,
                "python": platform.python_version(), "platform": platform.platform(),
                "logical_cpus": os.cpu_count(), "cuda_available": torch.cuda.is_available(), "device": "cpu",
                "dependencies": {p: importlib.metadata.version(p) for p in ["torch", "numpy", "pandas", "scikit-learn"]},
                "preprocessing": preprocessing, "parameter_count": sum(p.numel() for p in model.parameters()),
                "model_payload_bytes": state_bytes(clone_state(model)),
                "privacy": "NONE: public-data, single-process logical-client simulation; globally fitted train-only scaler",
                "knowledge_guidance": "disabled: agronomic source/unit validation incomplete",
                "selection": "lowest global validation NLL; test evaluated only after selected checkpoint restored",
                "test_evaluation_enabled": evaluate_test,
                "fedprox_variant": "uniform sampling without replacement plus selected-client sample-weighted averaging",
                "fednova_variant": "constant shared LR plain SGD, no momentum, weight decay, proximal term, or server momentum",
                "communication": "logical float tensor payload only; excludes serialization, transport, metadata and cryptography",
                "centralized_budget": "one full-training-set pass per configured local epoch per round; not step-equivalent to federation"}
    write_json(output / "manifest.json", manifest)
    clients = partition["clients"]
    names = sorted(clients)
    for round_number in range(completed + 1, config["rounds"] + 1):
        started = time.perf_counter()
        rng = np.random.default_rng(config["seed"] + round_number * 1000003)
        selected = sorted(rng.choice(names, size=max(1, int(np.ceil(len(names)*config["participation"]))), replace=False).tolist())
        updates = []
        if algorithm == "centralized":
            selected = ["centralized_reference"]
            local_ids = {selected[0]: train}
        else:
            local_ids = {name: clients[name] for name in selected}
        for name in selected:
            index = names.index(name) if name in names else 0
            update = train_client(model, x[local_ids[name]], y[local_ids[name]], name,
                                  round_number, schema_hash, config["seed"] + round_number*1000003 + index*1009,
                                  config["local_epochs"], config["batch_size"], config["learning_rate"],
                                  config["mu"] if algorithm == "fedprox" else 0.0)
            updates.append(update)
        aggregation_start = time.perf_counter()
        if algorithm == "centralized":
            model.load_state_dict(updates[0].state)
        else:
            state = aggregate(clone_state(model), updates, algorithm, round_number, schema_hash,
                              {name: len(local_ids[name]) for name in selected})
            model.load_state_dict(state)
        aggregation_seconds = time.perf_counter() - aggregation_start
        metrics, _ = evaluate(model, x[validation], y[validation], partition["classes"])
        if metrics["nll"] < best_nll:
            best_state, best_nll, best_round = clone_state(model), metrics["nll"], round_number
        payload = state_bytes(clone_state(model))
        records.append({"round": round_number, "validation": metrics,
                        "selected_clients": selected,
                        "client_metrics": [{"client": u.client, "samples": u.samples, "actual_sgd_steps": u.steps,
                                            "mean_supervised_training_loss": u.supervised_loss,
                                            "computation_seconds": u.seconds} for u in updates],
                        "aggregation_seconds": aggregation_seconds, "round_seconds": time.perf_counter()-started,
                        "upload_payload_bytes": len(updates)*payload if algorithm != "centralized" else 0,
                        "download_payload_bytes": len(updates)*payload if algorithm != "centralized" else 0})
        checkpoint = {"identity": identity, "completed_round": round_number,
                      "model": clone_state(model), "best_state": best_state, "best_nll": best_nll,
                      "best_round": best_round, "records": records}
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        _replace_portably(temporary, checkpoint_path)
        write_json(output / "rounds.json", records)
        if round_number == 1 or round_number % 10 == 0 or round_number == config["rounds"]:
            print(f"{algorithm}: round {round_number}, validation macro-F1={metrics['macro_f1']:.4f}", flush=True)
    model.load_state_dict(best_state)
    validation_metrics, _ = evaluate(model, x[validation], y[validation], partition["classes"], detailed=True)
    test_metrics, probability = None, None
    if evaluate_test:
        test_metrics, probability = evaluate(model, x[test], y[test], partition["classes"], detailed=True)
    result = {"algorithm": algorithm, "best_round": best_round, "test": test_metrics,
              "validation": validation_metrics, "test_evaluated": evaluate_test,
              "completed_rounds": config["rounds"], "experiment_type": "diagnostic unless preregistered repeated-seed study",
              "total_upload_payload_bytes": sum(r["upload_payload_bytes"] for r in records),
              "total_download_payload_bytes": sum(r["download_payload_bytes"] for r in records),
              "total_local_sgd_steps": sum(u["actual_sgd_steps"] for r in records for u in r["client_metrics"]),
              "training_round_seconds": sum(r["round_seconds"] for r in records)}
    write_json(output / "summary.json", result)
    if evaluate_test:
        np.savez_compressed(output / "test_predictions.npz", row_ids=np.array(test), labels=y_array[test], probabilities=probability)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--algorithm", choices=["centralized", "fedavg", "fedprox", "fednova"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validation-only", action="store_true", help="Never evaluate or save test predictions during tuning")
    args = parser.parse_args()
    result = run(args.partition, args.output, args.algorithm,
                 json.loads(args.config.read_text(encoding="utf-8")), args.resume, not args.validation_only)
    print(json.dumps({"algorithm": args.algorithm, "best_round": result["best_round"],
                      "validation_nll": result["validation"]["nll"],
                      "test_accuracy": result["test"]["accuracy"] if result["test"] else None}))


if __name__ == "__main__":
    main()
