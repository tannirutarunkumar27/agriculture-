"""Three-round validation-only diagnostic on ONE new-dataset inner fold.

No outer-test inference, tuning, KG or privacy. Not a research performance study.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from paper2.federated import aggregate, clone_state
from paper2.restart.regression_training import make_regressor, train_regression_client


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scores(y, prediction):
    if prediction.shape != y.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid regression predictions")
    error = prediction-y
    return {"validation_samples": len(y), "mae_t_ha": float(abs(error).mean()),
            "rmse_t_ha": float(np.sqrt((error**2).mean())), "negative_predictions": int((prediction < 0).sum())}


def run(data, verification, output):
    data, output = Path(data).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    checked = json.loads(Path(verification).read_text())
    manifest = json.loads((data/"manifest.json").read_text())
    if checked["status"] != "PASS" or checked["manifest_sha256"] != sha(data/"manifest.json"):
        raise ValueError("Verified dataset binding differs")
    fold_name = "outer_0_inner_1"  # fixed before any training; not a favorable-score selection
    folder = data/"folds"/fold_name
    for name, expected in manifest["assets"][fold_name].items():
        if sha(folder/name) != expected:
            raise ValueError("Inner fold artifact changed")
    fold = json.loads((folder/"partition.json").read_text())
    if fold["kind"] != "inner_selection":
        raise ValueError("Diagnostic requires an inner fold")
    with np.load(folder/"arrays.npz", allow_pickle=False) as bundle:
        if any("test" in name for name in bundle.files):
            raise ValueError("Inner diagnostic must not expose outer test tensors")
        x, y = bundle["x_train"].copy(), bundle["y_train"].copy()
        valx, valy = bundle["x_validation"].copy(), bundle["y_validation"].copy()
    config = {"scope": "ENGINEERING_DIAGNOSTIC_NOT_CONVERGED_RESEARCH", "fold": fold_name, "seed": 123,
              "rounds": 3, "epochs_per_round": 1, "hidden": 32, "batch_size": 32, "learning_rate": .001,
              "mu": .01, "participation": 1., "selection": "minimum inner-validation RMSE, first tie wins",
              "test_inference": False, "data_manifest_sha256": sha(data/"manifest.json"),
              "code_sha256": {p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name("regression_training.py"),
                               Path(__file__).resolve().parents[1]/"federated.py"]}}
    output.mkdir(parents=True, exist_ok=False)
    (output/"manifest.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    results = {"training_mean": scores(valy, np.full_like(valy, y.mean()))}
    references = {"ridge": Ridge(alpha=1.), "random_forest": RandomForestRegressor(n_estimators=100, random_state=123, n_jobs=1),
                  "xgboost": XGBRegressor(n_estimators=100, max_depth=3, learning_rate=.05, objective="reg:squarederror", random_state=123, n_jobs=1)}
    for name, model in references.items():
        model.fit(x, y)
        results[name] = scores(valy, model.predict(valx))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(123)
    initial = make_regressor(x.shape[1])
    tx, ty, vx = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(valx)
    index = {value: i for i, value in enumerate(fold["train"])}
    for algorithm in ("centralized", "fedavg", "fedprox", "fednova"):
        model = copy.deepcopy(initial)
        best, best_state, best_round, history = None, None, None, []
        for round_number in range(1, 4):
            clients = {"centralized": fold["train"]} if algorithm == "centralized" else fold["clients"]
            updates = []
            for number, (client, ids) in enumerate(sorted(clients.items())):
                local = [index[value] for value in ids]
                updates.append(train_regression_client(model, tx[local], ty[local], client, round_number,
                    fold["preprocessing_identity"], 123+1009*round_number+number,
                    mu=.01 if algorithm == "fedprox" else 0.))
            model.load_state_dict(updates[0].state if algorithm == "centralized" else aggregate(clone_state(model), updates,
                                  algorithm, round_number, fold["preprocessing_identity"], {c: len(ids) for c, ids in clients.items()}))
            model.eval()
            with torch.no_grad():
                prediction = model(vx)[:, 0].numpy()
            metric = scores(valy, prediction)
            history.append({"round": round_number, **metric, "steps": sum(u.steps for u in updates)})
            if best is None or metric["rmse_t_ha"] < best["rmse_t_ha"]:
                best, best_state, best_round = metric, clone_state(model), round_number
        torch.save(best_state, output/(algorithm+"_selected_weights.pt"))
        results[algorithm] = {**best, "selected_round": best_round, "history": history}
    (output/"validation_results.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "COMPLETE_VALIDATION_ONLY_SMOKE", "models": list(results),
                      "train_rows": len(y), "validation_rows": len(valy), "outer_test_inference": False}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--verification", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.data, args.verification, args.output)
