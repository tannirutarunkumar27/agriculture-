"""Audit completed search results against selected local models; validation ONLY."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import joblib
import numpy as np
import torch
from paper2.restart.validation_search import (verify_frozen, read, identity, writer_lock,
    load_inner, verify_completed, metrics, sha, make_regressor)


def verify(frozen, study, output):
    frozen, study, output = Path(frozen), Path(study), Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest, config, plan = verify_frozen(frozen)
    binding = identity(manifest)
    if read(study/"binding.json")["identity"] != binding:
        raise ValueError("Study binding differs")
    prepared = read(Path(manifest["data"])/"manifest.json")
    verified, pending, receipts = [], [], {}
    torch.set_num_threads(1)
    with writer_lock(study):
        for run in plan:
            folder = study/"runs"/run["id"]
            if not (folder/"completion.json").exists():
                pending.append(run["id"])
                continue
            result = verify_completed(folder, identity({"study": binding, "run": run}))
            fold, arrays = load_inner(manifest["data"], prepared, run)
            x, y = arrays["x_validation"], arrays["y_validation"]
            if run["algorithm"] in config["algorithms"]:
                model = make_regressor(x.shape[1], config["hidden"])
                model.load_state_dict(torch.load(folder/"model.pt", map_location="cpu", weights_only=True))
                model.eval()
                with torch.no_grad():
                    prediction = model(torch.from_numpy(x))[:, 0].numpy()
                history = result["history"]
                if len(history) != config["rounds"] or [row["round"] for row in history] != list(range(1, config["rounds"]+1)):
                    raise ValueError("Training budget/history differs")
                winner = min(history, key=lambda row: row["validation"]["rmse"])
                if winner["round"] != result["best_round"] or winner["validation"] != result["validation"]:
                    raise ValueError("Selected checkpoint differs from validation rule")
            else:
                model = joblib.load(folder/"model.joblib")  # only hashed locally generated model
                prediction = np.full(y.shape, model["training_mean"], dtype=np.float64) if run["algorithm"] == "training_mean" else model.predict(x)
            actual = metrics(y, prediction)
            for key in ("rmse", "mae"):
                if not np.isclose(actual[key], result["validation"][key], rtol=1e-7, atol=1e-7):
                    raise ValueError("Selected model does not reproduce validation metric")
            if actual["negative_predictions"] != result["validation"]["negative_predictions"]:
                raise ValueError("Negative prediction count differs")
            verified.append(run["id"])
            receipts[run["id"]] = sha(folder/"completion.json")
    report = {"status": "PASS_COMPLETE" if not pending else "PASS_PARTIAL", "verified_completed_runs": len(verified),
              "planned_runs": len(plan), "pending_runs": len(pending), "completion_sha256": receipts,
              "model_validation_reinference": True, "outer_test_inference": False, "metric_tolerance": "rtol=atol=1e-7",
              "frozen_manifest_sha256": sha(frozen/"manifest.json"), "verifier_sha256": sha(__file__)}
    output.mkdir(parents=True, exist_ok=False)
    (output/"verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "verified_completed_runs", "planned_runs", "pending_runs", "outer_test_inference")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", default="paper2/results/restart_validation_protocol_v1")
    parser.add_argument("--study", default="paper2/results/restart_validation_search_v1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    verify(args.frozen, args.study, args.output)
