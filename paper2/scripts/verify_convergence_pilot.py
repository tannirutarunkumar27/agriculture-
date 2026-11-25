"""Recompute candidate validation metrics from saved models; never access test rows for evaluation."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import joblib
import numpy as np
import torch

from paper2.data import load_dataset, file_hash, prepare_arrays
from paper2.federated import make_model
from paper2.metrics import probability_metrics
from paper2.train import evaluate, write_json


def require(condition,message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot",type=Path,required=True)
    parser.add_argument("--partitions",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    verification = json.loads((args.pilot/"verification.json").read_text())
    require(verification["status"] == "PASS" and not verification["test_evaluated"],"Pilot incomplete or test-evaluated")
    protocol = json.loads((args.pilot/"protocol.json").read_text())
    selections = json.loads((args.pilot/"selected_configs.json").read_text())["selections"]
    torch.set_num_threads(1)
    validated,selected_rows,regional_rows = [],[],[]
    config = protocol["protocol"]
    for dataset,strategy in config["datasets"].items():
        partition_path = args.partitions/f"{dataset}_{strategy}_default_seed{config['tuning_seed']}.json"
        require(file_hash(partition_path) == protocol["partition_hashes"][dataset],"Partition hash changed")
        partition = json.loads(partition_path.read_text())
        frame,schema,source = load_dataset(dataset)
        require(file_hash(source) == partition["source_sha256"],"Source hash changed")
        x,y,_ = prepare_arrays(frame,schema,partition)
        validation = partition["splits"]["validation"]
        configs = [(algorithm,index) for algorithm in config["neural_algorithms"] for index in range(len(config["learning_rates"]))]
        configs += [(kind,index) for kind,values in config["centralized_candidates"].items() for index in range(len(values))]
        for kind,index in configs:
            folder = args.pilot/f"{dataset}_{kind}_candidate{index}"
            summary = json.loads((folder/"summary.json").read_text())
            manifest = json.loads((folder/"manifest.json").read_text())
            require(summary["test"] is None and not summary["test_evaluated"],"Candidate accessed test")
            require(not (folder/"test_predictions.npz").exists(),"Pilot saved test predictions")
            require(manifest["source_sha256"] == file_hash(source),"Candidate source identity mismatch")
            require(manifest["partition_sha256"] == file_hash(partition_path),"Candidate partition identity mismatch")
            source_root = Path(__file__).resolve().parents[1]
            require(all(file_hash(source_root/name)==digest for name,digest in manifest["code_hashes"].items()),
                    "Candidate implementation hashes changed")
            neural = kind in config["neural_algorithms"]
            if neural:
                checkpoint = torch.load(folder/"checkpoint.pt",map_location="cpu",weights_only=True)
                require(checkpoint["identity"]==manifest["identity"],"Checkpoint identity mismatch")
                require(checkpoint["completed_round"]==config["neural_base"]["rounds"],"Incomplete checkpoint")
                model = make_model(x.shape[1],len(partition["classes"]),manifest["config"]["hidden"])
                model.load_state_dict(checkpoint["best_state"])
                model.eval()
                recomputed,probability = evaluate(model,torch.from_numpy(x[validation]),torch.from_numpy(y[validation]),partition["classes"])
                best_expected = min(checkpoint["records"],key=lambda record:record["validation"]["nll"])
                require(best_expected["round"]==checkpoint["best_round"],"Checkpoint selection mismatch")
                selected = index==selections[dataset]["selected_neural_candidate"]
            else:
                require(file_hash(folder/"model.joblib")==manifest["model_sha256"],"Model file changed")
                model = joblib.load(folder/"model.joblib")  # generated locally; never load untrusted pickles
                probability = model.predict_proba(x[validation])
                recomputed = probability_metrics(y[validation],probability,partition["classes"])
                selected = index==selections[dataset]["conventional"][kind]["candidate"]
            for metric in ["accuracy","macro_f1","macro_precision","macro_recall","weighted_f1","nll"]:
                require(abs(recomputed[metric]-summary["validation"][metric]) < 1e-6,f"Recomputed {metric} differs")
            validated.append({"dataset":dataset,"kind":kind,"candidate":index,"validation_nll":recomputed["nll"]})
            if selected:
                selected_rows.append({"dataset":dataset,"kind":kind,"candidate":index,
                                      "validation_accuracy":recomputed["accuracy"],"validation_macro_f1":recomputed["macro_f1"],
                                      "validation_nll":recomputed["nll"],"best_round":summary.get("best_round","")})
                if "region" in schema:
                    regions = frame.iloc[validation][schema["region"]].astype(str).to_numpy()
                    for region in sorted(set(regions)):
                        selected_ids = np.flatnonzero(regions==region)
                        metrics = probability_metrics(y[validation][selected_ids],probability[selected_ids],partition["classes"])
                        regional_rows.append({"dataset":dataset,"kind":kind,"region":region,
                                              "validation_samples":len(selected_ids),
                                              "present_true_crop_classes":len(np.unique(y[validation][selected_ids])),
                                              "macro_f1_global_schema":metrics["macro_f1"],
                                              "accuracy":metrics["accuracy"],"nll":metrics["nll"]})
    require(len(validated)==verification["completed_candidates"],"Candidate count mismatch")
    args.output.mkdir(parents=True)
    for name,rows in [("selected_validation_metrics.csv",selected_rows),("regional_validation_metrics.csv",regional_rows)]:
        with (args.output/name).open("x",encoding="utf-8",newline="") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(args.output/"verification.json",{"status":"PASS","validated_candidates":len(validated),
                                              "test_evaluated":False,"validation_recomputation":validated,
                                              "regional_metric_scope":"same-district validation, all global classes included in macro-F1; absent classes count zero"})
    print(f"Verified {len(validated)} candidate models against saved validation metrics; no test evaluation")


if __name__ == "__main__":
    main()
