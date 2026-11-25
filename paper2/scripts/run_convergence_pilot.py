"""Bounded, resumable-by-skipping-verified-completed-runs validation pilot."""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper2.centralized import run_reference
from paper2.data import file_hash, object_hash
from paper2.train import run, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--continue", dest="continue_run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_root = Path(__file__).resolve().parents[1]
    sources = [source_root/name for name in ["train.py","data.py","federated.py","centralized.py","metrics.py"]]
    identity = {"protocol":config, "code_hashes":{path.name:file_hash(path) for path in sources},
                "partition_hashes":{name:file_hash(args.partitions/f"{name}_{strategy}_default_seed{config['tuning_seed']}.json")
                                    for name,strategy in config["datasets"].items()}}
    identity_hash = object_hash(identity)
    if args.continue_run:
        saved = json.loads((args.output/"protocol.json").read_text(encoding="utf-8"))
        if saved["identity_hash"] != identity_hash:
            raise ValueError("Pilot code/protocol/partition changed; start a new version")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output/"protocol.json", identity | {"identity_hash":identity_hash, "test_evaluation":False})
    rows, selections = [], {}
    for dataset,strategy in config["datasets"].items():
        partition = args.partitions/f"{dataset}_{strategy}_default_seed{config['tuning_seed']}.json"
        neural_results = {}
        for index,rate in enumerate(config["learning_rates"]):
            values = []
            for algorithm in config["neural_algorithms"]:
                output = args.output/f"{dataset}_{algorithm}_candidate{index}"
                candidate_config = config["neural_base"] | {"learning_rate":rate}
                if (output/"summary.json").is_file() and args.continue_run:
                    result = json.loads((output/"summary.json").read_text(encoding="utf-8"))
                else:
                    result = run(partition, output, algorithm, candidate_config, evaluate_test=False,
                                 resume=args.continue_run and (output/"checkpoint.pt").exists())
                if result["test"] is not None or result["test_evaluated"]:
                    raise ValueError("Pilot must not use test metrics")
                if result["completed_rounds"] != candidate_config["rounds"]:
                    raise ValueError("Incomplete candidate summary")
                metrics = result["validation"]
                values.append(metrics["nll"])
                rows.append({"dataset":dataset,"kind":algorithm,"candidate":index,
                             "validation_nll":metrics["nll"],"validation_macro_f1":metrics["macro_f1"],
                             "validation_accuracy":metrics["accuracy"], "best_round":result["best_round"],
                             "parameters":json.dumps(candidate_config, sort_keys=True)})
            neural_results[index] = sum(values)/len(values)
        best_index = min(neural_results, key=lambda index:(neural_results[index],index))
        chosen = config["neural_base"] | {"learning_rate":config["learning_rates"][best_index]}
        selections[dataset] = {"common_neural_config":chosen, "mean_validation_nll_by_candidate":neural_results,
                               "selected_neural_candidate":best_index,"conventional":{}}
        for kind,candidates in config["centralized_candidates"].items():
            metrics_by_candidate = {}
            for index,parameters in enumerate(candidates):
                output = args.output/f"{dataset}_{kind}_candidate{index}"
                if (output/"summary.json").is_file() and args.continue_run:
                    result = json.loads((output/"summary.json").read_text(encoding="utf-8"))
                else:
                    result = run_reference(partition, output, kind, parameters, seed=config["tuning_seed"], evaluate_test=False)
                if result["test_evaluated"] or result["test"] is not None:
                    raise ValueError("Pilot must not use test metrics")
                metrics = result["validation"]
                metrics_by_candidate[index] = metrics["nll"]
                rows.append({"dataset":dataset,"kind":kind,"candidate":index,
                             "validation_nll":metrics["nll"],"validation_macro_f1":metrics["macro_f1"],
                             "validation_accuracy":metrics["accuracy"],"best_round":"",
                             "parameters":json.dumps(parameters,sort_keys=True)})
            index = min(metrics_by_candidate, key=lambda i:(metrics_by_candidate[i],i))
            selections[dataset]["conventional"][kind] = {"parameters":candidates[index], "candidate":index,
                                                       "validation_nll":metrics_by_candidate[index]}
        write_json(args.output/"selected_configs.json", {"identity_hash":identity_hash, "test_evaluated":False,
                                                         "selections":selections})
    with (args.output/"validation_results.csv").open("w",encoding="utf-8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output/"verification.json", {"status":"PASS","completed_candidates":len(rows),
                                               "test_evaluated":False,"identity_hash":identity_hash})
    print(f"Completed {len(rows)} validation-only candidates",flush=True)


if __name__ == "__main__":
    main()
