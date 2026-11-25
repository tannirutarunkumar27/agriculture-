"""Small diagnostic runs, not a benchmark or statistical research comparison."""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper2.train import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    protocols = {"canonical":"iid", "maharashtra":"district", "manipur":"district"}
    rows = []
    for dataset, strategy in protocols.items():
        path = args.partitions / f"{dataset}_{strategy}_default_seed{config['seed']}.json"
        for algorithm in ["centralized", "fedavg", "fedprox", "fednova"]:
            output = args.output / f"{dataset}_{algorithm}"
            result = run(path, output, algorithm, config)
            rows.append({"dataset":dataset, "algorithm":algorithm, "rounds":result["completed_rounds"],
                         "best_round":result["best_round"], "test_accuracy":result["test"]["accuracy"],
                         "test_macro_f1":result["test"]["macro_f1"], "test_nll":result["test"]["nll"],
                         "local_sgd_steps":result["total_local_sgd_steps"],
                         "upload_tensor_bytes":result["total_upload_payload_bytes"],
                         "download_tensor_bytes":result["total_download_payload_bytes"],
                         "round_seconds":result["training_round_seconds"]})
    with (args.output / "smoke_results.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "verification.json").open("x", encoding="utf-8") as stream:
        json.dump({"status":"PASS", "completed_runs":len(rows), "config":config,
                   "scope":"three-round diagnostic only; no convergence, novelty, superiority or privacy evidence",
                   "runs":rows}, stream, indent=2)
    print(f"Completed {len(rows)} smoke runs")


if __name__ == "__main__":
    main()
