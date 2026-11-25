"""Generate configured independent-dataset partitions and client statistics."""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper2.data import load_dataset, file_hash, make_partition, object_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    rows, records = [], []
    for name in config["datasets"]:
        frame, schema, path = load_dataset(name)
        source_hash = file_hash(path)
        for seed in config["seeds"]:
            protocols = list(config["simulated"])
            if name in config["district"]:
                protocols += [{"strategy":"district", "clients":5}]
            for protocol in protocols:
                partition = make_partition(frame, schema, source_hash, name, seed, **protocol)
                suffix = str(protocol.get("alpha", "default")).replace(".", "p")
                filename = f"{name}_{protocol['strategy']}_{suffix}_seed{seed}.json"
                with (args.output / filename).open("x", encoding="utf-8") as stream:
                    json.dump(partition, stream, indent=2)
                    stream.write("\n")
                records.append({"file":filename, "sha256":file_hash(args.output / filename),
                                "dataset":name, "source_sha256":source_hash,
                                "split_hash":object_hash(partition["splits"])})
                for client, ids in partition["clients"].items():
                    rows.append({"partition":filename, "dataset":name, "strategy":protocol["strategy"],
                                 "alpha":protocol.get("alpha", ""), "seed":seed, "client":client,
                                 "training_rows":len(ids),
                                 "present_crop_classes":sum(n>0 for n in partition["client_class_counts"][client].values()),
                                 "global_train_rows":len(partition["splits"]["train"]),
                                 "global_validation_rows":len(partition["splits"]["validation"]),
                                 "global_test_rows":len(partition["splits"]["test"])})
                print(f"Saved {filename}", flush=True)
    with (args.output / "client_distributions.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump({"config":config, "partitions":records, "partition_count":len(records)}, stream, indent=2)
    print(f"Generated {len(records)} partitions; {len(rows)} client entries")


if __name__ == "__main__":
    main()
