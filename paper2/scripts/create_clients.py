"""Create immutable source-hashed partitions; never modify raw inputs."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper2.data import SCHEMAS, load_dataset, file_hash, make_partition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SCHEMAS, required=True)
    parser.add_argument("--strategy", choices=["iid", "dirichlet", "unequal", "district"], default="iid")
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--min-client-rows", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    frame, schema, path = load_dataset(args.dataset)
    partition = make_partition(frame, schema, file_hash(path), args.dataset, args.seed,
                               args.strategy, args.clients, args.alpha, args.min_client_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(partition, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "splits": {k: len(v) for k,v in partition['splits'].items()},
                      "clients": {k: len(v) for k,v in partition['clients'].items()}}))


if __name__ == "__main__":
    main()
