"""Freeze or execute bounded validation-only regression search; default dry-run."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper2.restart.validation_search import freeze, execute


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--config", default="paper2/configs/restart_validation_search_v1.json")
    parser.add_argument("--data", default="paper2/results/restart_regression_data_v1")
    parser.add_argument("--verification", default="paper2/results/restart_regression_verified_v1/verification.json")
    parser.add_argument("--frozen", default="paper2/results/restart_validation_protocol_v1")
    parser.add_argument("--output")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--max-runs", type=int, default=1)
    args = parser.parse_args()
    if args.freeze:
        if args.execute or args.resume or args.retry_failed:
            parser.error("Freeze cannot train/resume/retry")
        result = freeze(args.config, args.data, args.verification, args.frozen)
    else:
        if not args.output:
            parser.error("--output required")
        result = execute(args.frozen, args.output, args.max_runs, args.resume, args.retry_failed, not args.execute)
    print(json.dumps(result, indent=2))
