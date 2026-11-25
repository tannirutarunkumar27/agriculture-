"""Frozen study runner; default is read-only dry-run, bounded execution explicit."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paper2.study_runner import execute_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--run-id",action="append",help="Repeat to choose frozen IDs; no settings overrides")
    parser.add_argument("--max-runs",type=int,default=1,help="Maximum new/resumed runs this invocation")
    parser.add_argument("--execute",action="store_true",help="Without this flag only verify/list the plan")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--retry-failed",action="store_true",help="Retry inspected pre-evaluation failures; ambiguous evaluation still blocks")
    args = parser.parse_args()
    report = execute_study(args.frozen,args.output,args.run_id,args.max_runs,args.resume,args.retry_failed,
                           dry_run=not args.execute)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
