"""Seal/check the planned baseline experiment; never start training."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paper2.study_protocol import freeze_protocol, verify_frozen_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--verify-only",action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        if args.config is None:
            parser.error("--config required when freezing")
        freeze_protocol(args.config,args.output)
    print(json.dumps(verify_frozen_protocol(args.output),indent=2))


if __name__ == "__main__":
    main()
