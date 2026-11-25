"""Verify and summarize complete matched seed groups without model test inference."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paper2.study_analysis import summarize_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen",type=Path,required=True)
    parser.add_argument("--study",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--allow-partial",action="store_true",help="Incomplete groups will be excluded and disclosed")
    args = parser.parse_args()
    print(json.dumps(summarize_study(args.frozen,args.study,args.output,args.allow_partial),indent=2))


if __name__ == "__main__":
    main()
