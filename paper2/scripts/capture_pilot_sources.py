"""Preserve exact manifested source versions for a completed verified pilot."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paper2.data import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    protocol = json.loads((args.pilot/"protocol.json").read_text(encoding="utf-8"))
    sources = Path(__file__).resolve().parents[1]
    allowed = {"train.py","data.py","federated.py","centralized.py","metrics.py"}
    hashes = protocol["code_hashes"]
    if set(hashes) != allowed:
        raise ValueError("Unknown source schema")
    for name,digest in hashes.items():
        if file_hash(sources/name) != digest:
            raise ValueError(f"Current source differs from recorded pilot: {name}")
    args.output.mkdir(parents=True,exist_ok=False)
    for name in hashes:
        shutil.copyfile(sources/name,args.output/name)
    for name,digest in hashes.items():
        if file_hash(args.output/name) != digest:
            raise ValueError(f"Snapshot hash mismatch: {name}")
    with (args.output/"manifest.json").open("x",encoding="utf-8") as stream:
        json.dump({"pilot_identity_hash":protocol["identity_hash"],"code_hashes":hashes,
                   "scope":"exact manifested training source snapshot; no raw datasets or credentials"},stream,indent=2)
    print("Captured and verified five manifested source files")


if __name__ == "__main__":
    main()
