"""Verify every file in a prepared remote bundle before execution."""
from __future__ import annotations

import argparse
from pathlib import Path

from paper2.restart.validation_search import read, sha


def verify(manifest_path: Path) -> dict:
    manifest_path = Path(manifest_path).resolve(); root = manifest_path.parent
    manifest = read(manifest_path)
    allowed = {"RR06_REMOTE_BUNDLE_READY_NO_TRAINING", "DP_SECAGG_REMOTE_BUNDLE_READY_NO_TRAINING",
               "RR07_REMOTE_GPU_BUNDLE_READY_NO_OUTER_INFERENCE"}
    if manifest.get("status") not in allowed or not isinstance(manifest.get("files"), dict):
        raise ValueError("Unexpected remote bundle manifest")
    errors = []
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file(): errors.append(f"missing:{relative}")
        elif sha(path) != expected: errors.append(f"hash:{relative}")
    if errors: raise ValueError("Remote bundle verification failed: " + ",".join(errors[:10]))
    return {"status": "PASS_REMOTE_BUNDLE", "bundle_status": manifest["status"], "verified_files": len(manifest["files"]),
            "bytes": manifest["bytes"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(); print(verify(args.manifest))
