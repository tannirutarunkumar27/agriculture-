"""Save fresh unittest execution evidence without hiding failures."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-B", "-m", "unittest", "discover", "-s", "paper2/tests", "-v"]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    (args.output / "unittest.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    (args.output / "verification.json").write_text(json.dumps({"command":command, "cwd":str(root),
                "exit_code":result.returncode, "status":"PASS" if result.returncode == 0 else "FAIL"}, indent=2), encoding="utf-8")
    print(result.stderr)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
