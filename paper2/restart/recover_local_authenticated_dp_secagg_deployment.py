"""Create a fail-closed receipt for an interrupted local deployment attempt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def recover(output: Path, reason: str) -> dict:
    output = Path(output).resolve()
    if (output / "receipt.json").exists():
        raise ValueError("Execution already has a receipt")
    diagnostics = output / "diagnostics"
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(diagnostics.iterdir()) if path.is_file()
    } if diagnostics.is_dir() else {}
    receipt = {
        "status": "ABORTED_LOCAL_AUTHENTICATED_DP_SECAGG_DEPLOYMENT_NO_AUTOMATIC_RETRY",
        "execution_scope": "single_laptop_wsl_loopback",
        "reason": reason,
        "tls_enabled": True,
        "cli_managed_supernode_auth": True,
        "secagg_completed": False,
        "central_dp_completed": False,
        "network_deployment_gate_passed": False,
        "privacy_claim_allowed": False,
        "diagnostic_sha256": hashes,
        "interpretation": "The attempt reached authenticated dispatch but did not complete. It is failure evidence only.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    print(recover(args.output, args.reason))
