"""Fail-closed structural validation for the Paper 2 evidence/claim registry."""
import argparse
import json
from pathlib import Path

from paper2.restart.validation_search import ROOT, atomic_json

ALLOWED = {"SUPPORTED", "SUPPORTED_WITH_EXPOSURE_QUALIFIER", "SUPPORTED_SIMULATION_ONLY", "SUPPORTED_LIMITATION", "BLOCKED"}


def validate(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    claims = data.get("claims", [])
    if not claims or len({c.get("id") for c in claims}) != len(claims):
        raise ValueError("Claims require unique IDs")
    blocked = 0
    for claim in claims:
        if claim.get("status") not in ALLOWED or not claim.get("claim") or not claim.get("qualifier"):
            raise ValueError(f"Incomplete claim: {claim.get('id')}")
        evidence = claim.get("evidence")
        if claim["status"] == "BLOCKED":
            blocked += 1
            if evidence:
                raise ValueError("Blocked claims cannot cite enabling evidence")
        else:
            if not evidence:
                raise ValueError("Supported claims require evidence")
            for item in evidence:
                if not (ROOT / item).exists():
                    raise ValueError(f"Missing evidence: {item}")
        if "QUALIFIER" in claim["status"] and "post-hoc" not in claim["qualifier"].lower():
            raise ValueError("Exposure-qualified claim must say post-hoc")
        if claim["status"] == "SUPPORTED_SIMULATION_ONLY" and "simulation" not in claim["qualifier"].lower():
            raise ValueError("Simulation claim must retain simulation qualifier")
    if blocked < 4:
        raise ValueError("Required blocked claim classes missing")
    return {"status": "PASS", "claims": len(claims), "blocked": blocked}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, default=ROOT / "paper2/research/evidence_claim_matrix.json")
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result = validate(args.matrix); args.output.mkdir(parents=True)
    atomic_json(args.output / "verification.json", result); print(result)
