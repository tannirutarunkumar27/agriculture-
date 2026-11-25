"""Compile independently approved agronomic rules into a bounded pH soft prior.

This is a feature compiler, not a crop recommender or a causal agronomy model.
It refuses unreviewed candidate rules and produces no model training or inference.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from paper2.restart.agronomic_domain_review_gate import REGISTRY, RESULTS, evaluate
from paper2.restart.paper2_completion_audit import read
from paper2.restart.review_ranking_experiments import atomic_json, sha


def soft_prior(ph, lower, upper):
    """Return one inside an approved interval and a continuous bounded penalty outside."""
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (ph, lower, upper)) or lower > upper:
        raise ValueError("Finite ordered pH interval required")
    distance = max(lower - ph, 0.0, ph - upper)
    return 1.0 / (1.0 + distance)


def compile_rules(registry, reviews):
    report, decisions = evaluate(registry, reviews)
    if report["status"] != "AGRONOMIC_DECLARATIONS_COMPLETE_INDEPENDENT_AUDIT_REQUIRED":
        raise PermissionError("Agronomic guidance remains blocked until every independent declaration is complete")
    included = set(decisions.loc[decisions.gate_status.eq("DECLARED_INCLUDE"), "rule_id"])
    rules = [rule for rule in registry["rules"] if rule["rule_id"] in included]
    if not rules or len({rule["rule_id"] for rule in rules}) != len(rules):
        raise ValueError("At least one unique approved agronomic rule is required")
    compiled = []
    for rule in sorted(rules, key=lambda value: value["rule_id"]):
        if rule.get("feature") != "soil_pH" or rule.get("transformation") != "identity; compare source optimal interval directly with dataset soil_pH":
            raise ValueError("Only unit-compatible approved soil-pH rules are supported")
        lower, upper = rule.get("original_min"), rule.get("original_max")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (lower, upper)) or lower > upper:
            raise ValueError("Approved rule interval is invalid")
        compiled.append({"rule_id": rule["rule_id"], "crop_taxon": rule["crop_taxon"], "feature": "soil_pH",
                         "optimal_min": lower, "optimal_max": upper, "prior": "1/(1+distance_to_closed_interval)",
                         "claim_limit": "soft feature only; not a causal agronomic recommendation"})
    return {"status": "COMPILED_APPROVED_AGRONOMIC_SOFT_PRIOR_NOT_MODEL_ENABLED", "rules": compiled,
            "guidance_enabled": False, "model_training": False, "model_inference": False}


def compile_from_review(output, reviews_path, registry_path=REGISTRY):
    output, reviews_path, registry_path = Path(output).resolve(), Path(reviews_path).resolve(), Path(registry_path).resolve()
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    result = compile_rules(read(registry_path), pd.read_csv(reviews_path, keep_default_na=False))
    output.mkdir(parents=True); atomic_json(output / "soft_prior_protocol.json", result)
    atomic_json(output / "manifest.json", {"status": result["status"], "registry_sha256": sha(registry_path),
                "reviews_sha256": sha(reviews_path), "protocol_sha256": sha(output / "soft_prior_protocol.json"),
                "guidance_enabled": False, "model_training": False, "model_inference": False})
    return {"status": result["status"], "compiled_rules": len(result["rules"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    args = parser.parse_args(); print(compile_from_review(args.output, args.reviews, args.registry))
