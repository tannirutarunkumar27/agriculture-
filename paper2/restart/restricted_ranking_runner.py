"""Execute only the independently approved, restricted observed-set ranking."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paper2.restart.ranking_approval_gate import RESULTS, validate_protocol
from paper2.restart.validation_search import atomic_json, read, sha


def rank(packages: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    required_packages = {"block_id", "management", "system_species", "audit_total_seed_dm_t_ha", "candidate_scope", "ranking_approved"}
    if not required_packages <= set(packages) or not {"block_id", "gate_status"} <= set(decisions):
        raise ValueError("Ranking input schema is incomplete")
    approved = decisions.loc[decisions.gate_status.eq("APPROVED_INCLUDE"), "block_id"]
    if not len(approved) or approved.duplicated().any():
        raise ValueError("At least one uniquely approved ranking block is required")
    selected = packages.loc[packages.block_id.isin(set(approved))].copy()
    if selected.empty or not selected.candidate_scope.eq("observed_management_package_only").all() or selected.ranking_approved.astype(str).str.lower().eq("true").any():
        raise ValueError("Candidate package scope/approval binding changed")
    selected["observed_utility_t_ha"] = pd.to_numeric(selected.audit_total_seed_dm_t_ha, errors="raise")
    if selected.observed_utility_t_ha.isna().any() or not np.isfinite(selected.observed_utility_t_ha).all():
        raise ValueError("Observed utility is invalid")
    if (selected.groupby("block_id").management.nunique() < 2).any():
        raise ValueError("Every approved block requires at least two observed packages")
    selected["best_observed_utility_t_ha"] = selected.groupby("block_id").observed_utility_t_ha.transform("max")
    selected["descriptive_regret_t_ha"] = selected.best_observed_utility_t_ha - selected.observed_utility_t_ha
    selected["descriptive_rank"] = selected.groupby("block_id").observed_utility_t_ha.rank(method="min", ascending=False).astype(int)
    selected["best_observed_package"] = selected.descriptive_regret_t_ha.eq(0.0)
    selected["interpretation"] = "observed restricted-block comparison only; not causal, prospective, or universal recommendation"
    return selected.sort_values(["block_id", "descriptive_rank", "management"]).reset_index(drop=True)


def execute(output, gate=RESULTS / "restart_ranking_approval_gate_v3", review=RESULTS / "restart_ranking_experiment_review_v2"):
    output, gate, review = Path(output).resolve(), Path(gate).resolve(), Path(review).resolve()
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    report, protocol = read(gate / "ranking_gate.json"), read(review / "ranking_protocol.json")
    validate_protocol(protocol)
    if report.get("status") != "RANKING_APPROVED_FOR_RESTRICTED_OBSERVED_SET" or report.get("ranking_enabled") is not True:
        raise PermissionError("Ranking remains blocked until the independent approval gate passes")
    ranked = rank(pd.read_csv(review / "candidate_packages.csv"), pd.read_csv(gate / "block_decisions.csv", keep_default_na=False))
    output.mkdir(parents=True)
    ranked.to_csv(output / "observed_restricted_ranking.csv", index=False)
    summary = {"status": "COMPLETE_RESTRICTED_OBSERVED_SET_RANKING", "approved_blocks": int(ranked.block_id.nunique()),
               "observed_packages": int(len(ranked)), "utility": "harvest seed dry matter (t/ha)", "model_training": False,
               "model_inference": False, "claim_limit": "descriptive observed packages in approved blocks only"}
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "manifest.json", {"status": summary["status"], "ranking_gate_sha256": sha(gate / "ranking_gate.json"),
                "protocol_sha256": sha(review / "ranking_protocol.json"), "packages_sha256": sha(review / "candidate_packages.csv"),
                "decisions_sha256": sha(gate / "block_decisions.csv"), "ranking_sha256": sha(output / "observed_restricted_ranking.csv")})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--gate", type=Path, default=RESULTS / "restart_ranking_approval_gate_v3")
    parser.add_argument("--review", type=Path, default=RESULTS / "restart_ranking_experiment_review_v2")
    args = parser.parse_args(); print(execute(args.output, args.gate, args.review))
