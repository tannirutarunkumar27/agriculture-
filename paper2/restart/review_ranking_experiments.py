"""Build a fail-closed review packet for restricted observed-set ranking.

This module does not rank crops or infer causal comparability.  It binds the
mechanically matched blocks to their experiment references, complete treatment
packages, outcomes, and the independent approvals still required before any
ranking analysis can run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "paper2/results"
RAW = ROOT / "KG-Hybrid/data/raw"
FEASIBILITY = RESULTS / "restart_feasibility_v1"
COVARIATES = RESULTS / "restart_ranking_covariates_v2"
MATCH_FIELDS = [
    "experiment_id", "crop_season", "sowing_date", "N_amount", "P_amount",
    "irrigation_amount", "pest_control",
]
PACKAGE_FIELDS = [
    "species", "cultivar", "density_level", "density_relative", "density_factor",
    "interrow", "mixing_pattern", "mixture_design",
]
APPROVAL_FIELDS = [
    "original_randomization_confirmed",
    "treatment_comparability_approved",
    "harvest_comparability_approved",
    "replicate_and_tie_policy_approved",
    "restricted_utility_approved",
]
DOI_RE = re.compile(r"https?://doi\.org/[^\s]+", re.IGNORECASE)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _required(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {','.join(missing)}")


def _text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _canonical(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return format(float(value), ".17g")
    return str(value)


def _block_id(row: pd.Series) -> str:
    payload = {field: _canonical(row[field]) for field in MATCH_FIELDS}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return f"RB-{digest[:16]}"


def normalize_references(references: pd.DataFrame, experiment_ids: set[str]) -> pd.DataFrame:
    """Validate the supplied reference registry and extract DOI URLs."""
    _required(references, {"experiment_id", "reference linked to original experiments"}, "references")
    refs = references[["experiment_id", "reference linked to original experiments"]].copy()
    refs.columns = ["experiment_id", "reference_text"]
    refs["experiment_id"] = refs["experiment_id"].map(_text)
    if (refs.experiment_id == "").any() or refs.experiment_id.duplicated().any():
        raise ValueError("Reference experiment identifiers must be nonblank and unique")
    if set(refs.experiment_id) != set(experiment_ids):
        missing = sorted(set(experiment_ids) - set(refs.experiment_id))
        extra = sorted(set(refs.experiment_id) - set(experiment_ids))
        raise ValueError(f"Reference/trial experiment mismatch; missing={missing}, extra={extra}")
    refs["reference_text"] = refs["reference_text"].map(_text)
    refs["doi_urls"] = refs.reference_text.map(lambda value: sorted(set(DOI_RE.findall(value))))
    refs["reference_present"] = refs.reference_text.ne("")
    malformed = refs.reference_present & refs.doi_urls.map(len).eq(0)
    if malformed.any():
        raise ValueError("Nonblank reference cells must contain at least one DOI URL")
    refs["doi_count"] = refs.doi_urls.map(len)
    refs["doi_urls"] = refs.doi_urls.map(lambda values: ";".join(values))
    return refs.sort_values("experiment_id").reset_index(drop=True)


def _component_packages(components: pd.DataFrame) -> pd.DataFrame:
    _required(components, {"experiment_id", "management"} | set(PACKAGE_FIELDS), "components")
    if components.duplicated(["experiment_id", "management", "species"]).any():
        raise ValueError("Component package keys must be unique")

    def join_field(group: pd.DataFrame, field: str) -> str:
        pairs = [f"{_text(row.species)}:{_text(getattr(row, field))}" for row in group.itertuples(index=False)]
        return "|".join(sorted(pairs))

    rows = []
    for (experiment_id, management), group in components.groupby(["experiment_id", "management"], sort=True):
        row = {"experiment_id": experiment_id, "management": management}
        for field in PACKAGE_FIELDS:
            row[f"component_{field}"] = join_field(group, field)
        row["component_count"] = int(len(group))
        rows.append(row)
    return pd.DataFrame(rows)


def review_frames(
    trials: pd.DataFrame,
    references: pd.DataFrame,
    blocks: pd.DataFrame,
    systems: pd.DataFrame,
    components: pd.DataFrame,
) -> dict[str, object]:
    """Return review tables and a summary without granting any approvals."""
    _required(trials, {"experiment_id", "site", "country", "year"}, "trials")
    _required(blocks, set(MATCH_FIELDS) | {"systems", "alternative_count", "management_count"}, "blocks")
    _required(systems, set(MATCH_FIELDS) | {
        "management", "crop_type", "system_species", "audit_total_seed_dm_t_ha",
        "component_harvest_dates", "site", "country", "year",
    }, "systems")
    if trials.experiment_id.isna().any() or trials.experiment_id.duplicated().any():
        raise ValueError("Trials must have unique nonblank experiment identifiers")
    if blocks.duplicated(MATCH_FIELDS).any():
        raise ValueError("Candidate block matching keys must be unique")
    if systems.duplicated(["experiment_id", "management"]).any():
        raise ValueError("System treatment keys must be unique")
    if systems.audit_total_seed_dm_t_ha.isna().any():
        raise ValueError("Ranking review cannot bind systems with missing outcomes")

    refs = normalize_references(references, set(trials.experiment_id.astype(str)))
    block_keys = blocks.copy()
    block_keys["block_id"] = block_keys.apply(_block_id, axis=1)
    if block_keys.block_id.duplicated().any():
        raise ValueError("Candidate block identities collide")
    selected = systems.merge(
        block_keys[MATCH_FIELDS + ["block_id"]], on=MATCH_FIELDS, how="inner", validate="many_to_many"
    )
    packages = _component_packages(components)
    selected = selected.merge(packages, on=["experiment_id", "management"], how="left", validate="one_to_one")
    if selected.component_count.isna().any():
        raise ValueError("Candidate systems are missing component package data")

    block_rows = []
    for block in block_keys.itertuples(index=False):
        group = selected.loc[selected.block_id.eq(block.block_id)].copy()
        expected_systems = set(_text(block.systems).split(";"))
        if len(group) != int(block.management_count):
            raise ValueError(f"Block {block.block_id} management count changed")
        if set(group.system_species) != expected_systems or group.system_species.nunique() != int(block.alternative_count):
            raise ValueError(f"Block {block.block_id} system alternatives changed")
        reference = refs.loc[refs.experiment_id.eq(block.experiment_id)].iloc[0]
        package_columns = [f"component_{field}" for field in PACKAGE_FIELDS]
        group["package_signature"] = group[["crop_type", "system_species"] + package_columns].astype(str).agg("||".join, axis=1)
        duplicate_system_packages = int(group.groupby("system_species").size().gt(1).sum())
        blocking = [
            "original_randomization_not_independently_confirmed",
            "treatment_comparability_not_approved",
            "harvest_comparability_not_approved",
            "replicate_and_tie_policy_not_approved",
            "restricted_utility_not_approved",
        ]
        if not bool(reference.reference_present):
            blocking.insert(0, "publication_reference_missing")
        block_rows.append({
            "block_id": block.block_id,
            **{field: getattr(block, field) for field in MATCH_FIELDS},
            "doi_urls": reference.doi_urls,
            "reference_present": bool(reference.reference_present),
            "system_alternative_count": int(group.system_species.nunique()),
            "candidate_package_count": int(len(group)),
            "system_species": ";".join(sorted(group.system_species.unique())),
            "management_ids": ";".join(sorted(group.management.astype(str))),
            "distinct_harvest_date_signatures": int(group.component_harvest_dates.nunique()),
            "system_species_with_multiple_packages": duplicate_system_packages,
            "all_component_packages_bound": True,
            "mechanical_management_match_verified": True,
            "ranking_approved": False,
            "blocking_reasons": ";".join(blocking),
        })
    block_review = pd.DataFrame(block_rows).sort_values(["experiment_id", "block_id"]).reset_index(drop=True)

    selected = selected.merge(
        refs[["experiment_id", "reference_present", "doi_urls"]], on="experiment_id", validate="many_to_one"
    ).sort_values(["block_id", "system_species", "management"]).reset_index(drop=True)
    selected["ranking_approved"] = False
    selected["candidate_scope"] = "observed_management_package_only"

    experiment_review = trials[["experiment_id", "site", "country", "year"]].merge(
        refs, on="experiment_id", validate="one_to_one"
    )
    block_counts = block_review.groupby("experiment_id").agg(
        candidate_blocks=("block_id", "size"),
        candidate_packages=("candidate_package_count", "sum"),
    ).reset_index()
    experiment_review = experiment_review.merge(block_counts, on="experiment_id", how="left")
    experiment_review[["candidate_blocks", "candidate_packages"]] = experiment_review[
        ["candidate_blocks", "candidate_packages"]
    ].fillna(0).astype(int)
    experiment_review["independent_design_review_complete"] = False
    experiment_review["ranking_approved"] = False
    experiment_review["status"] = np.where(
        ~experiment_review.reference_present,
        "MISSING_PUBLICATION_REFERENCE_AND_DESIGN_REVIEW",
        np.where(
            experiment_review.candidate_blocks.gt(0),
            "REFERENCE_AVAILABLE_DESIGN_REVIEW_REQUIRED",
            "REFERENCE_AVAILABLE_NO_MATCHED_RANKING_BLOCK",
        ),
    )
    experiment_review = experiment_review.sort_values("experiment_id").reset_index(drop=True)

    approval = block_review[["block_id", "experiment_id"]].copy()
    approval["reviewer"] = ""
    approval["reviewer_affiliation"] = ""
    approval["review_date"] = ""
    approval["source_check_date"] = ""
    approval["reviewer_independent"] = ""
    approval["conflict_of_interest_disclosed"] = ""
    approval["source_locator"] = ""
    for field in APPROVAL_FIELDS:
        approval[field] = ""
    approval["decision"] = ""
    approval["rationale"] = ""

    doi_rows = []
    for row in refs.itertuples(index=False):
        for doi in (_text(row.doi_urls).split(";") if row.doi_urls else []):
            doi_rows.append({"doi_url": doi, "experiment_id": row.experiment_id})
    doi_inventory = pd.DataFrame(doi_rows, columns=["doi_url", "experiment_id"])
    if len(doi_inventory):
        doi_inventory = doi_inventory.groupby("doi_url").agg(
            experiments=("experiment_id", lambda values: ";".join(sorted(values))),
            experiment_count=("experiment_id", "nunique"),
        ).reset_index().sort_values("doi_url")

    candidate_experiments = set(block_review.experiment_id)
    missing_candidate_experiments = set(
        block_review.loc[~block_review.reference_present, "experiment_id"]
    )
    summary = {
        "status": "RANKING_REVIEW_PACKET_READY_MANUAL_APPROVAL_REQUIRED",
        "experiments": int(len(experiment_review)),
        "experiments_with_publication_reference": int(experiment_review.reference_present.sum()),
        "experiments_missing_publication_reference": int((~experiment_review.reference_present).sum()),
        "unique_doi_urls": int(len(doi_inventory)),
        "candidate_blocks": int(len(block_review)),
        "candidate_experiments": int(len(candidate_experiments)),
        "candidate_blocks_with_publication_reference": int(block_review.reference_present.sum()),
        "candidate_blocks_missing_publication_reference": int((~block_review.reference_present).sum()),
        "candidate_experiments_missing_publication_reference": int(len(missing_candidate_experiments)),
        "candidate_packages": int(len(selected)),
        "blocks_with_multiple_packages_for_a_system_species": int(
            block_review.system_species_with_multiple_packages.gt(0).sum()
        ),
        "blocks_with_multiple_harvest_date_signatures": int(
            block_review.distinct_harvest_date_signatures.gt(1).sum()
        ),
        "blocks_approved_for_ranking": 0,
        "ranking_enabled": False,
        "model_training": False,
        "interpretation": (
            "Publication linkage and package binding are mechanical checks only. Original randomization, "
            "treatment and harvest comparability, replicate/tie handling, and restricted utility require "
            "independent approval for every included block."
        ),
    }
    return {
        "summary": summary,
        "experiment_review": experiment_review,
        "block_review": block_review,
        "candidate_packages": selected,
        "approval_template": approval,
        "doi_inventory": doi_inventory,
    }


def ranking_protocol() -> dict:
    """Prospective interpretation contract; activation still requires approvals."""
    return {
        "version": "restricted-observed-ranking-review-v1",
        "status": "FROZEN_REVIEW_CONTRACT_NOT_APPROVED_FOR_RANKING",
        "candidate_unit": "observed management package within one reviewed matched block",
        "candidate_scope": "only packages actually tested in the source experiment",
        "utility": {"measure": "harvest seed dry matter", "unit": "t/ha", "economic": False, "nutritional": False},
        "descriptive_regret": "max observed approved-package utility in block minus selected approved-package utility",
        "ties": "exactly equal stored utilities share best status and have zero descriptive regret",
        "replicates": "do not average management packages as replicates without independent design confirmation",
        "required_block_approvals": APPROVAL_FIELDS,
        "prohibited_claims": [
            "causal treatment effect", "universal best crop", "unobserved counterfactual performance",
            "profitability", "nutritional benefit", "prospective field utility",
        ],
        "activation": "A separate audited gate must validate signed block decisions before labels or scores are created.",
    }


def prepare(
    output: Path,
    trials_path: Path = RAW / "data_trials.csv",
    references_path: Path = RAW / "references.xlsx",
    blocks_path: Path = FEASIBILITY / "management_matched_candidates.csv",
    systems_path: Path = FEASIBILITY / "system_outcomes_audit.csv",
    components_path: Path = COVARIATES / "component_covariates.csv",
) -> dict:
    output = Path(output).resolve()
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    paths = [Path(p).resolve() for p in (trials_path, references_path, blocks_path, systems_path, components_path)]
    frames = review_frames(
        pd.read_csv(paths[0]), pd.read_excel(paths[1]), pd.read_csv(paths[2]),
        pd.read_csv(paths[3]), pd.read_csv(paths[4]),
    )
    output.mkdir(parents=True)
    for name in ("experiment_review", "block_review", "candidate_packages", "approval_template", "doi_inventory"):
        frames[name].to_csv(output / f"{name}.csv", index=False)
    atomic_json(output / "summary.json", frames["summary"])
    atomic_json(output / "ranking_protocol.json", ranking_protocol())
    atomic_json(output / "manifest.json", {
        "status": frames["summary"]["status"],
        "ranking_enabled": False,
        "model_training": False,
        "source_sha256": {path.name: sha(path) for path in paths},
        "code_sha256": sha(Path(__file__)),
        "outputs_sha256": {
            name: sha(output / name) for name in [
                "experiment_review.csv", "block_review.csv", "candidate_packages.csv",
                "approval_template.csv", "doi_inventory.csv", "summary.json", "ranking_protocol.json",
            ]
        },
    })
    return frames["summary"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.output))
