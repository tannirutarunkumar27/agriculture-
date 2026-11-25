"""Restart step 1: reproducible feasibility audit, not training or best-crop labels.

Run with the bundled Python (pandas/openpyxl/pypdf). Raw inputs never rewritten.
Derived CSVs are audit intermediates, not approved modelling tables.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader

KEY = ["experiment_id", "management", "species"]
FILES = ["data_trials.csv", "data_management.csv", "data_traits.csv", "data_climate.csv",
         "metadata.xlsx", "references.xlsx", "data_report.pdf"]
PUBLISHED_MD5 = dict(zip(FILES, ["a901bbd8964d5b3d346dc6b7389ea6d1", "064b98c6f6c2c635d5ea2c9fa0121276",
    "30a53e614cdda2e28cc6e410bab649fd", "2a507d29070cf15cd0fcd9d1ea2a4193",
    "afdfa0272724ea140616d1b9145a57d6", "3fd81168950f9f63ae5e3a39d7b4ffe9", "36e32aa41d2e47d3a9f384c4f8a3afc1"]))
MATCH_FIELDS = ["experiment_id", "crop_season", "sowing_date", "N_amount", "P_amount", "irrigation_amount", "pest_control"]


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def harvest_components(management, traits):
    """Reject ambiguous keys; match each component to its documented harvest date."""
    if management[KEY].isna().any().any() or management.duplicated(KEY).any():
        raise ValueError("Management component keys missing or duplicated")
    seed = traits.loc[traits.variable.eq("biomass_seed")].copy()
    seed["source_trait_row"] = seed.index+2
    # Do not silently trust duplicated management fields in the traits table.
    joined = seed.merge(management[KEY+["harvest_date", "crop_type"]], on=KEY, how="left",
                        validate="many_to_one", suffixes=("_trait", "_management"), indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("Seed biomass contains unmatched management components")
    if not joined.harvest_date_trait.eq(joined.harvest_date_management).all() or not joined.crop_type_trait.eq(joined.crop_type_management).all():
        raise ValueError("Traits/management disagree on harvest date or crop type")
    measured = pd.to_datetime(joined.measurement_date, format="%Y-%m-%d", errors="coerce")
    harvested = pd.to_datetime(joined.harvest_date_management, format="%Y-%m-%d", errors="coerce")
    numeric = pd.to_numeric(joined.value, errors="coerce")
    at_harvest = measured.notna() & harvested.notna() & measured.eq(harvested)
    valid = at_harvest & np.isfinite(numeric) & numeric.ge(0)
    selected = joined.loc[valid, KEY+["measurement_date", "source_trait_row"]].copy()
    selected["audit_seed_dm_t_ha"] = numeric[valid]
    if selected.duplicated(KEY).any():
        raise ValueError("Multiple at-harvest outcomes per component; no implicit averaging")
    combined = management.merge(selected, on=KEY, how="left", validate="one_to_one")
    return combined, {"seed_biomass_rows": len(seed), "dated_at_harvest": int(at_harvest.sum()),
                      "selected_component_rows": len(selected), "nonharvest_or_unknown_date": int((~at_harvest).sum()),
                      "invalid_or_negative_at_harvest": int((at_harvest & ~valid).sum())}


def system_outcomes(components):
    """Summed seed dry mass, NOT utility/profit or independent plot replication."""
    rows = []
    for (experiment, management), group in components.groupby(KEY[:2], sort=True):
        if group.crop_type.nunique() != 1:
            raise ValueError("Crop type differs within system")
        kind = group.crop_type.iloc[0]
        expected = 1 if kind == "SC" else 2 if kind == "IC" else None
        if expected is None or len(group) != expected or group.species.nunique() != expected:
            raise ValueError("Unexpected sole/intercrop component layout")
        row = {"experiment_id": experiment, "management": management, "crop_type": kind,
               "system_species": "+".join(sorted(group.species)), "expected_components": expected,
               "observed_components": int(group.audit_seed_dm_t_ha.notna().sum()),
               "audit_total_seed_dm_t_ha": float(group.audit_seed_dm_t_ha.sum()) if group.audit_seed_dm_t_ha.notna().all() else None,
               "component_harvest_dates": ";".join(sorted(group.harvest_date.unique()))}
        for key in MATCH_FIELDS[1:]:
            # A missing component management value cannot be filled from its partner.
            row[key] = group[key].iloc[0] if group[key].notna().all() and group[key].nunique() == 1 else None
        rows.append(row)
    return pd.DataFrame(rows)


def matching_candidates(systems):
    """Management-matched blocks only; cultivar/density differences still remain."""
    complete = systems.dropna(subset=MATCH_FIELDS+["audit_total_seed_dm_t_ha"])
    rows = []
    for block, group in complete.groupby(MATCH_FIELDS, dropna=False, sort=True):
        alternatives = sorted(group.system_species.unique())
        if len(alternatives) < 2:
            continue
        rows.append(dict(zip(MATCH_FIELDS, block)) | {"systems": ";".join(alternatives),
                    "alternative_count": len(alternatives), "management_count": len(group),
                    "status": "candidate_not_causally_certified"})
    return pd.DataFrame(rows, columns=MATCH_FIELDS+["systems", "alternative_count", "management_count", "status"])


def audit(raw, output):
    raw, output = Path(raw).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if raw == output or raw in output.parents or output in raw.parents:
        raise ValueError("Audit output must be separate from raw inputs")
    sources = {name: {"sha256": digest(raw/name), "md5": digest(raw/name, "md5"),
                       "published_v2_md5_match": digest(raw/name, "md5") == PUBLISHED_MD5[name]} for name in FILES}
    if not all(value["published_v2_md5_match"] for value in sources.values()):
        raise ValueError("Inputs differ from reviewed version 2; audit a separately reviewed version")
    trials, management, traits, climate = [pd.read_csv(raw/name) for name in FILES[:4]]
    metadata = pd.read_excel(raw/"metadata.xlsx")
    references = pd.read_excel(raw/"references.xlsx")
    report_text = "\n".join(p.extract_text() for p in PdfReader(raw/"data_report.pdf").pages)
    if trials.experiment_id.isna().any() or trials.experiment_id.duplicated().any():
        raise ValueError("Trials must uniquely identify experiments")
    for table in (management, traits, climate):
        if table.experiment_id.isna().any() or not set(table.experiment_id).issubset(set(trials.experiment_id)):
            raise ValueError("Orphan/missing experiment identifier")
    if climate.duplicated(["experiment_id", "date"]).any():
        raise ValueError("Ambiguous daily climate keys")
    units = metadata.loc[metadata.variable_name.eq("biomass_seed")]
    if len(units) != 1 or units.unit.iloc[0] != "t.ha-1" or "harvest" not in units.description.iloc[0]:
        raise ValueError("Seed outcome metadata changed")
    components, harvest_audit = harvest_components(management, traits)
    systems = system_outcomes(components)
    systems = systems.merge(trials[["experiment_id", "site", "country", "year"]], on="experiment_id", validate="many_to_one")
    blocks = matching_candidates(systems)
    coverage = components.merge(trials[["experiment_id", "site", "country"]], on="experiment_id", validate="many_to_one")
    species = coverage.groupby("species").agg(component_rows=("management", "size"),
                    observed_harvest_rows=("audit_seed_dm_t_ha", "count"),
                    sites=("site", "nunique"), experiments=("experiment_id", "nunique")).reset_index()
    folds = []
    for site in sorted(trials.site.unique()):
        test, train = coverage[coverage.site.eq(site)], coverage[coverage.site.ne(site)]
        known = set(train.species)
        folds.append({"held_out_site": site, "train_systems": int(systems.loc[systems.site.ne(site), "management"].count()),
                      "test_systems": int(systems.loc[systems.site.eq(site), "management"].count()),
                      "unseen_test_species": ";".join(sorted(set(test.species)-known)),
                      "status": "feasibility_only_no_split_selected"})
    missing = pd.DataFrame([{"table": name, "column": col, "missing": int(table[col].isna().sum()),
                             "rows": len(table)} for name, table in zip(FILES[:4], (trials, management, traits, climate)) for col in table])
    result = {"status": "FEASIBILITY_AUDIT_COMPLETE_NOT_APPROVED_TRAINING_PROTOCOL", "dataset_version": "2.0.0",
              "sources": sources, "audit_code_sha256": digest(__file__),
              "counts": {"experiments": len(trials), "sites": int(trials.site.nunique()), "countries": int(trials.country.nunique()),
                         "management_component_rows": len(management), "management_systems": len(systems),
                         "trait_measurements": len(traits), "climate_rows": len(climate), "metadata_rows": len(metadata),
                         "reference_rows": len(references), "species": len(species)},
              "harvest": harvest_audit, "complete_system_outcomes": int(systems.audit_total_seed_dm_t_ha.notna().sum()),
              "management_matched_candidate_blocks": len(blocks),
              "systems_in_matched_blocks": int(blocks.management_count.sum()) if len(blocks) else 0,
              "sites_by_species": species.set_index("species").sites.astype(int).to_dict(),
              "limitations": ["No ready-made best-crop labels or profit objective", "Trait rows are repeated measurements, not independent farms",
                  "Candidate matching omits cultivar/density and unmeasured confounders; not causal certification",
                  "Intercrop dry-mass sum is not economic/nutritional utility", "Only nine sites, uneven species/country coverage",
                  "Crop and management effects confounded; prices/costs and raw plot replication unavailable here",
                  "In-season realized climate is unavailable for pre-sowing recommendations",
                  "Agronomic units do not by themselves approve physiological rules", "No privacy guarantee or geographic deployment validation"],
              "training_performed": False, "split_selected": False, "best_crop_labels_created": False}
    output.mkdir(parents=True, exist_ok=False)
    for name, table in [("harvest_components_audit.csv", components), ("system_outcomes_audit.csv", systems),
                        ("management_matched_candidates.csv", blocks), ("species_coverage.csv", species),
                        ("leave_site_out_feasibility.csv", pd.DataFrame(folds)), ("missingness.csv", missing),
                        ("variable_dictionary.csv", metadata)]:
        table.to_csv(output/name, index=False)
    (output/"source_report_extracted.txt").write_text(report_text, encoding="utf-8")
    (output/"audit.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    # Verify no source mutation while reading/producing derived outputs.
    if any(digest(raw/name) != sources[name]["sha256"] for name in FILES):
        raise RuntimeError("Raw source changed during audit")
    print(json.dumps({"counts": result["counts"], "harvest": harvest_audit,
                      "complete_system_outcomes": result["complete_system_outcomes"],
                      "candidate_blocks": len(blocks), "output": str(output)}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path(__file__).resolve().parents[2]/"KG-Hybrid/data/raw")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.raw, args.output)
