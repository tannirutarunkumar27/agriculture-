"""Fresh regression preprocessing and geography-grouped nested splits.

No classification, test-based tuning, agronomic rules or privacy mechanisms.
"""
from __future__ import annotations
import hashlib
import json
import math

import numpy as np
import pandas as pd

SPECIES = ("barley", "chickpea", "fababean", "lentil", "lupin", "pea", "wheat_aestivum", "wheat_turgidum")
NUMERIC = ("soil_awc", "soil_clay", "soil_pH", "nitrogen_residual", "N_amount", "P_amount",
           "irrigation_amount", "sowing_sin", "sowing_cos")
CATEGORICAL = ("previous_crop", "crop_season", "pest_control")
FIXED = tuple("contains_"+species for species in SPECIES) + ("is_intercrop",)


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def distance_km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, [*a, *b])
    h = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 6371.0088*2*math.asin(math.sqrt(min(1., h)))


def geographic_groups(trials, radius_km=1.0):
    if not math.isfinite(radius_km) or radius_km < 0:
        raise ValueError("Invalid grouping radius")
    locations = trials[["site", "plot_lat", "plot_long"]].drop_duplicates()
    coords = locations[["plot_lat", "plot_long"]].to_numpy(float)
    if not np.isfinite(coords).all() or (abs(coords[:, 0]) > 90).any() or (abs(coords[:, 1]) > 180).any():
        raise ValueError("Invalid geographic coordinates")
    if trials.site.isna().any():
        raise ValueError("Missing source site")
    sites = sorted(trials.site.unique())
    parent = {site: site for site in sites}
    def find(site):
        while parent[site] != site:
            site = parent[site]
        return site
    pairs = []
    for i, a in enumerate(sites):
        for b in sites[i+1:]:
            left = locations.loc[locations.site.eq(a), ["plot_lat", "plot_long"]].to_numpy(float)
            right = locations.loc[locations.site.eq(b), ["plot_lat", "plot_long"]].to_numpy(float)
            separation = min(distance_km(x, y) for x in left for y in right)
            merge = separation <= radius_km
            if merge:
                parent[find(b)] = find(a)
            pairs.append({"site_a": a, "site_b": b, "minimum_distance_km": separation, "merged": merge})
    mapping = {site: "+".join(s for s in sites if find(s) == find(site)) for site in sites}
    return mapping, pairs


def build_rows(systems, trials):
    if systems.duplicated(["experiment_id", "management"]).any() or trials.experiment_id.duplicated().any():
        raise ValueError("Duplicate system/trial keys")
    merged = systems.merge(trials, on="experiment_id", how="left", validate="many_to_one", indicator=True)
    if not merged._merge.eq("both").all():
        raise ValueError("Unmatched system")
    dates = pd.to_datetime(merged.sowing_date, format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        raise ValueError("Sowing date missing/invalid")
    # Leap-year calendar phase; do not use harvest dates to derive duration.
    days = np.where(dates.dt.is_leap_year, 366, 365)
    phase = 2*np.pi*(dates.dt.dayofyear.to_numpy()-1)/days
    merged["sowing_sin"], merged["sowing_cos"] = np.sin(phase), np.cos(phase)
    for species in SPECIES:
        merged["contains_"+species] = merged.system_species.map(lambda value: int(species in value.split("+")))
    if any(not set(value.split("+")).issubset(SPECIES) for value in merged.system_species):
        raise ValueError("Unknown species outside public taxonomy")
    if not merged.crop_type.isin(["SC", "IC"]).all():
        raise ValueError("Unknown crop type")
    merged["is_intercrop"] = merged.crop_type.eq("IC").astype(int)
    expected = merged.is_intercrop+1
    if not merged[list(FIXED[:-1])].sum(axis=1).eq(expected).all():
        raise ValueError("Species/component count differs from crop type")
    merged["target_seed_dm_t_ha"] = pd.to_numeric(merged.audit_total_seed_dm_t_ha, errors="coerce")
    if not np.isfinite(merged.target_seed_dm_t_ha).all() or merged.target_seed_dm_t_ha.lt(0).any():
        raise ValueError("Invalid/incomplete target; no target imputation")
    rows = merged[["experiment_id", "management", "site", "country", "year", "system_species", "crop_type",
                   "target_seed_dm_t_ha", *NUMERIC, *CATEGORICAL, *FIXED]].copy()
    rows["row_id"] = [object_digest([a, b]) for a, b in zip(rows.experiment_id, rows.management)]
    return rows.sort_values("row_id").reset_index(drop=True)


def nested_partitions(rows):
    groups = sorted(rows.geo_group.unique())
    if len(groups) < 3 or rows.row_id.duplicated().any():
        raise ValueError("Need unique rows and at least three groups")
    folds = []
    for outer in groups:
        test = rows.index[rows.geo_group.eq(outer)].tolist()
        remaining = [group for group in groups if group != outer]
        train = rows.index[rows.geo_group.ne(outer)].tolist()
        folds.append({"id": "outer_"+str(groups.index(outer)), "kind": "outer_refit", "test_group": outer,
                      "validation_group": None, "train": train, "validation": [], "test": test})
        for validation in remaining:
            fit = rows.index[~rows.geo_group.isin([outer, validation])].tolist()
            folds.append({"id": "outer_"+str(groups.index(outer))+"_inner_"+str(groups.index(validation)),
                          "kind": "inner_selection", "test_group": outer, "validation_group": validation,
                          "train": fit, "validation": rows.index[rows.geo_group.eq(validation)].tolist(), "test": test})
    for fold in folds:
        check_partition(rows, fold)
        fold["clients"] = {group: [i for i in fold["train"] if rows.at[i, "geo_group"] == group]
                           for group in groups if group not in [fold["test_group"], fold["validation_group"]]}
    return folds


def check_partition(rows, fold):
    sets = [set(fold[name]) for name in ("train", "validation", "test")]
    if any(len(fold[name]) != len(sets[j]) for j, name in enumerate(("train", "validation", "test"))):
        raise ValueError("Duplicate partition index")
    if any(sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2))) or set.union(*sets) != set(rows.index):
        raise ValueError("Partitions overlap or fail coverage")
    for column in ("geo_group", "site", "experiment_id"):
        identities = [set(rows.loc[sorted(ids), column]) for ids in sets]
        if any(identities[a] & identities[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("Dependent group crosses partitions")
    if not sets[0] or not sets[2] or (fold["kind"] == "inner_selection" and not sets[1]):
        raise ValueError("Empty required partition")


def fit_preprocessor(rows, ids):
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Unique nonempty training ids required")
    train = rows.loc[ids]
    state = {"fit_row_ids": train.row_id.tolist(), "numeric": {}, "categorical": {}, "feature_names": []}
    for name in NUMERIC:
        values = pd.to_numeric(train[name], errors="raise").to_numpy(float)
        if np.isinf(values).any():
            raise ValueError("Infinite numeric feature")
        all_missing = bool(np.isnan(values).all())
        median = 0. if all_missing else float(np.nanmedian(values))
        filled = np.where(np.isnan(values), median, values)
        mean, scale = float(filled.mean()), float(filled.std())
        state["numeric"][name] = {"median": median, "mean": mean, "scale": scale if scale > 0 else 1.,
                                   "all_training_missing": all_missing}
        state["feature_names"] += [name, name+"__missing"]
    for name in CATEGORICAL:
        vocabulary = sorted(train[name].dropna().astype(str).unique().tolist())
        state["categorical"][name] = vocabulary
        state["feature_names"] += [name+"__missing", name+"__unknown"] + [name+"__value_"+str(i) for i in range(len(vocabulary))]
    state["feature_names"] += list(FIXED)
    state["identity"] = object_digest(state)
    return state


def transform(rows, state):
    columns = []
    for name in NUMERIC:
        values = pd.to_numeric(rows[name], errors="raise").to_numpy(float)
        if np.isinf(values).any():
            raise ValueError("Infinite numeric feature")
        missing = np.isnan(values)
        config = state["numeric"][name]
        columns.extend([((np.where(missing, config["median"], values)-config["mean"])/config["scale"]), missing.astype(float)])
    for name in CATEGORICAL:
        vocabulary = state["categorical"][name]
        values = rows[name].astype("string")
        missing = values.isna().to_numpy()
        columns.extend([missing.astype(float), ((~values.isin(vocabulary)) & values.notna()).to_numpy(float)])
        columns.extend([values.eq(value).fillna(False).to_numpy(float) for value in vocabulary])
    columns.extend(rows[name].to_numpy(float) for name in FIXED)
    result = np.column_stack(columns).astype(np.float32)
    if result.shape[1] != len(state["feature_names"]) or not np.isfinite(result).all():
        raise ValueError("Invalid transformed schema or values")
    return result


def support_flags(rows, train_ids, evaluation_ids):
    train = rows.loc[train_ids]
    known = set().union(*(set(value.split("+")) for value in train.system_species))
    systems = set(train.system_species)
    return [{"row_id": rows.at[i, "row_id"], "all_species_seen": set(rows.at[i, "system_species"].split("+")).issubset(known),
             "composition_seen": rows.at[i, "system_species"] in systems} for i in evaluation_ids]
