"""Validated schemas and group-preserving public-data simulation partitions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "KG-Hybrid/data/raw"
SCHEMAS = {
    "canonical": {"file": "Crop_recommendation.csv", "target": "label",
                  "numeric": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"], "categorical": []},
    "maharashtra": {"file": "Crop and fertilizer dataset.csv", "target": "Crop",
                   "numeric": ["Nitrogen", "Phosphorus", "Potassium", "pH", "Rainfall", "Temperature"],
                   "categorical": ["Soil_color"], "region": "District_Name"},
    "manipur": {"file": "AgriField-Manipur_v2025.3/AgriField-Manipur_v2.csv", "target": "crop",
                "numeric": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
                "categorical": [], "region": "district", "field": "field_id"},
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_dataset(name, raw=RAW):
    if name not in SCHEMAS:
        raise ValueError(f"Unsupported dataset: {name}")
    schema = SCHEMAS[name]
    path = Path(raw) / schema["file"]
    frame = pd.read_csv(path)
    required = schema["numeric"] + schema["categorical"] + [schema["target"]]
    required += [schema[key] for key in ["region", "field"] if key in schema]
    if not set(required).issubset(frame.columns):
        raise ValueError(f"Missing schema columns: {set(required) - set(frame.columns)}")
    if frame[required].isna().any().any():
        raise ValueError("Missing required values: use an explicit imputation protocol, not silent dropping.")
    if not np.isfinite(frame[schema["numeric"]].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite numeric inputs")
    if (frame[schema["target"]].astype(str).str.strip() == "").any():
        raise ValueError("Empty crop label")
    return frame, schema, path


def group_ids(frame, schema):
    """Union repeated input profiles and, when present, all rows of a field."""
    parent = list(range(len(frame)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        parent[find(b)] = find(a)
    profiles, fields = {}, {}
    columns = schema["numeric"] + schema["categorical"]
    field_values = frame[schema["field"]].astype(str).to_numpy() if "field" in schema else None
    for i, row in enumerate(frame[columns].itertuples(index=False, name=None)):
        if row in profiles:
            union(i, profiles[row])
        profiles[row] = i
        if "field" in schema:
            field = field_values[i]
            if field in fields:
                union(i, fields[field])
            fields[field] = i
    return np.array([find(i) for i in range(len(frame))])


def make_partition(frame, schema, source_hash, dataset, seed=42, strategy="iid",
                   clients=5, alpha=0.5, min_client_rows=5):
    if strategy not in {"iid", "dirichlet", "unequal", "district"}:
        raise ValueError("Unknown partition strategy")
    if type(clients) is not int or type(min_client_rows) is not int or clients < 2 or min_client_rows < 1:
        raise ValueError("Need at least two clients and positive minimum client size")
    if type(seed) is not int or seed < 0:
        raise ValueError("Partition seed must be a nonnegative integer")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    rng = np.random.default_rng(seed)
    groups = group_ids(frame, schema)
    labels = frame[schema["target"]].astype(str).to_numpy()
    members = {int(g): np.flatnonzero(groups == g).tolist() for g in np.unique(groups)}
    by_class = {}
    for g, rows in members.items():
        crops = np.unique(labels[rows])
        if len(crops) != 1:
            raise ValueError("Identical input/field group has conflicting crop labels; resolve explicitly.")
        by_class.setdefault(str(crops[0]), []).append(g)
    splits = {"train": [], "validation": [], "test": []}
    for crop in sorted(by_class):
        values = np.array(by_class[crop], dtype=int)
        rng.shuffle(values)
        if len(values) < 3:
            raise ValueError(f"Crop {crop} has fewer than three independent groups")
        held = max(1, int(np.ceil(0.15 * len(values))))
        held = min(held, (len(values) - 1) // 2)
        splits["validation"].extend(values[:held].tolist())
        splits["test"].extend(values[held:2 * held].tolist())
        splits["train"].extend(values[2 * held:].tolist())
    train_groups = np.array(splits["train"], dtype=int)
    if strategy == "district":
        if "region" not in schema:
            raise ValueError("Dataset has no district metadata; do not invent regions")
        allocation = {}
        for g in train_groups:
            regions = frame.iloc[members[int(g)]][schema["region"]].astype(str).unique()
            if len(regions) != 1:
                raise ValueError("Repeated input/field group crosses districts; explicit protocol required")
            allocation.setdefault(str(regions[0]), []).append(int(g))
    else:
        if len(train_groups) < clients:
            raise ValueError("Too many clients for independent training groups")
        for attempt in range(200):
            allocation = {f"client_{i:03d}": [] for i in range(clients)}
            names = list(allocation)
            if strategy == "iid":
                shuffled = rng.permutation(train_groups)
                for i, g in enumerate(shuffled):
                    allocation[names[i % clients]].append(int(g))
            elif strategy == "unequal":
                weights = rng.dirichlet(np.full(clients, alpha))
                for g in rng.permutation(train_groups):
                    allocation[names[int(rng.choice(clients, p=weights))]].append(int(g))
            else:
                # Label-wise Dirichlet categorical assignment of whole groups.
                for crop in sorted(by_class):
                    weights = rng.dirichlet(np.full(clients, alpha))
                    values = [g for g in train_groups if labels[members[int(g)][0]] == crop]
                    for g in rng.permutation(values):
                        allocation[names[int(rng.choice(clients, p=weights))]].append(int(g))
            if min(sum(len(members[g]) for g in values) for values in allocation.values()) >= min_client_rows:
                break
        else:
            raise ValueError("Unable to satisfy minimum client size in 200 draws; change configuration")
    expanded = {key: sorted(i for g in values for i in members[int(g)]) for key, values in splits.items()}
    client_rows = {key: sorted(i for g in values for i in members[int(g)]) for key, values in sorted(allocation.items())}
    if any(len(rows) < min_client_rows for rows in client_rows.values()):
        raise ValueError("A district client has fewer than minimum training rows")
    counts = lambda rows: {crop: int(sum(labels[rows] == crop)) for crop in sorted(by_class)}
    partition = {
        "version": 1, "dataset": dataset, "source_sha256": source_hash,
        "seed": int(seed), "strategy": strategy, "alpha": float(alpha),
        "requested_clients": clients, "actual_clients": len(client_rows), "min_client_rows": min_client_rows,
        "schema": schema, "row_count": len(frame), "classes": sorted(by_class),
        "splits": expanded, "clients": client_rows,
        "split_class_counts": {k: counts(v) for k, v in expanded.items()},
        "client_class_counts": {k: counts(v) for k, v in client_rows.items()},
        "actual_split_fractions": {k: len(v)/len(frame) for k, v in expanded.items()},
        "split_rule": "per-class whole groups; ceil(15%) validation and test, at least one each",
        "scope": "public-data single-process simulation; no privacy guarantee",
        "synthetic": dataset == "manipur",
        "partition_draw_attempts": (attempt + 1) if strategy != "district" else 1,
    }
    validate_partition(frame, schema, partition)
    return partition


def validate_partition(frame, schema, partition):
    if partition["row_count"] != len(frame) or partition["schema"] != schema:
        raise ValueError("Partition schema/row count mismatch")
    splits = partition["splits"]
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Unexpected split names")
    combined = [i for rows in splits.values() for i in rows]
    if any(type(i) is not int for i in combined) or sorted(combined) != list(range(len(frame))):
        raise ValueError("Splits must cover every row exactly once")
    groups = group_ids(frame, schema)
    group_sets = [set(groups[rows]) for rows in splits.values()]
    if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Input profile or field leakage across splits")
    assigned = [i for rows in partition["clients"].values() for i in rows]
    if any(type(i) is not int for i in assigned) or sorted(assigned) != sorted(splits["train"]):
        raise ValueError("Clients must cover only training rows exactly once")
    if len(partition["clients"]) < 2 or any(not rows for rows in partition["clients"].values()):
        raise ValueError("Need at least two nonempty clients")
    if partition["classes"] != sorted(frame[schema["target"]].astype(str).unique().tolist()):
        raise ValueError("Class schema mismatch")
    client_groups = [set(groups[rows]) for rows in partition["clients"].values()]
    if any(client_groups[i] & client_groups[j] for i in range(len(client_groups)) for j in range(i)):
        raise ValueError("Shared repeated profile/field across clients")


def prepare_arrays(frame, schema, partition):
    """Train-only mean/standardization and one-hot vocab; unknown categories all-zero.

    Global training statistics are a centralized reference for PUBLIC simulation,
    not a privately aggregated preprocessing protocol.
    """
    validate_partition(frame, schema, partition)
    train = partition["splits"]["train"]
    numeric = frame[schema["numeric"]].to_numpy(dtype=np.float64)
    mean = numeric[train].mean(axis=0)
    scale = numeric[train].std(axis=0)
    scale[scale == 0] = 1
    arrays = [(numeric - mean) / scale]
    vocab = {}
    features = list(schema["numeric"])
    for name in schema["categorical"]:
        vocab[name] = sorted(frame.iloc[train][name].astype(str).unique().tolist())
        values = frame[name].astype(str).to_numpy()
        arrays.append(np.column_stack([values == category for category in vocab[name]]))
        features.extend([f"{name}={category}" for category in vocab[name]])
    x = np.concatenate(arrays, axis=1).astype(np.float32)
    lookup = {crop: i for i, crop in enumerate(partition["classes"])}
    y = frame[schema["target"]].astype(str).map(lookup).to_numpy(dtype=np.int64, copy=True)
    metadata = {"mean": mean.tolist(), "scale": scale.tolist(), "vocab": vocab,
                "features": features, "classes": partition["classes"],
                "policy": "global train-only transforms for public-data simulation, NOT private preprocessing"}
    return x, y, metadata
