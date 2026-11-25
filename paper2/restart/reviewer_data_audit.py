"""Aggregate-only provenance and category-hash audit of public regression rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

CATEGORIES = ("previous_crop", "crop_season", "pest_control")
BINS = 8


def category_bin(field: str, value: str) -> int:
    digest = hashlib.sha256((field + "\0" + value).encode("utf-8")).digest()
    return int.from_bytes(digest, "big") % BINS


def analyze(rows: list[dict[str, str]]) -> dict:
    if not rows:
        raise ValueError("empty data")
    required = {"row_id", "experiment_id", "geo_group", *CATEGORIES}
    if any(not required.issubset(row) for row in rows):
        raise ValueError("missing required columns")
    if len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate row identity")
    by_experiment: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if not row["experiment_id"] or not row["geo_group"]:
            raise ValueError("missing experiment or geography")
        by_experiment[row["experiment_id"]].append(row)
        by_group[row["geo_group"]].append(row)
    nested = [exp for exp, items in by_experiment.items()
              if len({row["geo_group"] for row in items}) > 1]
    if nested:
        raise ValueError("experiment crosses geographic groups")
    collision_report = {}
    for field in CATEGORIES:
        observed = sorted({row[field] for row in rows if row[field] != ""})
        occupancy = Counter(category_bin(field, value) for value in observed)
        collision_report[field] = {
            "unique_nonmissing_categories": len(observed),
            "occupied_bins": len(occupancy),
            "categories_in_colliding_bins": sum(n for n in occupancy.values() if n > 1),
            "distinct_colliding_pairs": sum(n * (n - 1) // 2 for n in occupancy.values()),
            "missing_rows": sum(row[field] == "" for row in rows),
        }
    return {
        "n_systems": len(rows), "n_experiments": len(by_experiment),
        "systems_per_experiment": dict(sorted(Counter(map(len, by_experiment.values())).items())),
        "n_experiments_with_multiple_systems": sum(len(v) > 1 for v in by_experiment.values()),
        "max_systems_in_one_experiment": max(map(len, by_experiment.values())),
        "groups": {g: {"systems": len(items),
                       "experiments": len({row["experiment_id"] for row in items})}
                   for g, items in sorted(by_group.items())},
        "category_hash": {"bins_per_field": BINS, "fields": collision_report},
        "interpretation": "Shared experiment IDs show record clustering; no effective independent sample size is inferred. Hash collisions are distinct-category collisions in this observed dataset, not a privacy guarantee.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.read_bytes()
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        report = analyze(list(csv.DictReader(handle)))
    report["source_sha256"] = hashlib.sha256(source).hexdigest()
    report["source_path"] = str(args.input.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
