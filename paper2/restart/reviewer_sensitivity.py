"""Post-hoc geographic sensitivity from archived per-group metrics only.

This module never loads a model, checkpoint, target vector, or test cache.
The original eight-group outer results have already been exposed, so its
output is descriptive development evidence, not independent validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def index_all_panel(rows: list[dict]) -> tuple[dict, list[str], list[int], list[str]]:
    indexed = {}
    group_sizes = {}
    algorithms, seeds = set(), set()
    for row in rows:
        if row.get("panel") != "all":
            continue
        algorithm, seed, group = row["algorithm"], row["seed"], row["group"]
        if not isinstance(algorithm, str) or not isinstance(group, str):
            raise ValueError("algorithm and group must be strings")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        metrics = row["metrics"]
        n = metrics["samples"]
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise ValueError("group sample count must be a positive integer")
        rmse = _finite_number(metrics["rmse"], "rmse")
        if rmse < 0:
            raise ValueError("rmse must be nonnegative")
        key = (algorithm, seed, group)
        if key in indexed:
            raise ValueError(f"duplicate row: {key}")
        indexed[key] = (n, rmse)
        if group in group_sizes and group_sizes[group] != n:
            raise ValueError(f"inconsistent group size: {group}")
        group_sizes[group] = n
        algorithms.add(algorithm)
        seeds.add(seed)
    if not indexed:
        raise ValueError("no all-panel rows")
    groups = sorted(group_sizes)
    for algorithm in algorithms:
        for seed in seeds:
            for group in groups:
                if (algorithm, seed, group) not in indexed:
                    raise ValueError(f"incomplete paired grid: {(algorithm, seed, group)}")
    return indexed, sorted(algorithms), sorted(seeds), groups


def aggregate(indexed: dict, algorithm: str, seed: int, groups: list[str],
              weighting: str) -> float:
    if not groups:
        raise ValueError("no groups selected")
    pairs = [indexed[(algorithm, seed, group)] for group in groups]
    if weighting == "equal_group":
        return mean(rmse for _, rmse in pairs)
    if weighting == "pooled":
        return math.sqrt(sum(n * rmse * rmse for n, rmse in pairs) /
                         sum(n for n, _ in pairs))
    if weighting == "sqrt_n_group":
        return sum(math.sqrt(n) * rmse for n, rmse in pairs) / sum(
            math.sqrt(n) for n, _ in pairs)
    raise ValueError(f"unknown weighting: {weighting}")


def analyze(rows: list[dict]) -> dict:
    indexed, algorithms, seeds, groups = index_all_panel(rows)
    group_sizes = {group: indexed[(algorithms[0], seeds[0], group)][0]
                   for group in groups}
    scenarios = {"all_groups": groups,
                 "exclude_n_le_6": [g for g in groups if group_sizes[g] > 6]}
    scenarios.update({f"leave_out_{g}": [h for h in groups if h != g]
                      for g in groups})
    weighting_names = ("equal_group", "pooled", "sqrt_n_group")
    results = []
    for scenario, kept in scenarios.items():
        if not kept:
            continue
        for weighting in weighting_names:
            by_algorithm = {}
            for algorithm in algorithms:
                values = [aggregate(indexed, algorithm, seed, kept, weighting)
                          for seed in seeds]
                by_algorithm[algorithm] = {
                    "mean": mean(values), "seed_sd": stdev(values) if len(values) > 1 else None,
                    "by_seed": dict(zip(map(str, seeds), values)),
                }
            ranking = sorted(algorithms, key=lambda a: (by_algorithm[a]["mean"], a))
            results.append({"scenario": scenario, "weighting": weighting,
                            "groups": kept, "n_systems": sum(group_sizes[g] for g in kept),
                            "ranking": ranking, "methods": by_algorithm})
    paired = []
    for group in groups:
        for algorithm in algorithms:
            if algorithm == "xgboost" or "xgboost" not in algorithms:
                continue
            deltas = [indexed[(algorithm, seed, group)][1] -
                      indexed[("xgboost", seed, group)][1] for seed in seeds]
            paired.append({"group": group, "method": algorithm,
                           "reference": "xgboost", "n_systems": group_sizes[group],
                           "rmse_difference_mean": mean(deltas),
                           "rmse_difference_by_seed": dict(zip(map(str, seeds), deltas))})
    return {
        "interpretation": "Post-hoc descriptive sensitivity on reused outer groups; seeds are not independent geographies.",
        "group_sizes": group_sizes, "algorithms": algorithms, "seeds": seeds,
        "scenarios": results, "paired_group_rmse_differences": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.read_bytes()
    payload = json.loads(source)
    report = analyze(payload["per_region"])
    report["source_sha256"] = hashlib.sha256(source).hexdigest()
    report["source_path"] = str(args.input.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
