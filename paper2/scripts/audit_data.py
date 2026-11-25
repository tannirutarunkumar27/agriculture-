"""Read-only raw-data audit; explicit output must not already exist."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "KG-Hybrid" / "data" / "raw"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_csv(path: Path) -> dict:
    """Do not emit raw records. NASA -999 is missing, not a measurement."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        first = stream.readline()
        nasa = first.strip() == "-BEGIN HEADER-"
        if nasa:
            header = [first.strip()]
            for line in stream:
                header.append(line.strip())
                if line.strip() == "-END HEADER-":
                    break
            else:
                raise ValueError(f"NASA header has no end marker: {path.name}")
        else:
            header = []
            stream.seek(0)
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        if not columns or len(set(columns)) != len(columns):
            raise ValueError(f"Missing or duplicated column names: {path.name}")
        missing = Counter({name: 0 for name in columns})
        labels = Counter()
        label = next((c for c in columns if c.lower() in {"label", "crop"}), None)
        seen = set()
        duplicates = 0
        count = 0
        weather_valid = 0
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed row {count + 2}: {path.name}")
            values = tuple(row[c] for c in columns)
            duplicates += values in seen
            seen.add(values)
            count += 1
            for name, value in row.items():
                absent = value.strip().lower() in {"", "nan", "null", "na", "n/a"}
                if nasa and name not in {"YEAR", "DOY"}:
                    absent = absent or value.strip() in {"-999", "-999.0"}
                    weather_valid += not absent
                missing[name] += absent
            if label:
                labels[row[label]] += 1
    warnings = []
    if nasa and weather_valid == 0:
        warnings.append("No usable weather values: all values missing or NASA -999 sentinels.")
    if nasa and count < 365:
        warnings.append("Less than one year of daily weather; not adequate for planned seasonal analysis.")
    if "Fertilizer" in columns:
        warnings.append("Exclude Fertilizer and Link from crop inputs pending an explicit causal/protocol justification.")
    if "field_id" in columns and "date" in columns:
        warnings.append("Repeated field observations require field/time-aware splitting, not random rows.")
    if any("Yield-" in c for c in columns):
        warnings.append("Historical production statistics are not best-crop labels; separate aggregates from individual crops.")
    return {
        "path": str(path), "sha256": sha256(path), "rows": count,
        "columns": columns, "missing_by_column": dict(missing),
        "exact_duplicate_rows": duplicates, "label_column": label,
        "class_counts": dict(sorted(labels.items())), "warnings": warnings,
        "nasa_header": header, "usable_weather_values": weather_valid if nasa else None,
    }


def audit(raw: Path) -> dict:
    if not raw.is_dir():
        raise FileNotFoundError(raw)
    # FAOSTAT bulk tables require a separate, domain-specific audit; no silent inclusion.
    paths = sorted(p for p in raw.rglob("*.csv") if "FAOSTAT_A-S_E" not in p.parts)
    reports = [inspect_csv(p) for p in paths]
    groups = {}
    for report in reports:
        groups.setdefault(report["sha256"], []).append(report["path"])
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(raw.resolve()), "datasets": reports,
        "duplicate_files": [v for v in groups.values() if len(v) > 1],
        "excluded": ["FAOSTAT_A-S_E bulk tables: not inspected"],
        "scope": "Structural audit only; no source authority, agronomic units, or label validity certification.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {args.output}")
    report = audit(args.raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(f"Audited {len(report['datasets'])} CSVs; duplicate file groups: {len(report['duplicate_files'])}")
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
