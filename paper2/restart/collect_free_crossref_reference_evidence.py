"""Collect reproducible *candidate-only* bibliographic evidence from Crossref.

Crossref exposes public scholarly metadata without an account.  This helper
uses that free service to make the ranking-reference review easier, but it is
deliberately unable to approve a reference: bibliographic metadata cannot
establish an experiment's exact site, year, or treatment design.

The collector is conservative by design: it contacts only Crossref's fixed
HTTPS endpoint, queries serially, caches the relevant response fields, and
records failures instead of silently filling gaps.  It never modifies the
frozen candidate registry or enables ranking.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from paper2.restart.paper2_completion_audit import ROOT, RESULTS, atomic_json, sha


REGISTRY = ROOT / "paper2/configs/ranking_reference_candidates_v1.json"
CROSSREF_WORKS = "https://api.crossref.org/v1/works/"
USER_AGENT = "kg-hybrid-paper2-reference-evidence/1.0 (candidate-only; contact: local-user)"


def doi_from_candidate(value: object) -> str | None:
    """Return a normalised DOI only from a DOI URL or a bare DOI string."""
    text = str(value or "").strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered.startswith("https://doi.org/") or lowered.startswith("http://doi.org/"):
        text = urlparse(text).path.lstrip("/")
    elif lowered.startswith("doi:"):
        text = text[4:].strip()
    if text.lower().startswith("10.") and "/" in text and " " not in text:
        return text.lower()
    return None


def _year(message: dict) -> int | None:
    for field in ("published-print", "published-online", "issued", "created"):
        parts = message.get(field, {}).get("date-parts", [])
        if parts and parts[0] and isinstance(parts[0][0], int):
            return parts[0][0]
    return None


def retrieve_doi(doi: str, opener: Callable = urlopen) -> dict:
    """Retrieve and reduce one Crossref record to citeable metadata."""
    url = CROSSREF_WORKS + quote(doi, safe="/")
    request = Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
    try:
        with opener(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        message = payload.get("message")
        if not isinstance(message, dict):
            raise ValueError("Crossref response has no message object")
        titles = message.get("title", [])
        authors = [
            " ".join(part for part in (author.get("given"), author.get("family")) if part).strip()
            for author in message.get("author", []) if isinstance(author, dict)
        ]
        return {
            "status": "RETRIEVED",
            "doi": doi,
            "crossref_doi": str(message.get("DOI", "")).lower(),
            "title": titles[0] if titles else "",
            "authors": authors,
            "published_year": _year(message),
            "container_title": (message.get("container-title") or [""])[0],
            "type": message.get("type", ""),
            "publisher": message.get("publisher", ""),
            "url": message.get("URL", ""),
        }
    except HTTPError as error:
        return {"status": "NOT_RETRIEVED", "doi": doi, "error": f"HTTP_{error.code}"}
    except (URLError, TimeoutError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return {"status": "NOT_RETRIEVED", "doi": doi, "error": type(error).__name__}


def collect(
    output: Path,
    registry_path: Path = REGISTRY,
    *,
    sleep_seconds: float = 0.25,
    opener: Callable = urlopen,
) -> dict:
    """Write candidate-only Crossref evidence for every DOI in the registry."""
    output, registry_path = Path(output).resolve(), Path(registry_path).resolve()
    if output.exists() or RESULTS not in output.parents:
        raise ValueError("Output must be a new Paper 2 results directory")
    if sleep_seconds < 0:
        raise ValueError("sleep_seconds cannot be negative")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    candidates = registry.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Candidate registry must contain a non-empty candidates list")

    output.mkdir(parents=True)
    cache: dict[str, dict] = {}
    rows: list[dict] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"Candidate {index} must be an object")
        doi = doi_from_candidate(candidate.get("doi_or_url"))
        if doi is None:
            record = {"status": "NO_CROSSREF_DOI", "doi": None, "error": "candidate_is_not_a_doi"}
        elif doi in cache:
            record = dict(cache[doi])
            record["status"] = "REUSED_CACHED_RESULT" if record["status"] == "RETRIEVED" else record["status"]
        else:
            record = retrieve_doi(doi, opener)
            cache[doi] = dict(record)
            if sleep_seconds:
                time.sleep(sleep_seconds)
        rows.append({
            "experiment_id": candidate.get("experiment_id", ""),
            "candidate_source": candidate.get("candidate_source", ""),
            "candidate_doi_or_url": candidate.get("doi_or_url", ""),
            "exact_site_year_link_registry": candidate.get("exact_site_year_link", ""),
            "registry_coverage": candidate.get("coverage", ""),
            "registry_review_status": candidate.get("independent_review_status", ""),
            "candidate_only": True,
            "crossref": record,
        })

    evidence = {
        "status": "FREE_CROSSREF_METADATA_COLLECTED_CANDIDATE_ONLY",
        "collected_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "registry_sha256": sha(registry_path),
        "crossref_api_base": CROSSREF_WORKS,
        "candidate_records": len(rows),
        "crossref_retrieved": sum(row["crossref"]["status"] in {"RETRIEVED", "REUSED_CACHED_RESULT"} for row in rows),
        "not_retrieved_or_non_doi": sum(row["crossref"]["status"] not in {"RETRIEVED", "REUSED_CACHED_RESULT"} for row in rows),
        "rows": rows,
        "ranking_enabled": False,
        "source_registry_modified": False,
        "independent_review_completed": False,
        "interpretation": (
            "Crossref metadata supports bibliographic traceability only. It cannot verify the "
            "site/year/design match, source authority, or independence required to enable ranking."
        ),
    }
    atomic_json(output / "crossref_candidate_evidence.json", evidence)
    atomic_json(output / "manifest.json", {
        "status": evidence["status"],
        "registry_sha256": evidence["registry_sha256"],
        "evidence_sha256": sha(output / "crossref_candidate_evidence.json"),
        "ranking_enabled": False,
        "source_registry_modified": False,
    })
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    args = parser.parse_args()
    print(collect(args.output, args.registry, sleep_seconds=args.sleep_seconds))
