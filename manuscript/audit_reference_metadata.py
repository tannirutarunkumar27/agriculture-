"""Read-only Crossref lookup of a supplied bibliography; retain audit evidence."""
import argparse
import concurrent.futures
import difflib
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def normalize(value):
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower())


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "BibliographyVerification/1.0", "Accept": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))


def compact(message):
    return {field: message.get(field) for field in (
        "DOI", "title", "author", "container-title", "volume", "issue", "page",
        "article-number", "published", "published-print", "published-online", "type", "publisher", "URL")}


def lookup(entry):
    record = {"key": entry["key"], "supplied": entry, "doi_lookup": None, "title_candidates": []}
    if entry.get("doi"):
        try:
            message = get_json("https://api.crossref.org/works/" + urllib.parse.quote(entry["doi"], safe=""))["message"]
            record["doi_lookup"] = compact(message)
            record["doi_title_matches"] = normalize(message["title"][0]) == normalize(entry["title"])
        except (urllib.error.URLError, TimeoutError) as error:
            record["doi_error"] = str(error)
    if not record.get("doi_title_matches"):
        try:
            query = urllib.parse.urlencode({"query.title": entry["title"], "rows": 4})
            messages = get_json("https://api.crossref.org/works?" + query)["message"]["items"]
            record["title_candidates"] = [
                {**compact(message), "title_similarity": difflib.SequenceMatcher(
                    None, normalize(entry["title"]), normalize(message.get("title", [""])[0])).ratio()}
                for message in messages]
        except (urllib.error.URLError, TimeoutError) as error:
            record["title_error"] = str(error)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    entries = []
    for kind, key, body in re.findall(r"@(\w+)\{([^,]+),(.*?)\n\}", args.source.read_text(encoding="utf-8"), re.S):
        fields = dict(re.findall(r"^\s*(\w+)\s*=\s*\{(.*?)\},?\s*$", body, re.M))
        entries.append({"key": key.strip(), "kind": kind, **fields})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        for record in pool.map(lookup, entries):
            records.append(record)
            match = record.get("doi_title_matches", False)
            candidates = record["title_candidates"]
            best = max(candidates, key=lambda item: item["title_similarity"], default={})
            print(record["key"], "DOI_MATCH" if match else "TITLE_CANDIDATE",
                  best.get("title_similarity", ""), best.get("title", ""), flush=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved", len(records), "reference audit records", flush=True)


if __name__ == "__main__":
    main()
