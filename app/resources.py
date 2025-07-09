# app/resources.py
import csv, pathlib, typing as _t

# location of the CSV relative to this file
_CSV_PATH = pathlib.Path(__file__).parent / "resources.csv"

def _load_resources() -> list[dict]:
    """
    Read resources.csv → list[dict] with lowercase, trimmed keys.
    Blank lines are skipped.
    """
    with _CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: list[dict] = []
        for raw in reader:
            # normalise header names & strip cell whitespace
            row = {
                (k or "").strip().lower(): (v or "").strip()
                for k, v in raw.items()
            }
            if any(row.values()):      # skip empty rows
                rows.append(row)
        return rows

_RESOURCES = _load_resources()   # cache at import time


def match_resources(query: str, limit: int = 3) -> list[dict]:
    """
    Simple substring match against Name or Key topics.
    Returns at most `limit` results in original (mixed-case) form.
    """
    q = query.lower()
    hits: list[dict] = []

    for r in _RESOURCES:
        name = r.get("name")            # header “Name”
        topics = r.get("key topics")    # header “Key topics”
        if not name:
            continue
        if q in name.lower() or q in (topics or "").lower():
            # Provide a slim dict the mentor prompt can reference
            hits.append({
                "title": name,
                "type": r.get("type", ""),
                "difficulty": r.get("difficulty", ""),
                "study_time": r.get("study time", ""),
                "description": r.get("description", "")
            })
        if len(hits) >= limit:
            break
    return hits
