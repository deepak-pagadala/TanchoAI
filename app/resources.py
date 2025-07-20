"""
Resource lookup for Mentor mode  –  vector search + fuzzy fallback
──────────────────────────────────────────────────────────────────
• Primary path: Chroma similarity search on OpenAI embeddings.
• Fallback   : RapidFuzz partial-ratio against resources.csv
"""

from __future__ import annotations
import csv, pathlib, typing as _t
from openai import OpenAI                  # pip install openai
import chromadb                            # pip install chromadb
from rapidfuzz import fuzz                 # pip install rapidfuzz
from dotenv import load_dotenv 

load_dotenv()

# ————————————————————————————————————————————————
# CONFIG
ROOT       = pathlib.Path(__file__).parent
CSV_PATH   = ROOT / "resources.csv"
DB_DIR     = ROOT.parent / "vectordb"
COLL_NAME  = "tancho-res"
EMB_MODEL  = "text-embedding-3-small"
FUZZY_TH   = 75            # fuzzy threshold 0-100
TOP_K      = 3

# ————————————————————————————————————————————————
# set up clients (reuse across calls)
_openai = OpenAI()
_chroma = chromadb.PersistentClient(str(DB_DIR))
_coll   = _chroma.get_or_create_collection(COLL_NAME)

# ———— helper: get embedding ————
def _embed(text: str) -> list[float]:
    return _openai.embeddings.create(
        model=EMB_MODEL,
        input=text
    ).data[0].embedding

# ———— helper: load CSV once for fuzzy fallback ————
def _load_csv() -> list[dict]:
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows   = []
        for raw in reader:
            row = { (k or "").strip().lower(): (v or "").strip()
                    for k, v in raw.items() }
            if any(row.values()):
                rows.append(row)
        return rows

_CSV_ROWS = _load_csv()

# ———— fuzzy score ————
def _score(needle: str, hay: str) -> int:
    return fuzz.partial_ratio(needle.lower(), hay.lower())

# ————————————————————————————————————————————————
def _format_row(row: dict) -> dict:
    """Return canonical dict used by Mentor mode."""
    return {
        "title":       row.get("name", ""),
        "type":        row.get("type", ""),
        "difficulty":  row.get("difficulty", ""),
        "study_time":  row.get("study time", ""),
        "description": row.get("description", "")
    }

def match_resources(query: str, limit: int = TOP_K) -> list[dict]:
    """
    Semantic search with vector DB; fuzzy CSV fallback if no hit.
    Returns ≤ `limit` dicts with keys:
        title, type, difficulty, study_time, description
    """
    if not query:
        return []

    # ——— vector search ———
    try:
        vec   = _embed(query)
        resp  = _coll.query(vec, n_results=limit)
        hits  = [h["metadata"] for h in resp] if resp else []
    except Exception as e:            # DB missing? embedding error?
        print("Vector search failed →", e)
        hits = []

    if hits:
        return hits

    # ——— fuzzy fallback ———
    scored = []
    for r in _CSV_ROWS:
        s = max(_score(query, r.get("name","")), _score(query, r.get("key topics","")))
        if s >= FUZZY_TH:
            scored.append((s, r))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [_format_row(r) for _, r in scored[:limit]]
