"""
Multi-language resource lookup for Mentor mode
──────────────────────────────────────────────────────────────────
- Primary path: Chroma similarity search on OpenAI embeddings.
- Fallback   : RapidFuzz partial-ratio against resources.csv
- Now supports both Japanese and Korean resources
"""

from __future__ import annotations
import csv, pathlib, typing as _t
from openai import OpenAI
import chromadb
from rapidfuzz import fuzz
from dotenv import load_dotenv 

load_dotenv()

# ————————————————————————————————————————————————
# CONFIG
ROOT       = pathlib.Path(__file__).parent
CSV_PATH_JP = ROOT / "resources.csv"          # Japanese resources
CSV_PATH_KR = ROOT / "resources_korean.csv"   # Korean resources
DB_DIR     = ROOT.parent / "vectordb"
COLL_NAME_JP = "tancho-res-japanese"
COLL_NAME_KR = "tancho-res-korean"
EMB_MODEL  = "text-embedding-3-small"
FUZZY_TH   = 75
TOP_K      = 3

# ————————————————————————————————————————————————
# set up clients (reuse across calls)
_openai = OpenAI()
_chroma = chromadb.PersistentClient(str(DB_DIR))
_coll_jp = _chroma.get_or_create_collection(COLL_NAME_JP)
_coll_kr = _chroma.get_or_create_collection(COLL_NAME_KR)

# ———— helper: get embedding ————
def _embed(text: str) -> list[float]:
    return _openai.embeddings.create(
        model=EMB_MODEL,
        input=text
    ).data[0].embedding

# ———— helper: load CSV once for fuzzy fallback ————
def _load_csv(csv_path: pathlib.Path) -> list[dict]:
    if not csv_path.exists():
        return []
        
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows   = []
        for raw in reader:
            row = { (k or "").strip().lower(): (v or "").strip()
                    for k, v in raw.items() }
            if any(row.values()):
                rows.append(row)
        return rows

_CSV_ROWS_JP = _load_csv(CSV_PATH_JP)
_CSV_ROWS_KR = _load_csv(CSV_PATH_KR)

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

def match_resources(query: str, language: str = "japanese", limit: int = TOP_K) -> list[dict]:
    """
    Semantic search with vector DB; fuzzy CSV fallback if no hit.
    Now supports both Japanese and Korean resources.
    Returns ≤ `limit` dicts with keys:
        title, type, difficulty, study_time, description
    """
    if not query:
        return []

    # Select appropriate collection and CSV data
    if language.lower() == "korean":
        collection = _coll_kr
        csv_rows = _CSV_ROWS_KR
    else:
        collection = _coll_jp
        csv_rows = _CSV_ROWS_JP

    # ——— vector search ———
    try:
        vec   = _embed(query)
        resp  = collection.query(vec, n_results=limit)
        hits  = [h["metadata"] for h in resp] if resp else []
    except Exception as e:
        print(f"Vector search failed for {language} →", e)
        hits = []

    if hits:
        return hits

    # ——— fuzzy fallback ———
    scored = []
    for r in csv_rows:
        s = max(_score(query, r.get("name","")), _score(query, r.get("key topics","")))
        if s >= FUZZY_TH:
            scored.append((s, r))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [_format_row(r) for _, r in scored[:limit]]