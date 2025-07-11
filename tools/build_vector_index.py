# tools/build_vector_index.py
"""
Build / refresh the Chroma vector index for Tancho resources.
-------------------------------------------------------------
Run:
    $ python tools/build_vector_index.py
Requires:
    • OPENAI_API_KEY in env or .env
    • pip install openai chromadb python-dotenv
"""

from __future__ import annotations
import csv, pathlib, os, sys
from dotenv import load_dotenv      # ← pick up .env if present
from openai import OpenAI
import chromadb

# ───────── paths & constants ─────────
ROOT      = pathlib.Path(__file__).resolve().parents[1]
CSV_FILE  = ROOT / "app" / "resources.csv"
DB_DIR    = ROOT / "vectordb"
COLL_NAME = "tancho-res"
EMB_MODEL = "text-embedding-3-small"

# ───────── env & clients ─────────
load_dotenv()                       # looks for .env, okay if missing
client = OpenAI()                   # needs OPENAI_API_KEY
chroma = chromadb.PersistentClient(str(DB_DIR))
coll   = chroma.get_or_create_collection(COLL_NAME)

def _embed(text: str) -> list[float]:
    return client.embeddings.create(
        model=EMB_MODEL,
        input=text
    ).data[0].embedding

# ───────── read & normalise CSV ─────────
print(f"Building vector index from {CSV_FILE.name} …")

with CSV_FILE.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = []
    for raw in reader:
        # lower-case & trim all header keys
        row = { (k or "").strip().lower(): (v or "").strip()
                for k, v in raw.items() }
        if any(row.values()):
            rows.append(row)

# wipe existing vectors to avoid dupes
if coll.count():
    coll.delete(ids=coll.get()["ids"])

# ───────── embed & store ─────────
for row in rows:
    combined_text = " | ".join([
        row.get("name", ""),
        row.get("description", ""),
        row.get("key topics", "")
    ])
    emb = _embed(combined_text)

    coll.add(
        ids=[row["name"]],           # assumes “Name” field is unique
        embeddings=[emb],
        metadatas=[{
            "title":       row.get("name", ""),
            "type":        row.get("type", ""),
            "difficulty":  row.get("difficulty", ""),
            "study_time":  row.get("study time", ""),
            "description": row.get("description", "")
        }]
    )

print(f"Indexed {len(rows)} resources ✔︎  (DB: {DB_DIR}/)")
