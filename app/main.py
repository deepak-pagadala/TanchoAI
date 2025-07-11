# app/main.py
# ───────────────────────────────────────────────────────────────
# Tancho AI  –  Conversation & Mentor endpoints
#  ▸ Smart recommendation logic:
#       • explicit ask   → recommend immediately
#       • repeat topic ≥ 3 times → recommend
#       • generic “Any resources?” re-uses last real topic
#  ▸ Mentor sees full metadata (type, study-time, difficulty, description)
#    via RESOURCE_CONTEXT so it can answer detailed follow-ups.
# ───────────────────────────────────────────────────────────────

from typing import Literal, List, Dict
import os, sys, json

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

# ───────── external helper libs ──────────
from langdetect import detect          # pip install langdetect
import yake                             # pip install yake
from rapidfuzz import fuzz             # pip install rapidfuzz

# ───────── project-local helpers ─────────
from app.dummy_store import (
    get_history, write_turns,
    inc_topic, topic_hits,
    remember_resource, last_resource      # add these two tiny fns in dummy_store.py
)
from app.prompts   import PROMPTS
from app.resources import match_resources

# ───────── config & constants ─────────
HISTORY_WINDOW          = 6
MODEL_NAME              = "gpt-4o"
RECOMMEND_AFTER_N_HITS  = 3            # curiosity threshold

_GENERIC_TOKENS = {
    "resource", "resources", "material", "materials", "book", "books",
    "guide", "exercise", "exercises", "practice", "lesson", "lessons",
    "course", "courses", "app", "website", "reference", "recommend"
}

_RESOURCE_TRIGGERS = _GENERIC_TOKENS   # same list; clearer semantics
_last_topic: dict[str, str] = {}       # per-user memory of last concrete topic

# ───────── YAKE keyword extractor ─────────
_YAKE_CACHE: dict[str, yake.KeywordExtractor] = {}
def _extract_topic(text: str) -> str:
    """Return a single lowercase keyword; fallback first word."""
    try:
        lang = detect(text)
    except Exception:
        lang = "en"
    if lang not in _YAKE_CACHE:
        _YAKE_CACHE[lang] = yake.KeywordExtractor(lan=lang, n=1,
                                                  dedupLim=0.9, top=1)
    kws = _YAKE_CACHE[lang].extract_keywords(text)
    return (kws[0][0] if kws else text.split()[0]).lower()

def _wants_resources(text: str) -> bool:
    tl = text.lower()
    return any(t in tl for t in _RESOURCE_TRIGGERS)

# ───────── summary stubs (future) ─────────
def read_summary(uid: str) -> str | None: return None
def write_summary(uid: str, summary: str) -> None:   pass

# ───────── history helper ─────────
def _history_context(turns: List[Dict]) -> str:
    if not turns: return ""
    lines = [
        f'{"USER" if t["role"]=="user" else "ASSISTANT"}: {t["content"]}'
        for t in turns[-HISTORY_WINDOW:]
    ]
    return "\n".join(lines)

# ───────── build a small YAML context block ─────────
def _context_snippet(topic: str, k: int = 3) -> str:
    hits = match_resources(topic, limit=k)
    return "\n\n".join(
        f"- title: {r['title']}\n"
        f"  type: {r['type']}\n"
        f"  difficulty: {r['difficulty']}\n"
        f"  study_time: {r['study_time']}\n"
        f"  description: {r['description']}"
        for r in hits
    )

# ───────── FastAPI boot ─────────
load_dotenv()
client = AsyncOpenAI()
app    = FastAPI()

# ============================================================
#  1. Conversation endpoint (unchanged)
# ============================================================
class ChatRequest(BaseModel):
    uid: str
    mode: Literal["convFormal", "convCasual"]
    userMessage: str

@app.post("/chat")
async def chat(body: ChatRequest):
    uid, mode, user_msg = body.uid, body.mode, body.userMessage

    history       = get_history(uid)
    system_prompt = PROMPTS[mode]

    if ctx := _history_context(history):
        system_prompt += "\n\nLast turns:\n" + ctx
    if summary := read_summary(uid):
        system_prompt += "\n\nChat summary so far:\n" + summary

    messages = (
        [{"role": "system", "content": system_prompt}]
        + history[-HISTORY_WINDOW:]
        + [{"role": "user", "content": user_msg}]
    )

    try:
        resp      = await client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=0.7)
        assistant = resp.choices[0].message.content.strip()
        payload   = json.loads(assistant)             # Conversation returns JSON
        write_turns(uid, user_msg, assistant)
        return JSONResponse(payload)
    except Exception as e:
        print("CHAT ERROR:", e, file=sys.stderr)
        return JSONResponse(status_code=500,
                            content={"detail": str(e)})

# ============================================================
#  2. Mentor endpoint (smart, resource-aware)
# ============================================================
class MentorReq(BaseModel):
    uid: str
    question: str

@app.post("/mentor", response_class=StreamingResponse)
async def mentor(body: MentorReq):
    uid, q = body.uid, body.question

    # 1️⃣ explicit resource ask?
    explicit = _wants_resources(q)

    # 2️⃣ extract topic keyword
    candidate_topic = _extract_topic(q)
    is_generic      = candidate_topic in _GENERIC_TOKENS

    # 3️⃣ choose the real topic we track
    if is_generic:
        topic = _last_topic.get(uid) or last_resource(uid) or candidate_topic
    else:
        topic = candidate_topic
        _last_topic[uid] = topic

    # 4️⃣ update repetition counter (only real topics)
    if topic and not is_generic:
        inc_topic(uid, topic)
    hits = topic_hits(uid, topic) if topic else 0

    # 5️⃣ recommend?
    force_reco = explicit or (hits >= RECOMMEND_AFTER_N_HITS)

    # 6️⃣ fetch resource list + lines
    if force_reco and topic:
        res_list  = match_resources(topic)
        res_lines = "\n".join(
            f"- {r['title']} ({r['type']}, {r['study_time']} hrs, {r['difficulty']})"
            for r in res_list
        ) or "NONE"
        if res_list:
            remember_resource(uid, res_list[0]["title"])
    else:
        res_list  = []
        res_lines = "NONE"

    # 7️⃣ build system prompt with context snippet
    context_block = _context_snippet(topic) if topic else ""
    system_prompt = (
        PROMPTS["mentor"]
        + "\n\n### RESOURCE_CONTEXT\n" + context_block
        + f"\n\nTOPIC_HITS: {RECOMMEND_AFTER_N_HITS if force_reco else hits}"
        + f"\n\nAVAILABLE_RESOURCES:\n{res_lines}"
    )

    # 8️⃣ messages
    messages = (
        [{"role": "system", "content": system_prompt}]
        + get_history(uid)
        + [{"role": "user", "content": q}]
    )

    # 9️⃣ stream completion
    async def event_stream():
        assistant_full = ""
        try:
            stream = await client.chat.completions.create(
                model=MODEL_NAME,
                stream=True,
                messages=messages,
                temperature=0.7)
            async for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                assistant_full += token
                yield f"data: {token}\n\n"
            yield "event: done\ndata:[DONE]\n\n"
            write_turns(uid, q, assistant_full)
        except Exception as e:
            yield f"event: error\ndata:{e}\n\n"

    return StreamingResponse(event_stream(),
                             media_type="text/event-stream")
