# app/main.py
# ───────────────────────────────────────────────────────────────
# Tancho AI  –  Conversation & Mentor endpoints
# • Smart recommendation logic
# • Calendar-aware FREE_SLOT injection
# • Mentor sees full metadata via RESOURCE_CONTEXT
# ───────────────────────────────────────────────────────────────

from __future__ import annotations
import pathlib
from typing import Literal, List, Dict, Optional
import os, sys, json
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from openai import AsyncOpenAI
from pydantic import BaseModel
from fastapi import File, UploadFile, Form
from fastapi.staticfiles import StaticFiles


# ───────── external helper libs ──────────
from langdetect import detect              # pip install langdetect
import yake                                 # pip install yake
from rapidfuzz import fuzz                 # pip install rapidfuzz

# ───────── FastAPI boot ─────────
load_dotenv()
client = AsyncOpenAI()
app    = FastAPI()

# ── ensure ./app/static exists ──────────────────────────────────
static_root = pathlib.Path(__file__).resolve().parent / "static"
static_root.mkdir(parents=True, exist_ok=True)

# serve http://…/static/*
app.mount("/static", StaticFiles(directory=static_root), name="static")


# ───────── project-local helpers ─────────
from app.dummy_store import (
    get_history, write_turns,
    inc_topic, topic_hits,
    remember_resource, last_resource
)
from app.prompts   import PROMPTS
from app.resources import match_resources

# ───────── config & constants ─────────
HISTORY_WINDOW          = 6
MODEL_NAME              = "gpt-4o"
RECOMMEND_AFTER_N_HITS  = 3          # curiosity threshold

_GENERIC_TOKENS = {
    "resource","resources","material","materials","book","books",
    "guide","exercise","exercises","practice","lesson","lessons",
    "course","courses","app","website","reference","recommend"
}
_RESOURCE_TRIGGERS = _GENERIC_TOKENS.copy()
_last_topic: dict[str, str] = {}     # per-user last concrete topic

# ───────── YAKE keyword extractor ─────────
_YAKE: dict[str, yake.KeywordExtractor] = {}
def _extract_topic(text: str) -> str:
    try:    lang = detect(text)
    except: lang = "en"
    if lang not in _YAKE:
        _YAKE[lang] = yake.KeywordExtractor(lan=lang, n=1, dedupLim=0.9, top=1)
    kws = _YAKE[lang].extract_keywords(text)
    return (kws[0][0] if kws else text.split()[0]).lower()

def _wants_resources(t: str) -> bool:
    tl = t.lower()
    return any(x in tl for x in _RESOURCE_TRIGGERS)

# ───────── summary stubs (future) ─────────
def read_summary(uid: str) -> str|None: return None
def write_summary(uid: str, summary: str) -> None:  ...

# ───────── misc helpers ─────────
def _history_context(turns: List[Dict]) -> str:
    if not turns: return ""
    return "\n".join(
        f'{"USER" if t["role"]=="user" else "ASSISTANT"}: {t["content"]}'
        for t in turns[-HISTORY_WINDOW:]
    )

def _context_snippet(topic: str, k=3) -> str:
    hits = match_resources(topic, limit=k)
    return "\n\n".join(
        f"- title: {r['title']}\n"
        f"  type: {r['type']}\n"
        f"  difficulty: {r['difficulty']}\n"
        f"  study_time: {r['study_time']}\n"
        f"  description: {r['description']}"
        for r in hits
    )

async def _conversation_reply(uid:str, mode:str, user_msg:str)->Dict:
    history       = get_history(uid)
    user_name     = body.name or "フレンド"
    template      = PROMPTS[mode]
    system_prompt = template.format(USER_NAME=user_name)

    if ctx := _history_context(history):
        system_prompt += "\n\nLast turns:\n" + ctx
    messages = (
        [{"role":"system","content":system_prompt}]
        + history[-HISTORY_WINDOW:]
        + [{"role":"user","content":user_msg}]
    )
    resp = await client.chat.completions.create(model=MODEL_NAME,
            messages=messages, temperature=0.7)
    assistant = resp.choices[0].message.content.strip()
    write_turns(uid,user_msg,assistant)
    return json.loads(assistant)

async def _mentor_reply(uid:str, user_msg:str, free_slot_iso:str|None=None)->Dict:
    explicit = _wants_resources(user_msg)
    cand = _extract_topic(user_msg)
    generic = cand in _GENERIC_TOKENS
    topic = (_last_topic.get(uid) or last_resource(uid) or cand) if generic else cand
    if not generic: _last_topic[uid] = topic
    if topic and not generic: inc_topic(uid,topic)
    hits = topic_hits(uid,topic) if topic else 0
    force = explicit or (hits>=RECOMMEND_AFTER_N_HITS)
    if force and topic:
        res = match_resources(topic)
        res_lines = "\n".join(
            f"- {r['title']} ({r['type']}, {r['study_time']} hrs, {r['difficulty']})"
            for r in res) or "NONE"
        if res: remember_resource(uid,res[0]["title"])
    else:
        res_lines="NONE"
    sys_prompt = (PROMPTS["mentor"]
        + "\n\n### RESOURCE_CONTEXT\n" + (_context_snippet(topic) if topic else "")
        + f"\n\nTOPIC_HITS: {RECOMMEND_AFTER_N_HITS if force else hits}"
        + f"\n\nAVAILABLE_RESOURCES:\n{res_lines}")
    if free_slot_iso: sys_prompt += f"\n\nFREE_SLOT: {free_slot_iso}"
    messages = (
        [{"role":"system","content":sys_prompt}]
        + get_history(uid)
        + [{"role":"user","content":user_msg}]
    )
    resp = await client.chat.completions.create(model=MODEL_NAME,
            messages=messages, temperature=0.7)
    assistant = resp.choices[0].message.content.strip()
    write_turns(uid,user_msg,assistant)
    return json.loads(assistant)

# ───────── Voice helper (dict) ─────────
async def _voice_reply(uid:str, user_msg:str)->Dict:
    messages = (
        [{"role":"system","content":PROMPTS["voice"]}]
        + get_history(uid)
        + [{"role":"user","content":user_msg}]
    )
    resp = await client.chat.completions.create(model=MODEL_NAME,
            messages=messages, temperature=0.7)
    assistant = resp.choices[0].message.content.strip()
    write_turns(uid,user_msg,assistant)
    return json.loads(assistant)



# ===============================================================
# 1. Conversation endpoint  (unchanged)
# ===============================================================
class ChatRequest(BaseModel):
    uid: str
    mode: Literal["convFormal","convCasual"]
    userMessage: str
    name: Optional[str] = None

@app.post("/chat")
async def chat(body: ChatRequest):
    uid, mode, user_msg, user_name = body.uid, body.mode, body.userMessage,  body.name or "there"
    history       = get_history(uid)
    template      = PROMPTS[body.mode]
    system_prompt = template.format(USER_NAME=user_name)


    if ctx := _history_context(history):
        system_prompt += "\n\nLast turns:\n" + ctx
    if summary := read_summary(uid):
        system_prompt += "\n\nChat summary so far:\n" + summary

    messages = (
        [{"role":"system","content":system_prompt}]
        + history[-HISTORY_WINDOW:]
        + [{"role":"user","content":body.userMessage}]
    )

    try:
        resp  = await client.chat.completions.create(
                    model=MODEL_NAME, messages=messages, temperature=0.7)
        assistant = resp.choices[0].message.content.strip()
        payload   = json.loads(assistant)
        write_turns(uid, user_msg, assistant)
        return JSONResponse(payload)
    except Exception as e:
        print("CHAT ERROR:", e, file=sys.stderr)
        return JSONResponse(status_code=500, content={"detail": str(e)})

# ===============================================================
# 2. Mentor endpoint  (resource- & calendar-aware)
# ===============================================================

from app.dummy_store import (
    save_pending_calendar_event,
    get_pending_calendar_event,
    clear_pending_calendar_event,
)

class ConfirmReq(BaseModel):
    uid: str
    confirmation: str
@app.post("/mentor/confirm")
async def mentor_confirm(body: ConfirmReq):
    pending = get_pending_calendar_event(body.uid)
    if not pending:
        return JSONResponse({"message": "No pending event found."}, status_code=400)

    ans = body.confirmation.strip().lower()
    if ans.startswith(("y", "yes", "sure", "okay")):
        # We won’t create the event server-side—iOS will handle that.
        clear_pending_calendar_event(body.uid)
        return {"message": "Great! I’ve queued it on your device’s calendar."}
    else:
        clear_pending_calendar_event(body.uid)
        return {"message": "No problem—let me know if you change your mind."}

    
class MentorReq(BaseModel):
    uid: str
    question: str
    freeSlot: Optional[str] = None     # ISO-8601 start of free hour
    name: Optional[str] = None

@app.post("/mentor", response_class=StreamingResponse)
async def mentor(body: MentorReq):
    uid, q, free_slot_iso, user_name = body.uid, body.question, body.freeSlot, body.name or "there"

    # 1️⃣ explicit request?
    explicit = _wants_resources(q)

    # 2️⃣ topic extraction
    cand_topic = _extract_topic(q)
    is_generic = cand_topic in _GENERIC_TOKENS
    topic = (_last_topic.get(uid) or last_resource(uid) or cand_topic) if is_generic else cand_topic
    if not is_generic: _last_topic[uid] = topic

    # 3️⃣ hit counter
    if topic and not is_generic:
        inc_topic(uid, topic)
    hits = topic_hits(uid, topic) if topic else 0

    # 4️⃣ decide on recommendation
    force_reco = explicit or (hits >= RECOMMEND_AFTER_N_HITS)

    # 5️⃣ resource list
    if force_reco and topic:
        res_list = match_resources(topic)
        res_lines = "\n".join(
            f"- {r['title']} ({r['type']}, {r['study_time']} hrs, {r['difficulty']})"
            for r in res_list
        ) or "NONE"
        if res_list: remember_resource(uid, res_list[0]["title"])
    else:
        res_list  = []
        res_lines = "NONE"

    # 6️⃣ build system prompt
    template    = PROMPTS["mentor"]
    base_prompt = template.format(USER_NAME=user_name)

# 3) Build out the rest of your system prompt
    sys_prompt = (
        base_prompt
        + "\n\n### RESOURCE_CONTEXT\n"
        + ( _context_snippet(topic) if topic else "" )
        + f"\n\nTOPIC_HITS: {RECOMMEND_AFTER_N_HITS if force_reco else hits}"
        + f"\n\nAVAILABLE_RESOURCES:\n{res_lines}"
    )
    if free_slot_iso:
        sys_prompt += f"\n\nFREE_SLOT: {free_slot_iso}"

    # 7️⃣ messages
    messages = (
        [{"role":"system","content":sys_prompt}]
        + get_history(uid)
        + [{"role":"user","content":q}]
    )

    # 8️⃣ stream
    async def event_stream():
        answer = ""
        try:
            stream = await client.chat.completions.create(
                        model=MODEL_NAME, messages=messages,
                        stream=True, temperature=0.7)
            async for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                answer += token
                yield f"data: {token}\n\n"
            yield "event: done\ndata:[DONE]\n\n"
            write_turns(uid, q, answer)
        except Exception as e:
            yield f"event: error\ndata:{e}\n\n"

    return StreamingResponse(event_stream(),
                             media_type="text/event-stream")
# ===============================================================
# 3. Voice  • /voice_chat
# ===============================================================
class VoiceResp(BaseModel):
    transcript:  str
    jp:          str
    en:          str
    correction:  str | None = None
    ttsUrl:     str | None = None 
    # legacy keys for current iOS code
    answer:          str | None = None
    recommendation:  str | None = None

@app.post("/voice_chat", response_model=VoiceResp)
async def voice_chat(
    uid: str               = Form(...),
    voiceMode: str         = Form(...),   # "conversation","mentor","voice"
    convSubMode: str|None  = Form(None),  # convCasual / convFormal
    freeSlot: Optional[str]= Form(None),  # ISO slot from iOS
    audio: UploadFile      = File(...)
):
    # 1️⃣  Whisper transcription
    audio_bytes = await audio.read()
    txt = await client.audio.transcriptions.create(
        model="whisper-1",
        file=("speech.wav", audio_bytes, "audio/wav"),   # clearer mime
        response_format="text")
    transcript = txt.strip()

    # 2️⃣  Route
    if voiceMode == "conversation":
        sub = convSubMode if convSubMode in ("convCasual","convFormal") else "convCasual"
        r = await _conversation_reply(uid, sub, transcript)
        return VoiceResp(transcript=transcript,
                         jp=r["reply"], en="",
                         correction="",
                         answer=r["reply"])

    if voiceMode == "mentor":
        r = await _mentor_reply(uid, transcript, free_slot_iso=freeSlot)
        return VoiceResp(transcript=transcript,
                         jp=r["answer"], en="",
                         correction="",
                         answer=r["answer"],
                         recommendation=r.get("recommendation",""))

    # default → dedicated voice prompt
    r = await _voice_reply(uid, transcript)

    # 3️⃣  Neural TTS with OpenAI /audio/speech (≈1 sec)
    speech = await client.audio.speech.create(
        model="tts-1",
        voice="alloy",                    # try "nova" or "shimmer" too
        input=r["jp"]
    )
    mp3_bytes = b"".join([c async for c in (await speech.aiter_bytes())])


    # 4️⃣  persist & serve (simple: local /static; prod: S3 or CDN)
    static_dir = pathlib.Path(__file__).parents[0] / "static/tts"
    static_dir.mkdir(parents=True, exist_ok=True)

    tts_dir = static_root / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{uuid.uuid4()}.mp3"
    fpath = tts_dir / fname
    fpath.write_bytes(mp3_bytes)
    tts_url = f"/static/tts/{fname}"

    return VoiceResp(
        transcript=transcript,
        jp=r["jp"],
        en=r["en"],
        correction=r.get("correction", ""),
        ttsUrl=tts_url
    )

