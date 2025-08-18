# Complete updated main.py with working title generation

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
from datetime import datetime, timedelta, timezone
import calendar, json, zoneinfo

# ───────── external helper libs ──────────
from langdetect import detect              # pip install langdetect
import yake                                 # pip install yake
from rapidfuzz import fuzz                 # pip install rapidfuzz

# ───────── FastAPI boot ─────────
load_dotenv()
client = AsyncOpenAI()
app    = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
RECOMMEND_AFTER_N_HITS  = 2

_GENERIC_TOKENS = {
    "resource","resources","material","materials","book","books",
    "guide","exercise","exercises","practice","lesson","lessons",
    "course","courses","app","website","reference","recommend"
}
_RESOURCE_TRIGGERS = _GENERIC_TOKENS.copy()
_last_topic: dict[str, str] = {}

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

async def _generate_chat_title(user_message: str) -> str:
    """Generate a concise 3-5 word English title for the chat session"""
    try:
        print(f"🔍 Generating title for message: '{user_message}'")
        
        title_prompt = f"""Create a short title (maximum 4 words) in English that describes what this conversation is about:

User message: "{user_message}"

Requirements:
- Maximum 4 words
- English only
- Descriptive but concise
- No quotes or punctuation
- Examples: "Weather Talk", "Food Discussion", "Grammar Help", "Travel Plans"

Title:"""

        resp = await client.chat.completions.create(
            model=MODEL_NAME,  # Use cheaper model for titles
            messages=[{"role": "user", "content": title_prompt}],
            temperature=0.3,
            max_tokens=15
        )
        
        raw_title = resp.choices[0].message.content.strip()
        print(f"🤖 Raw AI response: '{raw_title}'")
        
        # Clean up the title
        title = raw_title.replace('"', '').replace("'", "").replace(".", "").replace(":", "")
        
        # Ensure it's not too long
        words = title.split()
        if len(words) > 4:
            title = " ".join(words[:4])
        
        # Capitalize first letter of each word
        title = " ".join(word.capitalize() for word in title.split())
        
        print(f"✅ Final title: '{title}'")
        return title or "New Chat"
        
    except Exception as e:
        print(f"❌ Title generation failed: {e}")
        return "New Chat"

async def _conversation_reply(uid: str, mode: str, user_msg: str, user_name: str = "フレンド") -> Dict:
    history = get_history(uid)
    template = PROMPTS[mode]
    system_prompt = template.format(USER_NAME=user_name)

    if ctx := _history_context(history):
        system_prompt += "\n\nLast turns:\n" + ctx
    messages = (
        [{"role":"system","content":system_prompt}]
        + history[-HISTORY_WINDOW:]
        + [{"role":"user","content":user_msg}]
    )
    resp = await client.chat.completions.create(model=MODEL_NAME,
            messages=messages,response_format={"type": "json_object"}, temperature=0.7)
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
            messages=messages, response_format={"type": "json_object"}, temperature=0.7)
    assistant = resp.choices[0].message.content.strip()
    write_turns(uid,user_msg,assistant)
    return json.loads(assistant)

async def _voice_reply(uid:str, user_msg:str)->Dict:
    messages = (
        [{"role":"system","content":PROMPTS["voice"]}]
        + get_history(uid)
        + [{"role":"user","content":user_msg}]
    )
    resp = await client.chat.completions.create(model=MODEL_NAME,
            messages=messages, response_format={"type": "json_object"}, temperature=0.7)
    assistant = resp.choices[0].message.content.strip()
    write_turns(uid,user_msg,assistant)
    return json.loads(assistant)

# ===============================================================
# 1. Conversation endpoint
# ===============================================================
class ChatRequest(BaseModel):
    uid: str
    mode: Literal["convFormal","convCasual"]
    userMessage: str
    name: Optional[str] = None

@app.post("/chat")
async def chat(body: ChatRequest):
    uid, mode, user_msg, user_name = body.uid, body.mode, body.userMessage, body.name or "there"
    
    print(f"📨 Chat request - UID: {uid}, Mode: {mode}, Message: '{user_msg}'")
    
    # Check if this is a new session (no history)
    history = get_history(uid)
    is_new_session = len(history) == 0
    
    print(f"🆕 New session: {is_new_session}, History length: {len(history)}")
    
    # Generate title for new sessions
    chat_title = None
    if is_new_session:
        chat_title = await _generate_chat_title(user_msg)
        print(f"🏷️ Generated title: '{chat_title}'")
    
    template = PROMPTS[body.mode]
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
        resp = await client.chat.completions.create(
            model=MODEL_NAME, messages=messages, response_format={"type": "json_object"}, temperature=0.7)
        assistant = resp.choices[0].message.content.strip()
        payload = json.loads(assistant)
        write_turns(uid, user_msg, assistant)
        
        # Include title in response for new sessions
        if chat_title:
            payload["chatTitle"] = chat_title
            print(f"✅ Added title to conversation response: '{chat_title}'")
        else:
            payload["chatTitle"] = None
            print("ℹ️ No title generated (not a new session)")
            
        return JSONResponse(payload)
    except Exception as e:
        print("CHAT ERROR:", e, file=sys.stderr)
        return JSONResponse(status_code=500, content={"detail": str(e)})

# ===============================================================
# 2. Mentor endpoint
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
        clear_pending_calendar_event(body.uid)
        return {"message": "Great! I've queued it on your device's calendar."}
    else:
        clear_pending_calendar_event(body.uid)
        return {"message": "No problem—let me know if you change your mind."}

def _ensure_calendar_prompt(payload: dict, slot_iso: str | None) -> dict:
    if not slot_iso or not payload.get("recommendation"):
        return payload

    friendly = _humanize_slot(slot_iso)
    if "calendar" in payload["answer"].lower():
        return payload

    payload["answer"] += (
        f"\nI noticed you're free {friendly}. "
        "Would you like me to add it to your calendar?"
    )
    return payload

def _humanize_slot(iso: str, duration_min: int = 30) -> str:
    dt = datetime.fromisoformat(iso)
    local = dt.astimezone()
    today = datetime.now(local.tzinfo).date()
    delta = (local.date() - today).days

    if   delta == 0:              day = "today"
    elif delta == 1:              day = "tomorrow"
    elif 2 <= delta <= 6:         day = local.strftime("%A")
    elif 7 <= delta <= 13:        day = "next " + local.strftime("%A")
    else:                         day = local.strftime("%b %d")

    start = local.strftime("%-I:%M %p").lower()
    stop  = (local + timedelta(minutes=duration_min)).strftime("%-I:%M %p").lower()
    return f"{day} at {start}–{stop}"

class MentorReq(BaseModel):
    uid: str
    question: str
    freeSlot: Optional[str] = None
    name: Optional[str] = None

@app.post("/mentor", response_class=StreamingResponse)
async def mentor(body: MentorReq):
    uid, q, free_slot_iso, user_name = body.uid, body.question, body.freeSlot, body.name or "there"
    
    print(f"📨 Mentor request - UID: {uid}, Question: '{q}'")
    
    # Check if this is a new session
    history = get_history(uid)
    is_new_session = len(history) == 0
    
    print(f"🆕 New session: {is_new_session}, History length: {len(history)}")
    
    # Generate title for new sessions
    chat_title = None
    if is_new_session:
        chat_title = await _generate_chat_title(q)
        print(f"🏷️ Generated title: '{chat_title}'")

    # Topic extraction and resource logic
    explicit = _wants_resources(q)
    cand_topic = _extract_topic(q)
    is_generic = cand_topic in _GENERIC_TOKENS
    topic = (_last_topic.get(uid) or last_resource(uid) or cand_topic) if is_generic else cand_topic
    if not is_generic: _last_topic[uid] = topic

    if topic and not is_generic:
        inc_topic(uid, topic)
    hits = topic_hits(uid, topic) if topic else 0

    force_reco = explicit or (hits >= RECOMMEND_AFTER_N_HITS)

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

    # Build system prompt
    template = PROMPTS["mentor"]
    base_prompt = template.format(USER_NAME=user_name)

    sys_prompt = (
        base_prompt
        + "\n\n### RESOURCE_CONTEXT\n"
        + ( _context_snippet(topic) if topic else "" )
        + f"\n\nTOPIC_HITS: {RECOMMEND_AFTER_N_HITS if force_reco else hits}"
        + f"\n\nAVAILABLE_RESOURCES:\n{res_lines}"
    )
    if free_slot_iso:
        pretty_slot = _humanize_slot(free_slot_iso)
        sys_prompt += f"\n\nFREE_SLOT: {pretty_slot}"

    messages = (
        [{"role":"system","content":sys_prompt}]
        + get_history(uid)
        + [{"role":"user","content":q}]
    )

    # Stream response
    async def event_stream():
        answer = ""
        try:
            stream = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                stream=True,
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            async for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                answer += token
                yield f"data: {token}\n\n"

            payload = json.loads(answer)
            payload = _ensure_calendar_prompt(payload, free_slot_iso)
            
            # Include title for new sessions
            if chat_title:
                payload["chatTitle"] = chat_title
                print(f"✅ Added title to mentor response: '{chat_title}'")
            else:
                payload["chatTitle"] = None
                print("ℹ️ No title generated (not a new session)")

            write_turns(uid, q, json.dumps(payload, ensure_ascii=False))
            yield f"event: done\ndata: {json.dumps(payload)}\n\n"

        except Exception as e:
            print(f"❌ Mentor error: {e}")
            yield f"event: error\ndata: {e}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ===============================================================
# 3. Voice chat endpoint
# ===============================================================
class VoiceResp(BaseModel):
    transcript: str
    jp: str
    en: str
    correction: str | None = None
    ttsUrl: str | None = None 
    chatTitle: str | None = None  # This field was missing!
    # legacy keys for current iOS code
    answer: str | None = None
    recommendation: str | None = None

@app.post("/voice_chat", response_model=VoiceResp)
async def voice_chat(
    uid: str = Form(...),
    voiceMode: str = Form(...),
    convSubMode: str|None = Form(None),
    freeSlot: Optional[str] = Form(None),
    audio: UploadFile = File(...)
):
    print(f"📨 Voice request - UID: {uid}, Mode: {voiceMode}")
    
    # Whisper transcription
    audio_bytes = await audio.read()
    txt = await client.audio.transcriptions.create(
        model="whisper-1",
        file=("speech.wav", audio_bytes, "audio/wav"),
        response_format="text")
    transcript = txt.strip()
    
    print(f"🎤 Transcript: '{transcript}'")
    
    # Check if this is a new session
    history = get_history(uid)
    is_new_session = len(history) == 0
    
    print(f"🆕 New session: {is_new_session}, History length: {len(history)}")
    
    # Generate title for new sessions
    chat_title = None
    if is_new_session:
        chat_title = await _generate_chat_title(transcript)
        print(f"🏷️ Generated title: '{chat_title}'")

    # Route based on voice mode
    if voiceMode == "conversation":
        sub = convSubMode if convSubMode in ("convCasual","convFormal") else "convCasual"
        r = await _conversation_reply(uid, sub, transcript)
        return VoiceResp(
            transcript=transcript,
            jp=r["reply"], 
            en="",
            correction="",
            answer=r["reply"],
            chatTitle=chat_title
        )

    if voiceMode == "mentor":
        r = await _mentor_reply(uid, transcript, free_slot_iso=freeSlot)
        return VoiceResp(
            transcript=transcript,
            jp=r["answer"], 
            en="",
            correction="",
            answer=r["answer"],
            recommendation=r.get("recommendation",""),
            chatTitle=chat_title
        )

    # Default voice mode
    r = await _voice_reply(uid, transcript)

    # Neural TTS generation
    speech = await client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=r["jp"]
    )
    mp3_bytes = b"".join([c async for c in (await speech.aiter_bytes())])

    # Save TTS file
    static_dir = pathlib.Path(__file__).parents[0] / "static/tts"
    static_dir.mkdir(parents=True, exist_ok=True)

    tts_dir = static_root / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{uuid.uuid4()}.mp3"
    fpath = tts_dir / fname
    fpath.write_bytes(mp3_bytes)
    tts_url = f"/static/tts/{fname}"

    print(f"✅ Voice response generated with title: '{chat_title}'")

    return VoiceResp(
        transcript=transcript,
        jp=r["jp"],
        en=r["en"],
        correction=r.get("correction", ""),
        ttsUrl=tts_url,
        chatTitle=chat_title
    )