# Complete updated main.py with working title generation

from __future__ import annotations
import pathlib
from typing import Literal, List, Dict, Optional, Tuple
import os, sys, json
import uuid
import random
import hashlib
from pathlib import Path

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
from app.prompts   import PROMPTS, DICTIONARY_PROMPTS, SENTENCE_ANALYSIS_PROMPTS, CONJUGATION_PROMPTS, CROSSWORD_PROMPTS
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

# Add Korean language detection support
def _detect_language(text: str) -> str:
    """Detect if text is Korean or Japanese"""
    try:
        detected = detect(text)
        if detected == 'ko':
            return 'korean'
        elif detected == 'ja':
            return 'japanese'
        else:
            # Fallback: check for Korean/Japanese characters
            has_korean = any('\uac00' <= char <= '\ud7af' for char in text)
            has_japanese = any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in text)
            
            if has_korean:
                return 'korean'
            elif has_japanese:
                return 'japanese'
            else:
                return 'japanese'  # default
    except:
        return 'japanese'  # default

# Update YAKE extractor to support Korean
def _extract_topic_with_language(text: str, language: str) -> str:
    lang_code = "ko" if language == "korean" else "ja"
    if lang_code not in _YAKE:
        _YAKE[lang_code] = yake.KeywordExtractor(lan=lang_code, n=1, dedupLim=0.9, top=1)
    kws = _YAKE[lang_code].extract_keywords(text)
    return (kws[0][0] if kws else text.split()[0]).lower()

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
    mode: Literal["convFormal", "convCasual"]
    userMessage: str
    name: Optional[str] = None
    language: Literal["japanese", "korean"] = "japanese"  # NEW
 

@app.post("/chat")
async def chat(body: ChatRequest):
    uid, mode, user_msg, user_name = body.uid, body.mode, body.userMessage, body.name or "Friend"
    language = body.language  # NEW
    
    # Auto-detect language if not specified
    if not language or language == "japanese":
        detected_lang = _detect_language(user_msg)
        if detected_lang == "korean":
            language = "korean"
    
    print(f"🔨 Chat request - UID: {uid}, Mode: {mode}, Language: {language}, Message: '{user_msg}'")
    
    # Check if this is a new session
    history = get_history(uid, f"{mode}_{language}")  # Separate history per language
    is_new_session = len(history) == 0
    
    # Generate title for new sessions
    chat_title = None
    if is_new_session:
        chat_title = await _generate_chat_title(user_msg)
    
    # Select appropriate prompt template
    prompt_key = f"{language}_{mode}" if language == "korean" else mode
    if prompt_key not in PROMPTS:
        prompt_key = mode  # fallback to Japanese
        
    template = PROMPTS[prompt_key]
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
        write_turns(uid, user_msg, assistant, f"{mode}_{language}")  # Language-specific history
        
        if chat_title:
            payload["chatTitle"] = chat_title
        else:
            payload["chatTitle"] = None
            
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

class MentorRequest(BaseModel):
    uid: str
    question: str
    freeSlot: Optional[str] = None
    name: Optional[str] = None
    language: Literal["japanese", "korean"] = "japanese"  # NEW

@app.post("/mentor", response_class=StreamingResponse)
async def mentor(body: MentorRequest):
    uid, q, free_slot_iso, user_name = body.uid, body.question, body.freeSlot, body.name or "Friend"
    language = body.language
    
    # Auto-detect language
    if not language or language == "japanese":
        detected_lang = _detect_language(q)
        if detected_lang == "korean":
            language = "korean"
    
    print(f"🔨 Mentor request - UID: {uid}, Language: {language}, Question: '{q}'")
    
    # Check if this is a new session  
    history = get_history(uid, f"mentor_{language}")
    is_new_session = len(history) == 0
    
    # Generate title for new sessions
    chat_title = None
    if is_new_session:
        chat_title = await _generate_chat_title(q)
        print(f"🏷️ Generated mentor title: '{chat_title}'")

    # Topic extraction with language support
    explicit = _wants_resources(q)
    cand_topic = _extract_topic_with_language(q, language)
    is_generic = cand_topic in _GENERIC_TOKENS
    topic = (_last_topic.get(uid) or last_resource(uid) or cand_topic) if is_generic else cand_topic
    if not is_generic: _last_topic[uid] = topic

    if topic and not is_generic:
        inc_topic(uid, topic)
    hits = topic_hits(uid, topic) if topic else 0

    force_reco = explicit or (hits >= RECOMMEND_AFTER_N_HITS)

    if force_reco and topic:
        res_list = match_resources(topic)  # Fixed: removed language parameter that doesn't exist
        res_lines = "\n".join(
            f"- {r['title']} ({r['type']}, {r['study_time']} hrs, {r['difficulty']})"
            for r in res_list
        ) or "NONE"
        if res_list: remember_resource(uid, res_list[0]["title"])
    else:
        res_list  = []
        res_lines = "NONE"

    # Build system prompt with language-specific template
    prompt_key = f"{language}_mentor" if language == "korean" else "mentor"
    if prompt_key not in PROMPTS:
        prompt_key = "mentor"  # fallback
        
    template = PROMPTS[prompt_key]
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
        + get_history(uid, f"mentor_{language}")
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

            # Parse the complete JSON response
            payload = json.loads(answer)
            payload = _ensure_calendar_prompt(payload, free_slot_iso)
            
            # ✅ CRITICAL FIX: Always include chatTitle in the payload
            if chat_title:
                payload["chatTitle"] = chat_title
                print(f"✅ Mentor payload includes title: {payload.get('chatTitle')}")
            else:
                payload["chatTitle"] = None
                print("ℹ️ No title generated (not a new session)")

            # Write to history with language-specific mode
            write_turns(uid, q, json.dumps(payload, ensure_ascii=False), f"mentor_{language}")
            
            # Send the final payload with title included
            yield f"event: done\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

        except Exception as e:
            print(f"❌ Mentor error: {e}")
            # Include title even in error response for new sessions
            error_payload = {"error": str(e)}
            if chat_title:
                error_payload["chatTitle"] = chat_title
            yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"

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
    language: str = Form("japanese"),
    freeSlot: Optional[str] = Form(None),
    audio: UploadFile = File(...)
):
    print(f"🔨 Voice request - UID: {uid}, Mode: {voiceMode}, Language: {language}")
    
    # Read and validate audio
    audio_bytes = await audio.read()
    print(f"🎤 Received audio: {len(audio_bytes)} bytes")
    
    if len(audio_bytes) < 8192:
        return VoiceResp(
            transcript="",
            jp="音声が短すぎます。もう少し長く話してください。",
            en="Audio too short. Please speak longer.",
            correction="",
            chatTitle=None
        )
    
    try:
        # Whisper transcription
        whisper_language = "ko" if language == "korean" else "ja"
        
        txt = await client.audio.transcriptions.create(
            model="whisper-1",
            file=("speech.wav", audio_bytes, "audio/wav"),
            response_format="text",
            language=whisper_language
        )
        transcript = txt.strip()
        print(f"🎤 Transcript: '{transcript}'")
        
        # Validate transcript length
        if len(transcript.strip()) < 2:
            return VoiceResp(
                transcript=transcript,
                jp="もう一度、はっきり話してください。",
                en="Please speak more clearly.",
                correction="",
                chatTitle=None
            )
        
        # Auto-detect language if needed
        if language == "japanese":
            detected_lang = _detect_language(transcript)
            if detected_lang == "korean":
                language = "korean"
                print(f"🔄 Auto-detected Korean")
        
        # Session management
        history = get_history(uid, f"voice_{language}")
        is_new_session = len(history) == 0
        
        chat_title = None
        if is_new_session:
            chat_title = await _generate_chat_title(transcript)
        
        # Route to appropriate handler
        if voiceMode == "conversation":
            sub = convSubMode if convSubMode in ("convCasual","convFormal") else "convCasual"
            r = await _conversation_reply(uid, f"{language}_{sub}" if language == "korean" else sub, transcript)
            return VoiceResp(
                transcript=transcript,
                jp=r["reply"],
                en="",
                correction="",
                answer=r["reply"],
                chatTitle=chat_title
            )
        
        elif voiceMode == "mentor":
            r = await _mentor_reply(uid, transcript, language, free_slot_iso=freeSlot)
            return VoiceResp(
                transcript=transcript,
                jp=r["answer"],
                en="",
                correction="",
                answer=r["answer"],
                recommendation=r.get("recommendation",""),
                chatTitle=chat_title
            )
        
        else:  # Default voice mode
            r = await _voice_reply(uid, transcript, language)
            
            # Generate TTS
            speech = await client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=r["jp"]
            )
            mp3_bytes = b"".join([c async for c in (await speech.aiter_bytes())])
            
            # Save TTS file
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
    
    except Exception as e:
        print(f"❌ Voice processing error: {e}")
        error_msg = "申し訳ありません。もう一度お試しください。" if language == "japanese" else "죄송합니다. 다시 시도해 주세요."
        
        return VoiceResp(
            transcript="",
            jp=error_msg,
            en="Sorry, please try again.",
            correction="",
            chatTitle=None
        )

async def _conversation_reply(uid: str, mode: str, user_msg: str, user_name: str = "Friend") -> Dict:
    history = get_history(uid, mode)  # Mode now includes language
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
    write_turns(uid,user_msg,assistant, mode)
    return json.loads(assistant)

async def _mentor_reply(uid:str, user_msg:str, language:str = "japanese", free_slot_iso:str|None=None)->Dict:
    mode = f"mentor_{language}"
    explicit = _wants_resources(user_msg)
    cand = _extract_topic_with_language(user_msg, language)
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
    
    prompt_key = f"{language}_mentor" if language == "korean" else "mentor"
    if prompt_key not in PROMPTS:
        prompt_key = "mentor"
        
    sys_prompt = (PROMPTS[prompt_key]
        + "\n\n### RESOURCE_CONTEXT\n" + (_context_snippet(topic) if topic else "")
        + f"\n\nTOPIC_HITS: {RECOMMEND_AFTER_N_HITS if force else hits}"
        + f"\n\nAVAILABLE_RESOURCES:\n{res_lines}")
    if free_slot_iso: sys_prompt += f"\n\nFREE_SLOT: {free_slot_iso}"
    messages = (
        [{"role":"system","content":sys_prompt}]
        + get_history(uid, mode)
        + [{"role":"user","content":user_msg}]
    )
    resp = await client.chat.completions.create(model=MODEL_NAME,
            messages=messages, response_format={"type": "json_object"}, temperature=0.7)
    assistant = resp.choices[0].message.content.strip()
    write_turns(uid,user_msg,assistant, mode)
    return json.loads(assistant)

async def _voice_reply(uid:str, user_msg:str, language:str = "japanese")->Dict:
    mode = f"voice_{language}"
    prompt_key = f"{language}_voice" if language == "korean" else "voice"
    if prompt_key not in PROMPTS:
        prompt_key = "voice"
        
    messages = (
        [{"role":"system","content":PROMPTS[prompt_key]}]
        + get_history(uid, mode)
        + [{"role":"user","content":user_msg}]
    )
    resp = await client.chat.completions.create(model=MODEL_NAME,
            messages=messages, response_format={"type": "json_object"}, temperature=0.7)
    assistant = resp.choices[0].message.content.strip()
    write_turns(uid,user_msg,assistant, mode)
    return json.loads(assistant)


# ===============================================================
# 4. Dictionary endpoint
# ===============================================================

class DictionaryRequest(BaseModel):
    uid: str
    word: str
    language: Literal["japanese", "korean"] = "japanese"

class DictionaryResponse(BaseModel):
    word: str
    reading: Optional[str] = None
    level: Optional[str] = None
    meanings: List[str]
    part_of_speech: Optional[str] = None
    kanji_breakdown: Optional[Dict] = None
    hangul_breakdown: Optional[Dict] = None
    example_sentences: List[Dict] = []
    conjugations: Optional[List[Dict]] = None
    found: bool = True
    error: Optional[str] = None

# In-memory cache for dictionary lookups
_dictionary_cache: Dict[str, Dict] = {}

def _cache_key(word: str, language: str) -> str:
    return f"{language}:{word.lower().strip()}"

def _increment_dictionary_stat(uid: str):
    """Increment user's dictionary lookup counter"""
    try:
        # This would integrate with your existing stats system
        # For now, just print - you can connect to Firestore later
        print(f"📊 Dictionary lookup count incremented for user: {uid}")
    except Exception as e:
        print(f"Warning: Failed to increment dictionary stats: {e}")


@app.post("/dictionary", response_model=DictionaryResponse)
async def dictionary_lookup(body: DictionaryRequest):
    uid, word, language = body.uid, body.word.strip(), body.language
    
    if not word:
        return DictionaryResponse(
            word="",
            found=False,
            error="Empty search query",
            meanings=[]
        )
    
    print(f"📚 Dictionary lookup - UID: {uid}, Language: {language}, Word: '{word}'")
    
    # Check cache first
    cache_key = _cache_key(word, language)
    if cache_key in _dictionary_cache:
        print(f"💾 Cache hit for: {word}")
        cached_result = _dictionary_cache[cache_key]
        _increment_dictionary_stat(uid)
        return DictionaryResponse(**cached_result)
    
    try:
        # Get the appropriate prompt template
        prompt_template = DICTIONARY_PROMPTS.get(language, DICTIONARY_PROMPTS["japanese"])
        prompt = prompt_template.format(word=word)
        
        # Call OpenAI with explicit JSON mode instructions
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a dictionary API. You MUST respond with valid JSON only. Do not use markdown formatting, code blocks, or any other formatting. Return raw JSON that can be parsed directly."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.1,  # Very low temperature for consistent JSON
            max_tokens=1000,
            response_format={"type": "json_object"}  # Force JSON mode
        )
        
        result_text = response.choices[0].message.content.strip()
        print(f"🤖 AI Response length: {len(result_text)} chars")
        
        # Clean up any potential markdown formatting
        if result_text.startswith('```json'):
            result_text = result_text[7:]  # Remove ```json
        if result_text.endswith('```'):
            result_text = result_text[:-3]  # Remove ```
        
        result_text = result_text.strip()
        
        # Additional cleaning - remove any remaining markdown
        import re
        result_text = re.sub(r'^```\w*\n?', '', result_text)
        result_text = re.sub(r'\n?```$', '', result_text)
        
        print(f"🧹 Cleaned response: {result_text[:100]}...")
        
        try:
            # Parse the JSON response
            result_data = json.loads(result_text)
            
            # Validate required fields
            if "meanings" not in result_data:
                result_data["meanings"] = []
            if "found" not in result_data:
                result_data["found"] = len(result_data["meanings"]) > 0
                
            # Cache the successful result
            _dictionary_cache[cache_key] = result_data
            
            # Increment user stats
            _increment_dictionary_stat(uid)
            
            print(f"✅ Dictionary lookup successful for: {word}")
            return DictionaryResponse(**result_data)
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            print(f"Raw response: {result_text[:200]}...")
            
            # Try to extract JSON from malformed response
            try:
                # Look for JSON object boundaries
                start = result_text.find('{')
                end = result_text.rfind('}') + 1
                
                if start >= 0 and end > start:
                    json_part = result_text[start:end]
                    print(f"🔧 Attempting to parse extracted JSON: {json_part[:100]}...")
                    result_data = json.loads(json_part)
                    
                    # Validate and cache
                    if "meanings" not in result_data:
                        result_data["meanings"] = []
                    if "found" not in result_data:
                        result_data["found"] = len(result_data["meanings"]) > 0
                    
                    _dictionary_cache[cache_key] = result_data
                    _increment_dictionary_stat(uid)
                    
                    print(f"✅ Dictionary lookup successful after JSON extraction for: {word}")
                    return DictionaryResponse(**result_data)
                    
            except json.JSONDecodeError:
                pass
            
            return DictionaryResponse(
                word=word,
                found=False,
                error="Failed to parse dictionary response",
                meanings=[]
            )
    
    except Exception as e:
        print(f"❌ Dictionary lookup failed: {e}")
        return DictionaryResponse(
            word=word,
            found=False,
            error=f"Dictionary service error: {str(e)}",
            meanings=[]
        )
    
# Optional: Dictionary cache management endpoints
@app.get("/dictionary/cache/stats")
async def dictionary_cache_stats():
    """Get dictionary cache statistics"""
    return {
        "total_cached_entries": len(_dictionary_cache),
        "cache_keys": list(_dictionary_cache.keys())[:10],  # Show first 10 keys
        "memory_usage_mb": round(sys.getsizeof(_dictionary_cache) / 1024 / 1024, 2)
    }

@app.delete("/dictionary/cache")
async def clear_dictionary_cache():
    """Clear the dictionary cache"""
    global _dictionary_cache
    cache_size = len(_dictionary_cache)
    _dictionary_cache.clear()
    return {"message": f"Cleared {cache_size} cached entries"}

class SentenceAnalysisRequest(BaseModel):
    uid: str
    sentence: str
    language: Literal["japanese", "korean"] = "japanese"
    source_language: Literal["english", "japanese", "korean"] = "english"  # What language user typed in

class SentenceAnalysisResponse(BaseModel):
    # Core analysis
    correctness_score: int  # 0-100
    grammar_score: int
    particle_score: int
    word_usage_score: int
    spelling_score: int
    kanji_usage_score: Optional[int] = None  # Japanese only
    
    # User's input analysis
    user_sentence_breakdown: List[Dict]  # Word-by-word analysis
    user_meaning: str  # What the user actually said
    user_highlighted_errors: str  # HTML with highlighted errors
    
    # AI corrections
    ai_corrected_sentence: str  # Fixed version
    ai_meaning: str  # What the corrected sentence means
    ai_corrections_highlighted: str  # HTML showing changes
    
    # Explanations
    corrections_explanation: List[Dict]  # Detailed explanations of each change
    found: bool = True
    error: Optional[str] = None



@app.post("/analyze_sentence", response_model=SentenceAnalysisResponse)
async def analyze_sentence(body: SentenceAnalysisRequest):
    uid, sentence, language = body.uid, body.sentence.strip(), body.language
    source_language = body.source_language
    
    # Input validation
    if not sentence:
        return SentenceAnalysisResponse(
            correctness_score=0, grammar_score=0, particle_score=0,
            word_usage_score=0, spelling_score=0,
            user_sentence_breakdown=[], user_meaning="", user_highlighted_errors="",
            ai_corrected_sentence="", ai_meaning="", ai_corrections_highlighted="",
            corrections_explanation=[], found=False, error="Empty sentence"
        )
    
    # Character/word limit (prevent abuse)
    max_chars = 500
    if len(sentence) > max_chars:
        return SentenceAnalysisResponse(
            correctness_score=0, grammar_score=0, particle_score=0,
            word_usage_score=0, spelling_score=0,
            user_sentence_breakdown=[], user_meaning="", user_highlighted_errors="",
            ai_corrected_sentence="", ai_meaning="", ai_corrections_highlighted="",
            corrections_explanation=[], found=False, 
            error=f"Sentence too long. Maximum {max_chars} characters allowed."
        )
    
    print(f"📝 Sentence analysis - UID: {uid}, Language: {language}, Sentence: '{sentence}'")
    
    try:
        prompt_template = SENTENCE_ANALYSIS_PROMPTS.get(language, SENTENCE_ANALYSIS_PROMPTS["japanese"])
        prompt = prompt_template.format(sentence=sentence, source_language=source_language)
        
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a language analysis API. Respond with valid JSON only."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Clean any markdown formatting
        if result_text.startswith('```json'):
            result_text = result_text[7:]
        if result_text.endswith('```'):
            result_text = result_text[:-3]
        result_text = result_text.strip()
        
        try:
            result_data = json.loads(result_text)
            
            # Validate required fields and set defaults
            required_fields = {
                "correctness_score": 0, "grammar_score": 0, "particle_score": 0,
                "word_usage_score": 0, "spelling_score": 0, "user_sentence_breakdown": [],
                "user_meaning": "", "user_highlighted_errors": "",
                "ai_corrected_sentence": "", "ai_meaning": "", "ai_corrections_highlighted": "",
                "corrections_explanation": []
            }
            
            for field, default in required_fields.items():
                if field not in result_data:
                    result_data[field] = default
            
            print(f"✅ Sentence analysis successful for: {sentence}")
            return SentenceAnalysisResponse(**result_data)
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            return SentenceAnalysisResponse(
                correctness_score=0, grammar_score=0, particle_score=0,
                word_usage_score=0, spelling_score=0, user_sentence_breakdown=[],
                user_meaning="", user_highlighted_errors="", ai_corrected_sentence="",
                ai_meaning="", ai_corrections_highlighted="", corrections_explanation=[],
                found=False, error="Failed to parse analysis response"
            )
    
    except Exception as e:
        print(f"❌ Sentence analysis failed: {e}")
        return SentenceAnalysisResponse(
            correctness_score=0, grammar_score=0, particle_score=0,
            word_usage_score=0, spelling_score=0, user_sentence_breakdown=[],
            user_meaning="", user_highlighted_errors="", ai_corrected_sentence="",
            ai_meaning="", ai_corrections_highlighted="", corrections_explanation=[],
            found=False, error=f"Analysis service error: {str(e)}"
        )
    

# ===============================================================
# 5. Sentence Analysis endpoint
# ===============================================================


class GrammarBreakdown(BaseModel):
    grammar: int        # 0-100
    particles: int      # 0-100
    wordUsage: int      # 0-100
    spelling: int       # 0-100
    kanjiUsage: Optional[int] = None    # Optional for Japanese
    honorifics: Optional[int] = None    # Optional for Korean

class WordAnalysis(BaseModel):
    word: str
    reading: Optional[str] = None
    partOfSpeech: str
    meaning: str
    usage: str
    isCorrect: bool
    correction: Optional[str] = None
    position: int       # Position in sentence for highlighting

class Improvement(BaseModel):
    type: str        # "grammar", "particle", "word_choice", etc.
    explanation: str
    original: str
    corrected: str

class SentenceAnalysisResponse(BaseModel):
    originalSentence: str
    correctnessScore: int # 0-100
    grammarBreakdown: GrammarBreakdown
    userTranslation: str
    correctedSentence: str
    correctedTranslation: str
    improvements: List[Improvement]
    wordAnalysis: List[WordAnalysis]
    found: bool = True
    error: Optional[str] = None

# Sentence analysis cache
_sentence_analysis_cache: Dict[str, Dict] = {}

def _sentence_cache_key(sentence: str, language: str) -> str:
    return f"{language}:{sentence.lower().strip()}"

@app.post("/sentence_analysis", response_model=SentenceAnalysisResponse)
async def analyze_sentence(body: SentenceAnalysisRequest):
    uid, sentence, language = body.uid, body.sentence.strip(), body.language
    
    # Input validation
    if not sentence:
        return SentenceAnalysisResponse(
            originalSentence="",
            correctnessScore=0,
            grammarBreakdown=GrammarBreakdown(grammar=0, particles=0, wordUsage=0, spelling=0),
            userTranslation="",
            correctedSentence="",
            correctedTranslation="",
            improvements=[],
            wordAnalysis=[],
            found=False,
            error="Empty sentence"
        )
    
    # Character limit (prevent abuse)
    max_chars = 500
    if len(sentence) > max_chars:
        return SentenceAnalysisResponse(
            originalSentence=sentence,
            correctnessScore=0,
            grammarBreakdown=GrammarBreakdown(grammar=0, particles=0, wordUsage=0, spelling=0),
            userTranslation="",
            correctedSentence="",
            correctedTranslation="",
            improvements=[],
            wordAnalysis=[],
            found=False,
            error=f"Sentence too long. Maximum {max_chars} characters allowed."
        )
    
    print(f"📝 Sentence analysis - UID: {uid}, Language: {language}, Sentence: '{sentence}'")
    
    # Check cache first
    cache_key = _sentence_cache_key(sentence, language)
    if cache_key in _sentence_analysis_cache:
        print(f"💾 Cache hit for sentence: {sentence}")
        cached_result = _sentence_analysis_cache[cache_key]
        return SentenceAnalysisResponse(**cached_result)
    
    try:
        # Get the appropriate prompt template
        prompt_template = SENTENCE_ANALYSIS_PROMPTS.get(language, SENTENCE_ANALYSIS_PROMPTS["japanese"])
        prompt = prompt_template.format(sentence=sentence)
        
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a language analysis API. Respond with valid JSON only. Do not use markdown or code blocks."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.1,  # Low temperature for consistent analysis
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Clean any potential markdown formatting
        if result_text.startswith('```json'):
            result_text = result_text[7:]
        if result_text.endswith('```'):
            result_text = result_text[:-3]
        result_text = result_text.strip()
        
        try:
            # Parse the JSON response
            result_data = json.loads(result_text)
            
            # Transform the data to match our response model
            transformed_data = {
                "originalSentence": sentence,
                "correctnessScore": result_data.get("correctness_score", 0),
                "grammarBreakdown": {
                    "grammar": result_data.get("grammar_score", 0),
                    "particles": result_data.get("particle_score", 0),
                    "wordUsage": result_data.get("word_usage_score", 0),
                    "spelling": result_data.get("spelling_score", 0),
                    "kanjiUsage": result_data.get("kanji_usage_score") if language == "japanese" else None,
                    "honorifics": result_data.get("honorifics_score") if language == "korean" else None
                },
                "userTranslation": result_data.get("user_meaning", ""),
                "correctedSentence": result_data.get("corrected_sentence", ""),
                "correctedTranslation": result_data.get("corrected_meaning", ""),
                "improvements": [
                    {
                        "type": imp.get("type", "general"),
                        "explanation": imp.get("explanation", ""),
                        "original": imp.get("original", ""),
                        "corrected": imp.get("corrected", "")
                    }
                    for imp in result_data.get("improvements", [])
                ],
                "wordAnalysis": [
                    {
                        "word": word.get("word", ""),
                        "reading": word.get("reading"),
                        "partOfSpeech": word.get("part_of_speech", ""),
                        "meaning": word.get("meaning", ""),
                        "usage": word.get("usage_note", ""),
                        "isCorrect": word.get("is_correct", True),
                        "correction": word.get("correction"),
                        "position": word.get("position", 0)
                    }
                    for word in result_data.get("word_analysis", [])
                ],
                "found": True
            }
            
            # Cache the result
            _sentence_analysis_cache[cache_key] = transformed_data
            
            print(f"✅ Sentence analysis successful: {transformed_data['correctnessScore']}% correct")
            return SentenceAnalysisResponse(**transformed_data)
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            print(f"Raw response: {result_text[:200]}...")
            
            return SentenceAnalysisResponse(
                originalSentence=sentence,
                correctnessScore=0,
                grammarBreakdown=GrammarBreakdown(grammar=0, particles=0, wordUsage=0, spelling=0),
                userTranslation="",
                correctedSentence="",
                correctedTranslation="",
                improvements=[],
                wordAnalysis=[],
                found=False,
                error="Failed to parse analysis response"
            )
    
    except Exception as e:
        print(f"❌ Sentence analysis failed: {e}")
        return SentenceAnalysisResponse(
            originalSentence=sentence,
            correctnessScore=0,
            grammarBreakdown=GrammarBreakdown(grammar=0, particles=0, wordUsage=0, spelling=0),
            userTranslation="",
            correctedSentence="",
            correctedTranslation="",
            improvements=[],
            wordAnalysis=[],
            found=False,
            error=f"Analysis service error: {str(e)}"
        )

# Cache management endpoints for sentence analysis
@app.get("/sentence_analysis/cache/stats")
async def sentence_analysis_cache_stats():
    """Get sentence analysis cache statistics"""
    return {
        "total_cached_entries": len(_sentence_analysis_cache),
        "cache_keys": list(_sentence_analysis_cache.keys())[:5],
        "memory_usage_mb": round(sys.getsizeof(_sentence_analysis_cache) / 1024 / 1024, 2)
    }

@app.delete("/sentence_analysis/cache")
async def clear_sentence_analysis_cache():
    """Clear the sentence analysis cache"""
    global _sentence_analysis_cache
    cache_size = len(_sentence_analysis_cache)
    _sentence_analysis_cache.clear()
    return {"message": f"Cleared {cache_size} cached sentence analysis entries"}

class EnhancedSentenceAnalysisRequest(BaseModel):
    uid: str
    sentence: str
    intended_english_meaning: str
    language: Literal["japanese", "korean"] = "japanese"
    analysis_type: str = "enhanced_with_context"

@app.post("/sentence_analysis_enhanced", response_model=SentenceAnalysisResponse)
async def analyze_sentence_enhanced(body: EnhancedSentenceAnalysisRequest):
    uid = body.uid
    sentence = body.sentence.strip()
    intended_meaning = body.intended_english_meaning.strip()
    language = body.language
    
    # Input validation
    if not sentence:
        return SentenceAnalysisResponse(
            originalSentence="",
            correctnessScore=0,
            grammarBreakdown=GrammarBreakdown(grammar=0, particles=0, wordUsage=0, spelling=0),
            userTranslation="",
            correctedSentence="",
            correctedTranslation="",
            improvements=[],
            wordAnalysis=[],
            found=False,
            error="Empty sentence"
        )
    
    if not intended_meaning:
        return SentenceAnalysisResponse(
            originalSentence=sentence,
            correctnessScore=0,
            grammarBreakdown=GrammarBreakdown(grammar=0, particles=0, wordUsage=0, spelling=0),
            userTranslation="",
            correctedSentence="",
            correctedTranslation="",
            improvements=[],
            wordAnalysis=[],
            found=False,
            error="Empty intended meaning"
        )
    
    # Character limits
    if len(sentence) > 500:
        return SentenceAnalysisResponse(
            originalSentence=sentence,
            correctnessScore=0,
            grammarBreakdown=GrammarBreakdown(grammar=0, particles=0, wordUsage=0, spelling=0),
            userTranslation="",
            correctedSentence="",
            correctedTranslation="",
            improvements=[],
            wordAnalysis=[],
            found=False,
            error="Sentence too long. Maximum 500 characters allowed."
        )
    
    if len(intended_meaning) > 200:
        return SentenceAnalysisResponse(
            originalSentence=sentence,
            correctnessScore=0,
            grammarBreakdown=GrammarBreakdown(grammar=0, particles=0, wordUsage=0, spelling=0),
            userTranslation="",
            correctedSentence="",
            correctedTranslation="",
            improvements=[],
            wordAnalysis=[],
            found=False,
            error="Intended meaning too long. Maximum 200 characters allowed."
        )
    
    print(f"🔍 Enhanced sentence analysis - UID: {uid}, Language: {language}")
    print(f"   Sentence: '{sentence}'")
    print(f"   Intended: '{intended_meaning}'")
    
    try:
        # Create enhanced prompt with user's intended meaning
        base_prompt_template = SENTENCE_ANALYSIS_PROMPTS.get(language, SENTENCE_ANALYSIS_PROMPTS["japanese"])
        
        # Enhanced prompt that includes user's intended English meaning
        enhanced_prompt = f"""
{base_prompt_template}

ADDITIONAL CONTEXT:
The user has clarified that they intended to express: "{intended_meaning}"

Please analyze their sentence "{sentence}" with this context in mind:
1. Compare what they actually wrote vs. what they intended to say
2. Focus on how to help them express their intended meaning correctly
3. Provide specific guidance on how to say "{intended_meaning}" properly in {language}
4. In your analysis, acknowledge their intended meaning and show the gap between intention and execution

Your corrected_sentence should express: "{intended_meaning}"
Your corrected_meaning should be: "{intended_meaning}" (or very close to it)
"""
        
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a language analysis API with access to user intent. Respond with valid JSON only."
                },
                {
                    "role": "user", 
                    "content": enhanced_prompt.format(sentence=sentence)
                }
            ],
            temperature=0.1,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Clean any potential markdown formatting
        if result_text.startswith('```json'):
            result_text = result_text[7:]
        if result_text.endswith('```'):
            result_text = result_text[:-3]
        result_text = result_text.strip()
        
        try:
            result_data = json.loads(result_text)
            
            # Transform the data to match our response model
            transformed_data = {
                "originalSentence": sentence,
                "correctnessScore": result_data.get("correctness_score", 0),
                "grammarBreakdown": {
                    "grammar": result_data.get("grammar_score", 0),
                    "particles": result_data.get("particle_score", 0),
                    "wordUsage": result_data.get("word_usage_score", 0),
                    "spelling": result_data.get("spelling_score", 0),
                    "kanjiUsage": result_data.get("kanji_usage_score") if language == "japanese" else None,
                    "honorifics": result_data.get("honorifics_score") if language == "korean" else None
                },
                "userTranslation": f"You intended: '{intended_meaning}' but your sentence suggests: '{result_data.get('user_meaning', '')}'",
                "correctedSentence": result_data.get("corrected_sentence", ""),
                "correctedTranslation": intended_meaning,  # Should match user's intended meaning
                "improvements": [
                    {
                        "type": imp.get("type", "general"),
                        "explanation": imp.get("explanation", ""),
                        "original": imp.get("original", ""),
                        "corrected": imp.get("corrected", "")
                    }
                    for imp in result_data.get("improvements", [])
                ],
                "wordAnalysis": [
                    {
                        "word": word.get("word", ""),
                        "reading": word.get("reading"),
                        "partOfSpeech": word.get("part_of_speech", ""),
                        "meaning": word.get("meaning", ""),
                        "usage": word.get("usage_note", ""),
                        "isCorrect": word.get("is_correct", True),
                        "correction": word.get("correction"),
                        "position": word.get("position", 0)
                    }
                    for word in result_data.get("word_analysis", [])
                ],
                "found": True
            }
            
            print(f"✅ Enhanced analysis successful: {transformed_data['correctnessScore']}% correct")
            print(f"   Intended vs actual meaning alignment improved")
            
            return SentenceAnalysisResponse(**transformed_data)
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            print(f"Raw response: {result_text[:200]}...")
            
            return SentenceAnalysisResponse(
                originalSentence=sentence,
                correctnessScore=0,
                grammarBreakdown=GrammarBreakdown(grammar=0, particles=0, wordUsage=0, spelling=0),
                userTranslation=f"Intended: '{intended_meaning}' (analysis failed)",
                correctedSentence="",
                correctedTranslation="",
                improvements=[],
                wordAnalysis=[],
                found=False,
                error="Failed to parse enhanced analysis response"
            )
    
    except Exception as e:
        print(f"❌ Enhanced sentence analysis failed: {e}")
        return SentenceAnalysisResponse(
            originalSentence=sentence,
            correctnessScore=0,
            grammarBreakdown=GrammarBreakdown(grammar=0, particles=0, wordUsage=0, spelling=0),
            userTranslation=f"Intended: '{intended_meaning}' (service error)",
            correctedSentence="",
            correctedTranslation="",
            improvements=[],
            wordAnalysis=[],
            found=False,
            error=f"Enhanced analysis service error: {str(e)}"
        )

# ===============================================================
# 6. Conjugation Analysis endpoint
# ===============================================================

class ConjugationRequest(BaseModel):
    uid: str
    word: str
    language: Literal["japanese", "korean"] = "japanese"

class ConjugationForm(BaseModel):
    formName: str
    conjugated: str
    reading: Optional[str] = None
    romanization: Optional[str] = None
    explanation: str
    usage: str
    politenessLevel: Optional[str] = None  # casual, polite, formal

class VerbInfo(BaseModel):
    baseForm: str
    reading: Optional[str] = None
    romanization: Optional[str] = None
    meaning: str
    verbType: str  # "regular verb", "irregular verb", "adjective", etc.
    conjugationGroup: Optional[str] = None  # "ichidan", "godan", "irregular" for Japanese
    isConjugated: bool
    originalInput: str

class ConjugationResponse(BaseModel):
    verbInfo: VerbInfo
    conjugations: Dict[str, List[ConjugationForm]]  # grouped by category
    found: bool = True
    error: Optional[str] = None

# Conjugation analysis cache
_conjugation_cache: Dict[str, Dict] = {}

def _conjugation_cache_key(word: str, language: str) -> str:
    return f"{language}:conjugation:{word.lower().strip()}"

@app.post("/conjugation", response_model=ConjugationResponse)
async def analyze_conjugation(body: ConjugationRequest):
    uid, word, language = body.uid, body.word.strip(), body.language
    
    if not word:
        return ConjugationResponse(
            verbInfo=VerbInfo(
                baseForm="", reading=None, romanization=None, meaning="", 
                verbType="", isConjugated=False, originalInput=""
            ),
            conjugations={},
            found=False,
            error="Empty word provided"
        )
    
    print(f"🔄 Conjugation analysis - UID: {uid}, Language: {language}, Word: '{word}'")
    
    # Check cache first
    cache_key = _conjugation_cache_key(word, language)
    if cache_key in _conjugation_cache:
        print(f"💾 Cache hit for conjugation: {word}")
        cached_result = _conjugation_cache[cache_key]
        return ConjugationResponse(**cached_result)
    
    try:
        # Get the appropriate prompt template
        prompt_template = CONJUGATION_PROMPTS.get(language, CONJUGATION_PROMPTS["japanese"])
        prompt = prompt_template.format(word=word)
        
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a language conjugation expert. Respond with valid JSON only."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=2500,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Clean any potential markdown formatting
        if result_text.startswith('```json'):
            result_text = result_text[7:]
        if result_text.endswith('```'):
            result_text = result_text[:-3]
        result_text = result_text.strip()
        
        try:
            result_data = json.loads(result_text)
            
            # Validate required fields
            if "verb_info" not in result_data:
                result_data["verb_info"] = {
                    "base_form": word,
                    "meaning": "Unknown",
                    "verb_type": "Unknown",
                    "is_conjugated": False,
                    "original_input": word
                }
            if "conjugations" not in result_data:
                result_data["conjugations"] = {}
            if "found" not in result_data:
                result_data["found"] = True
            
            # Transform to match our response model
            transformed_data = {
                "verbInfo": {
                    "baseForm": result_data["verb_info"].get("base_form", word),
                    "reading": result_data["verb_info"].get("reading"),
                    "romanization": result_data["verb_info"].get("romanization"),
                    "meaning": result_data["verb_info"].get("meaning", "Unknown"),
                    "verbType": result_data["verb_info"].get("verb_type", "Unknown"),
                    "conjugationGroup": result_data["verb_info"].get("conjugation_group"),
                    "isConjugated": result_data["verb_info"].get("is_conjugated", False),
                    "originalInput": word
                },
                "conjugations": {},
                "found": result_data.get("found", True)
            }
            
            # Transform conjugations
            for category, forms in result_data.get("conjugations", {}).items():
                transformed_forms = []
                for form in forms:
                    transformed_forms.append({
                        "formName": form.get("form_name", ""),
                        "conjugated": form.get("conjugated", ""),
                        "reading": form.get("reading"),
                        "romanization": form.get("romanization"),
                        "explanation": form.get("explanation", ""),
                        "usage": form.get("usage", ""),
                        "politenessLevel": form.get("politeness_level")
                    })
                transformed_data["conjugations"][category] = transformed_forms
            
            # Cache the result
            _conjugation_cache[cache_key] = transformed_data
            
            print(f"✅ Conjugation analysis successful for: {word}")
            return ConjugationResponse(**transformed_data)
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            print(f"Raw response: {result_text[:200]}...")
            
            return ConjugationResponse(
                verbInfo=VerbInfo(
                    baseForm=word, reading=None, romanization=None, meaning="Unknown", 
                    verbType="Unknown", isConjugated=False, originalInput=word
                ),
                conjugations={},
                found=False,
                error="Failed to parse conjugation response"
            )
    
    except Exception as e:
        print(f"❌ Conjugation analysis failed: {e}")
        return ConjugationResponse(
            verbInfo=VerbInfo(
                baseForm=word, reading=None, romanization=None, meaning="Unknown", 
                verbType="Unknown", isConjugated=False, originalInput=word
            ),
            conjugations={},
            found=False,
            error=f"Conjugation service error: {str(e)}"
        )

# Cache management endpoints for conjugation analysis
@app.get("/conjugation/cache/stats")
async def conjugation_cache_stats():
    """Get conjugation cache statistics"""
    return {
        "total_cached_entries": len(_conjugation_cache),
        "cache_keys": list(_conjugation_cache.keys())[:5],
        "memory_usage_mb": round(sys.getsizeof(_conjugation_cache) / 1024 / 1024, 2)
    }

@app.delete("/conjugation/cache")
async def clear_conjugation_cache():
    """Clear the conjugation cache"""
    global _conjugation_cache
    cache_size = len(_conjugation_cache)
    _conjugation_cache.clear()
    return {"message": f"Cleared {cache_size} cached conjugation entries"}

# ===============================================================
# 7. Crossword  —  UPDATED
# ===============================================================

# Add these imports at the top if not already present
from typing import Tuple, List, Dict, Literal
import random
import json
from pydantic import BaseModel

# ───────────────────────── Models ─────────────────────────
class CrosswordWord(BaseModel):
    word: str
    clue_original: str
    clue_english: str
    hints: List[str]
    difficulty: str
    start_row: int
    start_col: int
    direction: str  # "across" | "down"
    number: int

class CrosswordPuzzle(BaseModel):
    id: str
    date: str
    language: str
    words: List[CrosswordWord]
    grid_size: List[int]         # [rows, cols]
    grid: List[List[str]]        # "." for blocks, 1-char strings for letters

class CrosswordRequest(BaseModel):
    uid: str
    date: str
    language: Literal["japanese", "korean"] = "japanese"


from itertools import combinations
import math

# ───── Local fallback corpora (4–6 length for JP; 4–6 syllables for KR) ─────
FALLBACK_JA = [
    "ひこうき","こうえん","がっこう","すいえい","じてんしゃ","けいたい","なつやすみ","たいふう",
    "どうぶつ","きょうしつ","しんかんせん","きっさてん","にんじん","すいぞくかん","たいいく",
    "えんそく","しょうがつ","おとしだま","ゆうびんきょく","りょうり","にっき","びよういん",
    "でんきゅう","すいどう","じどうしゃ","てんらんかい","こうさてん","おんがくかい","こくばん",
    "せんせい","ともだち","けんがく","ぶんぼうぐ","きょうかしょ","たいいくかん","こうつうあんぜん"
]

FALLBACK_KO = [
    "놀이공원","지하철역","버스정류장","휴대전화","전화번호","초등학교","고등학교","어린이집",
    "유치원생","자동판매기","세탁기방","보건소장","영화관람","과학시간","문예활동","체육대회",
    "도서관장","운동경기","생활안전","교통안전","자전거도로","축구경기장","농구경기장","수영수업",
    "피아노학원","미술학원","컴퓨터실","우편번호","일기예보","환경보호","동물병원","소방서장"
]

def _has_middle_overlap(a: str, b: str) -> bool:
    if len(a) <= 2 or len(b) <= 2:
        return False
    return bool(set(a[1:-1]) & set(b[1:-1]))

def _valid_word(w: str, language: str, min_len=4, max_len=6) -> bool:
    if not isinstance(w, str): return False
    w = w.strip()
    if not (min_len <= len(w) <= max_len): return False
    if w[0] == w[-1]: return False
    # crude particle filter (same as your validate_words)
    if language == "japanese":
        invalid = ["の","が","を","に","へ","と","や","から","まで","より","で","は"]
    else:
        invalid = ["의","가","을","를","에","에서","으로","로","와","과","이","은","는","도"]
    if any(w.endswith(p) for p in invalid): return False
    if sum(1 for p in invalid if p in w) > 1: return False
    return True

def _pick_starter_trio(pool: list[str], language: str) -> list[str] | None:
    """
    Pick 3 words such that:
      - each is valid (len, no particles, first!=last)
      - at least the second intersects the first by a MIDDLE char
      - the third intersects the first or second by a MIDDLE char
    Prefer higher total middle-overlap 'score'.
    """
    # Filter valid first
    cand = [w for w in pool if _valid_word(w, language, 4, 6)]
    best = None
    for a, b, c in combinations(cand, 3):
        # enforce middle intersections chain
        ab = _has_middle_overlap(a, b)
        ac = _has_middle_overlap(a, c)
        bc = _has_middle_overlap(b, c)
        # Need at least a connected shape: (ab and (ac or bc)) or ((ac and bc))
        connected = (ab and (ac or bc)) or (ac and bc)
        if not connected:
            continue
        # score = total distinct middle overlaps
        score = 0
        if ab: score += 1
        if ac: score += 1
        if bc: score += 1
        # prefer diverse middles (more unique middle chars)
        mid_chars = set(a[1:-1]) | set(b[1:-1]) | set(c[1:-1])
        diversity = len(mid_chars)
        key = (score, diversity)
        if best is None or key > best[0]:
            best = (key, [a, b, c])
    return best[1] if best else None

def _fill_intersecting(existing: list[str], language: str, target_total: int, pool: list[str]) -> list[str]:
    added = []
    used = set(existing)
    def intersects_mid(w: str) -> bool:
        return any(_has_middle_overlap(w, x) for x in existing + added)

    for w in pool:
        if len(existing) + len(added) >= target_total: break
        if w in used: continue
        if not _valid_word(w, language, 3, 8): continue
        if intersects_mid(w):
            added.append(w)
            used.add(w)
    return added

# ─────────────────────── Generation ───────────────────────
async def generate_crossword_vocabulary(language: str, date: str) -> List[str]:
    """Robust generation: try LLM; if no trio, expand pool; then fall back to local corpus. Always returns 10."""
    rng = random.Random(f"{date}:{language}:gen")

    # ---------- Step A: Try to get a valid starter trio from the LLM ----------
    max_retries = 6
    starters: list[str] | None = None

    # 1) Small ask first (exactly 3), pick trio if already valid+connected
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Generate REAL, COMMON words. First and last MUST differ. No particles."},
                    {"role": "user", "content": (
                        # In generate_crossword_vocabulary, update the initial 3-word request:
f"Generate EXACTLY 3 SIMPLE, BASIC {('Japanese (hiragana)' if language=='japanese' else 'Korean')} words.\n"
"CRITICAL RULES:\n"
"- SINGLE SIMPLE WORDS ONLY - NO compound nouns, NO phrases, NO spaces\n"
"- Length: {('3-5 hiragana characters' if language=='japanese' else '2-4 hangul syllables')}\n"
"- Examples of GOOD words:\n"
f"  {('- みず (water), はな (flower), かぜ (wind)' if language=='japanese' else '- 학교 (school), 친구 (friend), 음식 (food)')}\n"
"- Examples of BAD words (too complex):\n"
f"  {('- ひらがなもじ (hiragana letters) - TOO LONG' if language=='japanese' else '- 도서관장 (library director), 영화관람 (movie viewing) - TOO COMPLEX')}\n"
"- First character must differ from last character\n"
"- Must be extremely common, everyday words a beginner would know\n"
'Return ONLY JSON: {{"words": ["word1","word2","word3"]}}'
                    )},
                ],
                temperature=0.7 + 0.05 * attempt,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content.strip())
            w = data.get("words", [])
            v = validate_words(w, language, min_len=4, max_len=6, enforce_diff_ends=True)
            if len(v) >= 3 and has_any_middle_overlap(v[:3]):
                starters = v[:3]
                print(f"Starter generation successful on attempt {attempt+1}")
                break
            else:
                print(f"Attempt {attempt+1}: Got {len(v)} valid words, need 3 (with middle overlaps)")
        except Exception as e:
            print(f"Starter attempt {attempt+1} failed: {e}")

    # 2) If that failed, ask for a bigger pool (8–12) and pick a trio locally
    if starters is None:
        pool_attempts = 4
        for ptry in range(pool_attempts):
            try:
                batch = 8 + 2 * ptry  # 8,10,12,14
                resp = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "Generate REAL, COMMON words. First and last MUST differ. No particles."},
                        {"role": "user", "content": (
                            f"Give {batch} {('Japanese (hiragana)' if language=='japanese' else 'Korean')} words for a crossword word bank.\n"
                            "Rules:\n"
                            "- length 4-6 (for Korean: 4-6 syllables)\n"
                            "- first and last must differ\n"
                            "- common words only\n"
                            "Return ONLY JSON: {\"words\": [ ... ]}"
                        )},
                    ],
                    temperature=0.9 + 0.05 * ptry,
                    response_format={"type": "json_object"},
                )
                data = json.loads(resp.choices[0].message.content.strip())
                pool = data.get("words", [])
                pool = validate_words(pool, language, min_len=4, max_len=6, enforce_diff_ends=True)
                trio = _pick_starter_trio(pool, language)
                if trio:
                    starters = trio
                    print(f"Starter trio selected from pool on attempt {ptry+1}: {trio}")
                    break
                else:
                    print(f"Pool attempt {ptry+1}: no connected trio found in {len(pool)} candidates")
            except Exception as e:
                print(f"Pool attempt {ptry+1} failed: {e}")

    # 3) Final fallback: pick a trio from local corpus (deterministic)
    if starters is None:
        corpus = FALLBACK_JA if language == "japanese" else FALLBACK_KO
        # Shuffle deterministically, then pick trio
        c = corpus[:]
        rng.shuffle(c)
        trio = _pick_starter_trio(c, language)
        if not trio:
            # try unshuffled as last resort
            trio = _pick_starter_trio(corpus, language)
        if not trio:
            # ultra-safe: just pick 3 valid words (even if not connected; grid placer will enforce)
            trio = [w for w in corpus if _valid_word(w, language, 4, 6)][:3]
        starters = trio
        print(f"Using local fallback starters: {starters}")

    # ---------- Step B: Build up to 10 words ----------
    all_words = starters[:]
    target_total = 10
    max_continuation_attempts = 10
    attempt_idx = 0

    while len(all_words) < target_total and attempt_idx < max_continuation_attempts:
        attempt_idx += 1
        remaining = target_total - len(all_words)
        batch_size = min(4, remaining)

        # Analyze existing words for prompt
        word_analysis = analyze_middle_characters(all_words)
        priority_chars = get_priority_characters(all_words, language)

        continuation_prompt = f"""
You have these existing crossword words: {", ".join(all_words)}
Generate {batch_size} MORE {('Japanese (hiragana)' if language=='japanese' else 'Korean')} words that will intersect well.

Rules:
1) Return EXACTLY {batch_size} words
2) SINGLE WORDS ONLY - NO SPACES - Must be single compound words
3) Length 3-8 characters/syllables
4) First and last must differ
5) REAL, common single words only (no particles, no multi-word phrases)
5) EACH new word MUST share at least ONE MIDDLE character/syllable with an existing word (not first/last)

Existing words middle characters/syllables:
{word_analysis}

Priority to use: {priority_chars}

Return ONLY JSON: {{"words": ["w1","w2","w3","w4"]}}
""".strip()

        try:
            resp = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Generate REAL words that intersect. No particles. First/last must differ."},
                    {"role": "user", "content": continuation_prompt},
                ],
                temperature=0.8 + attempt_idx * 0.02,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content.strip())
            new_words = data.get("words", [])
            valid_new = []
            for w in new_words:
                v = validate_words([w], language, min_len=3, max_len=8, enforce_diff_ends=True)
                if v and has_middle_intersection(v[0], all_words) and v[0] not in all_words:
                    valid_new.append(v[0])

            if valid_new:
                print(f"Added {len(valid_new)} intersecting words (attempt {attempt_idx})")
                all_words.extend(valid_new)
            else:
                print(f"Continuation attempt {attempt_idx}: No valid words generated")
        except Exception as e:
            print(f"Continuation attempt {attempt_idx} error: {e}")

    # ---------- Step C: If still short, fill from local corpus ----------
    if len(all_words) < target_total:
        corpus = FALLBACK_JA if language == "japanese" else FALLBACK_KO
        # Put most promising (by middle-char frequency) first, but shuffle deterministically within ties
        extra_pool = corpus[:]
        rng.shuffle(extra_pool)
        added = _fill_intersecting(all_words, language, target_total, extra_pool)
        if added:
            print(f"Filled {len(added)} words from local corpus")
            all_words.extend(added)

    # Last resort: if still short (should be rare), just append any valid uniques
    if len(all_words) < target_total:
        corpus = FALLBACK_JA if language == "japanese" else FALLBACK_KO
        for w in corpus:
            if len(all_words) >= target_total: break
            if w not in all_words and _valid_word(w, language, 3, 8):
                all_words.append(w)
        print(f"Final emergency fill; total now {len(all_words)}")

    final_words = all_words[:target_total]
    print(f"Final word list ({len(final_words)} words): {final_words}")
    return final_words

# ────────────────────── Validators/helpers ──────────────────────
def validate_words(words: List[str], language: str, min_len: int = 3, max_len: int = 8, enforce_diff_ends: bool = True) -> List[str]:
    """Validate words - check length, filter particles, ensure first≠last, basic cleanliness."""
    validated: List[str] = []

    if language == "japanese":
        invalid_patterns = ["の", "が", "を", "に", "へ", "と", "や", "から", "まで", "より", "で", "は"]
    else:
        invalid_patterns = ["의", "가", "을", "를", "에", "에서", "으로", "로", "와", "과", "이", "은", "는", "도"]

    for w in words:
        if not isinstance(w, str):
            continue
        w = w.strip()
        
        # ✅ REJECT WORDS WITH SPACES
        if ' ' in w:
            print(f"Rejected '{w}' - contains space")
            continue
            
        if not (min_len <= len(w) <= max_len):
            continue
        if enforce_diff_ends and len(w) >= 2 and w[0] == w[-1]:
            continue
        # end particle
        if any(w.endswith(p) for p in invalid_patterns):
            continue
        # overstuffed with particles
        if sum(1 for p in invalid_patterns if p in w) > 1:
            continue
        validated.append(w)
    return validated

def get_priority_characters(words: List[str], language: str) -> str:
    """Get most common middle characters for intersection."""
    counts: Dict[str, int] = {}
    for w in words:
        if len(w) > 2:
            for ch in w[1:-1]:
                counts[ch] = counts.get(ch, 0) + 1
    top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
    return ", ".join([c for c, _ in top]) if top else "any common characters"

def analyze_middle_characters(words: List[str]) -> str:
    lines = []
    for w in words:
        if len(w) > 2:
            lines.append(f"  - '{w}': {', '.join(list(w[1:-1]))}")
    return "\n".join(lines)

def has_middle_intersection(new_word: str, existing_words: List[str]) -> bool:
    if len(new_word) <= 2:
        return False
    S = set(new_word[1:-1])
    for ex in existing_words:
        if len(ex) > 2 and S & set(ex[1:-1]):
            return True
    return False

def has_any_middle_overlap(words: List[str]) -> bool:
    """At least one pair shares a middle character."""
    mids = [set(w[1:-1]) for w in words if len(w) > 2]
    for i in range(len(mids)):
        for j in range(i+1, len(mids)):
            if mids[i] & mids[j]:
                return True
    return False

# ───────────────────── Clue generation (unchanged) ─────────────────────
async def generate_crossword_clues(words: List[str], language: str) -> Dict[str, Dict]:
    prompt_template = CROSSWORD_PROMPTS.get(language, CROSSWORD_PROMPTS["japanese"])
    prompt = prompt_template.format(words=", ".join(words))
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a crossword puzzle creator. Return ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```json"):
            text = text[7:-3].strip()
        return json.loads(text)
    except Exception as e:
        print(f"Clue generation error: {e}")
        fallback: Dict[str, Dict] = {}
        for w in words:
            fallback[w] = {
                "clue_original": f"{w}の意味は?" if language == "japanese" else f"{w}의 의미는?",
                "clue_english": f"What does {w} mean?",
                "hints": ["Common word", "Basic vocabulary"],
                "difficulty": "N5" if language == "japanese" else "TOPIK1",
            }
        return fallback

# ───────────────────── Grid helpers ─────────────────────
def place_word_in_grid(grid: List[List[str]], word: str, r: int, c: int, direction: str) -> None:
    if direction == "across":
        for i, ch in enumerate(word):
            grid[r][c + i] = ch
    else:
        for i, ch in enumerate(word):
            grid[r + i][c] = ch

def _in_bounds(r: int, c: int, n: int) -> bool:
    return 0 <= r < n and 0 <= c < n

def _is_letter(ch: str) -> bool:
    return isinstance(ch, str) and ch not in (".", "", " ")
def can_place_word_strict(
    grid: List[List[str]],
    word: str,
    start_row: int,
    start_col: int,
    direction: str,
    grid_size: int,
    require_overlap: bool = True
) -> bool:
    """Classic crossword checks with proper overlap exemption:
       - Letters must match on overlaps
       - No side contact (perpendicular adjacency) **except at overlap cells**
       - Ends capped by '.' or bounds
       - Require ≥1 real overlap for non-first words (optional)
    """
    def _in_bounds(r: int, c: int) -> bool:
        return 0 <= r < grid_size and 0 <= c < grid_size

    def _is_letter(ch: str) -> bool:
        return isinstance(ch, str) and ch not in (".", "", " ")

    dr, dc = (0, 1) if direction == "across" else (1, 0)

    end_r = start_row + dr * (len(word) - 1)
    end_c = start_col + dc * (len(word) - 1)
    if not _in_bounds(start_row, start_col) or not _in_bounds(end_r, end_c):
        return False

    overlaps = 0
    for i, ch in enumerate(word):
        r = start_row + dr * i
        c = start_col + dc * i
        cell = grid[r][c]

        # must be empty or same letter
        if cell != "." and cell != ch:
            return False

        is_overlap = (cell == ch)
        if is_overlap:
            overlaps += 1

        # ── Perpendicular adjacency (skip on true overlap cells) ──
        if direction == "across":
            if not is_overlap:  # ← key fix
                for rr in (r - 1, r + 1):
                    if _in_bounds(rr, c) and _is_letter(grid[rr][c]):
                        return False
        else:  # down
            if not is_overlap:  # ← key fix
                for cc in (c - 1, c + 1):
                    if _in_bounds(r, cc) and _is_letter(grid[r][cc]):
                        return False

    # end caps
    before_r, before_c = start_row - dr, start_col - dc
    after_r, after_c   = end_r + dr,   end_c + dc
    if _in_bounds(before_r, before_c) and (grid[before_r][before_c] != "."):
        return False
    if _in_bounds(after_r, after_c) and (grid[after_r][after_c] != "."):
        return False

    if require_overlap and overlaps == 0:
        return False
    return True

def normalize_grid_chars(grid: List[List[str]]) -> List[List[str]]:
    """Ensure all non-letter cells are '.' so the UI renders black blocks correctly."""
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            ch = grid[r][c]
            if not _is_letter(ch):
                grid[r][c] = "."
    return grid

def trim_grid(grid: List[List[str]]) -> Tuple[List[List[str]], int, int]:
    """Trim grid to the minimal bounding box around letters, with 1-cell padding."""
    rows, cols = len(grid), len(grid[0])
    min_r, max_r = rows, -1
    min_c, max_c = cols, -1

    for r in range(rows):
        for c in range(cols):
            if _is_letter(grid[r][c]):
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)

    if max_r == -1:  # no letters
        return [["."]], 0, 0

    min_r = max(0, min_r - 1)
    max_r = min(rows - 1, max_r + 1)
    min_c = max(0, min_c - 1)
    max_c = min(cols - 1, max_c + 1)

    trimmed = [row[min_c : max_c + 1] for row in grid[min_r : max_r + 1]]
    return trimmed, min_r, min_c

# ───────────────────── Grid creation ─────────────────────
def _connectivity_score(w: str, placed: List[CrosswordWord]) -> int:
    """How many middle-char overlaps this word could make with placed words."""
    if len(w) <= 2 or not placed:
        return 0
    mids = set(w[1:-1])
    score = 0
    for ex in placed:
        exm = set(ex.word[1:-1]) if len(ex.word) > 2 else set()
        if mids & exm:
            score += 1
    return score

def _try_layout(words: List[str], word_data: Dict, date: str, language: str, grid_size: int, anchor_idx: int, rng: random.Random):
    """Build one layout using a specific anchor index; return (placed_count, grid, placed_words)."""
    # Copy so we can reorder per attempt
    wlist = words[:]
    # Put chosen anchor at index 0
    wlist[0], wlist[anchor_idx] = wlist[anchor_idx], wlist[0]

    # Fresh grid
    grid = [["." for _ in range(grid_size)] for _ in range(grid_size)]
    placed_words: List[CrosswordWord] = []

    # 1) Place first anchor ACROSS centered
    first = wlist[0]
    r0 = grid_size // 2
    c0 = (grid_size - len(first)) // 2
    place_word_in_grid(grid, first, r0, c0, "across")
    placed_words.append(
        CrosswordWord(
            word=first,
            clue_original=word_data[first]["clue_original"],
            clue_english=word_data[first]["clue_english"],
            hints=word_data[first]["hints"],
            difficulty=word_data[first]["difficulty"],
            start_row=r0,
            start_col=c0,
            direction="across",
            number=1,
        )
    )
    next_number = 2

    # 2) Choose BEST second word (from remaining) that intersects first going DOWN
    best = None
    for cand in wlist[1:]:
        for i, ch1 in enumerate(first):
            for j, ch2 in enumerate(cand):
                if ch1 != ch2:
                    continue
                new_r = r0 - j
                new_c = c0 + i
                if can_place_word_strict(grid, cand, new_r, new_c, "down", grid_size, require_overlap=True):
                    mid_score = -(abs(i - len(first)//2) + abs(j - len(cand)//2))
                    center_bias = -(abs(new_r - grid_size//2) + abs(new_c - grid_size//2))
                    score = mid_score + center_bias
                    # deterministic tie-break
                    tb = rng.random()
                    if best is None or (score, tb) > (best[0], best[1]):
                        best = (score, tb, cand, new_r, new_c)
    if best is None:
        return 1, grid, placed_words  # only the anchor placed

    _, _, second_word, r2, c2 = best
    place_word_in_grid(grid, second_word, r2, c2, "down")
    placed_words.append(
        CrosswordWord(
            word=second_word,
            clue_original=word_data[second_word]["clue_original"],
            clue_english=word_data[second_word]["clue_english"],
            hints=word_data[second_word]["hints"],
            difficulty=word_data[second_word]["difficulty"],
            start_row=r2,
            start_col=c2,
            direction="down",
            number=next_number,
        )
    )
    next_number += 1

    # 3) Deterministic shuffle of remaining words, but bias by connectivity potential
    remaining = [w for w in wlist[1:] if w != second_word]
    # stable shuffle
    rng.shuffle(remaining)

    # Placement loop: at each step, sort remaining by connectivity to currently placed words (desc)
    for _ in range(len(remaining)):
        # re-sort by connectivity each iteration
        remaining.sort(key=lambda w: (_connectivity_score(w, placed_words), rng.random()), reverse=True)
        w = remaining.pop(0)

        candidates = []
        for ex in placed_words:
            for i, wch in enumerate(w):
                for j, ech in enumerate(ex.word):
                    if wch != ech:
                        continue
                    if ex.direction == "across":
                        nr, nc, dirn = ex.start_row - i, ex.start_col + j, "down"
                    else:
                        nr, nc, dirn = ex.start_row + j, ex.start_col - i, "across"
                    if can_place_word_strict(grid, w, nr, nc, dirn, grid_size, require_overlap=True):
                        center_bias = -(abs(nr - grid_size//2) + abs(nc - grid_size//2))
                        mid_score = -(abs(i - len(w)//2) + abs(j - len(ex.word)//2))
                        score = mid_score + center_bias
                        candidates.append((score, rng.random(), nr, nc, dirn))

        candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
        placed = False
        for score, _tb, nr, nc, dirn in candidates:  # try ALL
            if can_place_word_strict(grid, w, nr, nc, dirn, grid_size, require_overlap=True):
                place_word_in_grid(grid, w, nr, nc, dirn)
                placed_words.append(
                    CrosswordWord(
                        word=w,
                        clue_original=word_data[w]["clue_original"],
                        clue_english=word_data[w]["clue_english"],
                        hints=word_data[w]["hints"],
                        difficulty=word_data[w]["difficulty"],
                        start_row=nr,
                        start_col=nc,
                        direction=dirn,
                        number=next_number,
                    )
                )
                next_number += 1
                placed = True
                break
        if not placed:
            # couldn't place this one now; push it back to the end and try others first
            remaining.append(w)

    return len(placed_words), grid, placed_words

def create_crossword_grid(word_data: Dict, date: str, language: str) -> Tuple[List[List[str]], List[CrosswordWord]]:
    """
    Multi-attempt deterministic placer:
      - Try several different anchor choices
      - Prefer layouts that place the most words
      - If best < 7 placed, retry with a different anchor order & slight score jitter
    """
    words: List[str] = list(word_data.keys())
    grid_size = 22
    rng_base = random.Random(f"{date}:{language}:place")

    best_result = (0, None, None)  # (placed_count, grid, placed_words)

    # Try up to K different anchors (first K words)
    K = min(5, len(words))  # try top 5 as starting anchors
    for k in range(K):
        # make a per-attempt RNG (deterministic)
        rng = random.Random(f"{date}:{language}:place:{k}")
        placed_count, grid, placed_words = _try_layout(words, word_data, date, language, grid_size, k, rng)
        if placed_count > best_result[0]:
            best_result = (placed_count, grid, placed_words)

    placed_count, grid, placed_words = best_result

    # Guard: if too few placed, retry with a different deterministic “jitter”
    if placed_count < 7:
        for k in range(K, K + 4):  # a few more tries with different seeds
            rng = random.Random(f"{date}:{language}:place:jitter:{k}")
            # also rotate input words so a different candidate becomes the early anchor
            rotated = words[k % len(words):] + words[:k % len(words)]
            pc, g2, pw2 = _try_layout(rotated, word_data, date, language, grid_size, 0, rng)
            if pc > placed_count:
                placed_count, grid, placed_words = pc, g2, pw2
            if placed_count >= 7:
                break

    print(f"📊 Placed {placed_count}/{len(words)} words")
    return grid, placed_words


# (Optional) Fallback kept for debugging, not used by the endpoint
def create_crossword_grid_fallback(word_data: Dict) -> Tuple[List[List[str]], List[CrosswordWord]]:
    words = list(word_data.keys())
    grid_size = 25
    grid = [["." for _ in range(grid_size)] for _ in range(grid_size)]
    placed_words: List[CrosswordWord] = []

    first = words[0]
    r0, c0 = grid_size // 2, (grid_size - len(first)) // 2
    place_word_in_grid(grid, first, r0, c0, "across")
    placed_words.append(
        CrosswordWord(
            word=first,
            clue_original=word_data[first]["clue_original"],
            clue_english=word_data[first]["clue_english"],
            hints=word_data[first]["hints"],
            difficulty=word_data[first]["difficulty"],
            start_row=r0,
            start_col=c0,
            direction="across",
            number=1,
        )
    )
    num = 2

    for w in words[1:]:
        done = False
        for ex in placed_words:
            if done:
                break
            for i, wch in enumerate(w):
                for j, ech in enumerate(ex.word):
                    if wch != ech:
                        continue
                    if ex.direction == "across":
                        nr, nc, dirn = ex.start_row - i, ex.start_col + j, "down"
                    else:
                        nr, nc, dirn = ex.start_row + j, ex.start_col - i, "across"
                    if can_place_word_strict(grid, w, nr, nc, dirn, grid_size, require_overlap=True):
                        place_word_in_grid(grid, w, nr, nc, dirn)
                        placed_words.append(
                            CrosswordWord(
                                word=w,
                                clue_original=word_data[w]["clue_original"],
                                clue_english=word_data[w]["clue_english"],
                                hints=word_data[w]["hints"],
                                difficulty=word_data[w]["difficulty"],
                                start_row=nr,
                                start_col=nc,
                                direction=dirn,
                                number=num,
                            )
                        )
                        num += 1
                        done = True
                        print(f"Fallback placed '{w}'")
                        break
    print(f"Fallback: {len(placed_words)}/{len(words)} words")
    return grid, placed_words

# ─── Daily puzzle cache ─────────────────────────────────────────
CACHE_DIR = Path("./daily_crossword_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _cache_key(date: str, language: str) -> str:
    return f"{date}:{language}"

def _cache_path(date: str, language: str) -> Path:
    # stable filename
    h = hashlib.sha256(_cache_key(date, language).encode()).hexdigest()[:16]
    return CACHE_DIR / f"{language}_{date}_{h}.json"

def load_cached_puzzle(date: str, language: str) -> dict | None:
    p = _cache_path(date, language)
    if p.exists():
        try:
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def save_cached_puzzle(date: str, language: str, puzzle_dict: dict) -> None:
    p = _cache_path(date, language)
    try:
        with p.open("w", encoding="utf-8") as f:
            json.dump(puzzle_dict, f, ensure_ascii=False)
    except Exception:
        pass

@app.post("/crossword/daily")
async def get_daily_crossword(body: CrosswordRequest):
    """Get daily crossword (cached per date+language)."""
    try:
        # 1) Serve from cache if available
        cached = load_cached_puzzle(body.date, body.language)
        if cached:
            print(f"Cache hit for {body.date} / {body.language}")
            return cached

        # 2) Generate fresh
        words = await generate_crossword_vocabulary(body.language, body.date)
        word_data = await generate_crossword_clues(words, body.language)

        # NOTE: use deterministic placer you already installed
        grid, placed_words = create_crossword_grid(word_data, body.date, body.language)
        print(f"Final: {len(placed_words)} words")

        # Normalize + trim for clean UI rendering
        grid = normalize_grid_chars(grid)
        trimmed_grid, off_r, off_c = trim_grid(grid)

        for w in placed_words:
            w.start_row -= off_r
            w.start_col -= off_c

        puzzle = CrosswordPuzzle(
            id=f"{body.date}_{body.language}",
            date=body.date,
            language=body.language,
            words=placed_words,
            grid_size=[len(trimmed_grid), len(trimmed_grid[0])],
            grid=trimmed_grid,
        )
        puzzle_dict = puzzle.dict()

        # 3) Save to cache
        save_cached_puzzle(body.date, body.language, puzzle_dict)

        # 4) Return
        return puzzle_dict

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

# Add these models and endpoint to your main.py

# ===============================================================
# 8. Word-a-thon (Wordle-like) endpoint
# ===============================================================

class WordathonRequest(BaseModel):
    uid: str
    date: str
    language: Literal["japanese", "korean"] = "japanese"

class WordathonResponse(BaseModel):
    id: str
    date: str
    language: str
    target_word: str
    word_length: int
    max_attempts: int

# Word-a-thon cache
WORDATHON_CACHE_DIR = Path("./daily_wordathon_cache")
WORDATHON_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---- Hangul → compatibility jamo helpers ----
S_BASE, L_BASE, V_BASE, T_BASE = 0xAC00, 0x1100, 0x1161, 0x11A7
L_COUNT, V_COUNT, T_COUNT = 19, 21, 28
N_COUNT = V_COUNT * T_COUNT

L_COMPAT = ["ㄱ","ㄲ","ㄴ","ㄷ","ㄸ","ㄹ","ㅁ","ㅂ","ㅃ","ㅅ","ㅆ","ㅇ","ㅈ","ㅉ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]
V_COMPAT = ["ㅏ","ㅐ","ㅑ","ㅒ","ㅓ","ㅔ","ㅕ","ㅖ","ㅗ","ㅘ","ㅙ","ㅚ","ㅛ","ㅜ","ㅝ","ㅞ","ㅟ","ㅠ","ㅡ","ㅢ","ㅣ"]
T_COMPAT = ["","ㄱ","ㄲ","ㄳ","ㄴ","ㄵ","ㄶ","ㄷ","ㄹ","ㄺ","ㄻ","ㄼ","ㄽ","ㄾ","ㄿ","ㅀ","ㅁ","ㅂ","ㅄ","ㅅ","ㅆ","ㅇ","ㅈ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]

def _decompose_hangul_syllable_to_compat(ch: str):
    code = ord(ch)
    if not (0xAC00 <= code <= 0xD7A3):
        return None
    s_index = code - S_BASE
    l_index = s_index // (N_COUNT)
    v_index = (s_index % (N_COUNT)) // T_COUNT
    t_index = s_index % T_COUNT
    parts = [L_COMPAT[l_index], V_COMPAT[v_index]]
    if t_index != 0:
        parts.append(T_COMPAT[t_index])
    return parts

def hangul_to_compat_jamo(text: str):
    out = []
    for ch in text:
        parts = _decompose_hangul_syllable_to_compat(ch)
        if parts is not None:
            out.extend(parts)
        elif 0x3130 <= ord(ch) <= 0x318F:  # direct compatibility jamo
            out.append(ch)
        else:
            # keep anything else as a single unit; we’ll reject it later
            out.append(ch)
    return out

def korean_jamo_len(text: str) -> int:
    return len(hangul_to_compat_jamo(text))

def is_all_hangul_syllables(text: str) -> bool:
    if not text or any(c.isspace() for c in text):
        return False
    for ch in text:
        if not (0xAC00 <= ord(ch) <= 0xD7A3):  # composed Hangul syllables only
            return False
    return True


def _wordathon_cache_path(date: str, language: str) -> Path:
    h = hashlib.sha256(f"{date}:{language}:wordathon".encode()).hexdigest()[:16]
    return WORDATHON_CACHE_DIR / f"wordathon_{language}_{date}_{h}.json"

def load_cached_wordathon(date: str, language: str) -> dict | None:
    p = _wordathon_cache_path(date, language)
    if p.exists():
        try:
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def save_cached_wordathon(date: str, language: str, word_dict: dict) -> None:
    p = _wordathon_cache_path(date, language)
    try:
        with p.open("w", encoding="utf-8") as f:
            json.dump(word_dict, f, ensure_ascii=False)
    except Exception:
        pass
async def generate_wordathon_word(language: str, date: str) -> dict:
    """
    Generate a word for Word-a-thon.
    - Japanese: EXACTLY 5 hiragana (unchanged)
    - Korean: a single Hangul word whose TOTAL JAMO count is 4–6 (inclusive)
    """
    rng = random.Random(f"{date}:{language}:wordathon")

    if language == "japanese":
        constraints = """
Return ONLY valid JSON.

Task:
- Generate EXACTLY ONE real Japanese word written ONLY in HIRAGANA.
- EXACTLY 5 characters (not 4, not 6).
- Reasonably common; intermediate learners should recognize it.
- No particles-only forms, no bare verb inflections (e.g., ～た／～る only).
- No proper nouns or place names.

Return JSON ONLY:
{"word":"ひらがな5文字","meaning":"English meaning"}
"""
    else:  # korean
        constraints = """
Return ONLY valid JSON.

Task:
- Generate EXACTLY ONE real Korean word written in composed Hangul syllables (no spaces).
- The word's TOTAL number of JAMO (consonant+vowel+final consonant counts) must be BETWEEN 4 AND 6 inclusive.
  * Example: "강" = ㄱ+ㅏ+ㅇ (3 jamo); "학교" = ㅎ+ㅏ+ㄱ + ㄱ+ㅛ (5 jamo).
- Reasonably common; suitable for intermediate learners.
- NO particles-only forms, NO pure verb endings, NO proper nouns/place names.

Return JSON ONLY:
{"word":"한글단어","meaning":"English meaning"}
"""

    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            resp = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role":"system","content":"You are a language game word generator. Return ONLY valid JSON with a word and its meaning."},
                    {"role":"user","content":constraints}
                ],
                temperature=0.7 + attempt * 0.02,
                max_tokens=100,
                response_format={"type":"json_object"}
            )

            result_text = resp.choices[0].message.content.strip()
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            data = json.loads(result_text)

            word = (data.get("word") or "").strip()
            meaning = (data.get("meaning") or "").strip()
            if not word or not meaning:
                continue

            if language == "japanese":
                # must be exactly 5 hiragana
                if len(word) == 5 and all(0x3040 <= ord(c) <= 0x309F for c in word):
                    return {"word": word, "meaning": meaning}
            else:
                # must be composed Hangul, jamo length 4–6
                if is_all_hangul_syllables(word):
                    jl = korean_jamo_len(word)
                    if 4 <= jl <= 6:
                        return {"word": word, "meaning": meaning}

        except Exception as e:
            print(f"⚠️ word gen attempt {attempt+1} failed: {e}")

    raise Exception(f"Failed to generate valid Word-a-thon word after {max_attempts} attempts")

@app.post("/wordathon/daily")
async def get_daily_wordathon(body: WordathonRequest):
    try:
        cached = load_cached_wordathon(body.date, body.language)
        if cached:
            return cached

        word_data = await generate_wordathon_word(body.language, body.date)

        if body.language == "korean":
            length_for_grid = korean_jamo_len(word_data["word"])   # 4–6
        else:
            length_for_grid = len(word_data["word"])               # 5 (JP)

        response = {
            "id": f"wordathon_{body.date}_{body.language}",
            "date": body.date,
            "language": body.language,
            "target_word": word_data["word"],
            "word_meaning": word_data["meaning"],
            "word_length": length_for_grid,
            "max_attempts": 6
        }

        save_cached_wordathon(body.date, body.language, response)
        return response

    except Exception as e:
        print(f"✖ Word-a-thon error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
