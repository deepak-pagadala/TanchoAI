# Complete updated main.py with working title generation

from __future__ import annotations
import pathlib
from typing import Literal, List, Dict, Optional, Tuple
import os, sys, json
import uuid
import random

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
from app.prompts   import PROMPTS, DICTIONARY_PROMPTS, SENTENCE_ANALYSIS_PROMPTS, CONJUGATION_PROMPTS
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
# 7. Crossword
# ===============================================================


# Add these models after your existing models
class CrosswordWord(BaseModel):
    word: str
    clue_original: str  # Japanese/Korean
    clue_english: str
    hints: List[str]    # 2 hints in English
    difficulty: str     # N5, N4, etc or TOPIK 1-6
    start_row: int
    start_col: int
    direction: str      # "across" or "down"
    number: int

class CrosswordPuzzle(BaseModel):
    id: str
    date: str
    language: str
    words: List[CrosswordWord]
    grid_size: List[int]  # [rows, cols]
    grid: List[List[str]]       # The actual letter grid

class CrosswordRequest(BaseModel):
    uid: str
    date: str  # YYYY-MM-DD format
    language: Literal["japanese", "korean"] = "japanese"

# Word pools by difficulty (expand these as needed)
WORD_POOLS = {
    "japanese": {
        "N5": ["犬", "猫", "本", "水", "食べる", "見る", "大きい", "小さい", "赤い", "青い", "新しい", "古い"],
        "N4": ["病院", "電話", "写真", "勉強", "買い物", "料理", "時間", "お金", "仕事", "学校"],
        "N3": ["会議", "経験", "関係", "説明", "連絡", "準備", "計画", "問題", "方法", "結果"],
        "N2": ["環境", "技術", "政治", "経済", "文化", "社会", "自然", "科学", "歴史", "未来"],
        "N1": ["哲学", "概念", "抽象", "具体", "理論", "実践", "構造", "機能", "本質", "現象"]
    },
    "korean": {
        "TOPIK1": ["개", "고양이", "책", "물", "먹다", "보다", "크다", "작다", "빨갛다", "파랗다", "새롭다", "오래되다"],
        "TOPIK2": ["병원", "전화", "사진", "공부", "쇼핑", "요리", "시간", "돈", "일", "학교"],
        "TOPIK3": ["회의", "경험", "관계", "설명", "연락", "준비", "계획", "문제", "방법", "결과"],
        "TOPIK4": ["환경", "기술", "정치", "경제", "문화", "사회", "자연", "과학", "역사", "미래"],
        "TOPIK5": ["철학", "개념", "추상", "구체", "이론", "실천", "구조", "기능", "본질", "현상"],
        "TOPIK6": ["인식론", "존재론", "현상학", "변증법", "체계", "형이상학", "윤리학", "미학", "논리학", "방법론"]
    }
}

def select_daily_words(language: str, target_date: str) -> List[str]:
    """Select words for the day based on difficulty distribution"""
    # Use date as seed for consistent daily puzzles
    random.seed(target_date + language)
    
    if language == "japanese":
        selected = []
        selected.extend(random.sample(WORD_POOLS["japanese"]["N5"], 3))
        selected.extend(random.sample(WORD_POOLS["japanese"]["N4"], 2))
        selected.extend([random.choice(WORD_POOLS["japanese"]["N3"])])
        selected.extend([random.choice(WORD_POOLS["japanese"]["N2"])])
        selected.extend([random.choice(WORD_POOLS["japanese"]["N1"])])
    else:
        selected = []
        selected.extend(random.sample(WORD_POOLS["korean"]["TOPIK1"], 3))
        selected.extend(random.sample(WORD_POOLS["korean"]["TOPIK2"], 2))
        selected.extend([random.choice(WORD_POOLS["korean"]["TOPIK3"])])
        selected.extend([random.choice(WORD_POOLS["korean"]["TOPIK4"])])
        selected.extend([random.choice(WORD_POOLS["korean"]["TOPIK5"])])
    
    return selected

async def generate_crossword_clues(words: List[str], language: str) -> Dict[str, Dict]:
    """Generate clues and hints for words using AI"""
    
    # Use your existing PROMPTS system
    prompt = f"""Generate crossword clues and hints for these {language} words: {', '.join(words)}

For each word, provide:
1. A crossword clue in {language} (short, cryptic style like newspaper crosswords)
2. A crossword clue in English (short, cryptic style)
3. Two helpful hints in English (not direct translations, but helpful context)
4. The difficulty level for each word

Format as JSON:
{{
  "word1": {{
    "clue_original": "clue in {language}",
    "clue_english": "clue in English", 
    "hints": ["hint 1", "hint 2"],
    "difficulty": "N5" or "TOPIK1" etc
  }},
  "word2": {{ ... }}
}}

Make clues challenging but fair. Use wordplay, synonyms, or context clues rather than direct definitions."""

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

def create_crossword_grid(word_data: Dict) -> Tuple[List[List[str]], List[CrosswordWord]]:
    """Create a proper rectangular crossword grid"""
    
    words = list(word_data.keys())
    grid_size = 15
    # Initialize with black squares (represented as '.')
    grid = [['.' for _ in range(grid_size)] for _ in range(grid_size)]
    
    placed_words = []
    
    # Place first word horizontally in center
    first_word = words[0]
    start_row = grid_size // 2
    start_col = (grid_size - len(first_word)) // 2
    
    # Mark this row/area as "word space" - we'll place letters here
    for i, letter in enumerate(first_word):
        grid[start_row][start_col + i] = letter
    
    placed_words.append(CrosswordWord(
        word=first_word,
        clue_original=word_data[first_word]["clue_original"],
        clue_english=word_data[first_word]["clue_english"],
        hints=word_data[first_word]["hints"],
        difficulty=word_data[first_word]["difficulty"],
        start_row=start_row,
        start_col=start_col,
        direction="across",
        number=1
    ))
    
    number = 2
    
    # Strategy: Place words in a more structured pattern
    # This creates a more traditional crossword appearance
    
    for word in words[1:]:
        placed = False
        
        # Try to intersect with existing words first
        for existing_word in placed_words:
            if placed:
                break
                
            for i, word_letter in enumerate(word):
                for j, existing_letter in enumerate(existing_word.word):
                    if word_letter == existing_letter:
                        # Calculate perpendicular placement
                        if existing_word.direction == "across":
                            new_start_row = existing_word.start_row - i
                            new_start_col = existing_word.start_col + j
                            direction = "down"
                        else:
                            new_start_row = existing_word.start_row + j
                            new_start_col = existing_word.start_col - i
                            direction = "across"
                        
                        if is_valid_crossword_placement(grid, word, new_start_row, new_start_col, direction, grid_size):
                            place_word_in_grid(grid, word, new_start_row, new_start_col, direction)
                            placed_words.append(CrosswordWord(
                                word=word,
                                clue_original=word_data[word]["clue_original"],
                                clue_english=word_data[word]["clue_english"],
                                hints=word_data[word]["hints"],
                                difficulty=word_data[word]["difficulty"],
                                start_row=new_start_row,
                                start_col=new_start_col,
                                direction=direction,
                                number=number
                            ))
                            number += 1
                            placed = True
                            break
                if placed:
                    break
        
        # If no intersection found, place strategically in grid pattern
        if not placed:
            placement = find_strategic_placement(grid, word, grid_size, placed_words)
            if placement:
                row, col, direction = placement
                place_word_in_grid(grid, word, row, col, direction)
                placed_words.append(CrosswordWord(
                    word=word,
                    clue_original=word_data[word]["clue_original"],
                    clue_english=word_data[word]["clue_english"],
                    hints=word_data[word]["hints"],
                    difficulty=word_data[word]["difficulty"],
                    start_row=row,
                    start_col=col,
                    direction=direction,
                    number=number
                ))
                number += 1
    
    # Keep the grid as-is - black squares ('.') will be handled by the frontend
    return grid, placed_words

def is_valid_crossword_placement(grid, word, start_row, start_col, direction, grid_size):
    """Check if word placement is valid for crossword rules"""
    
    if direction == "across":
        # Check bounds
        if (start_row < 0 or start_row >= grid_size or 
            start_col < 0 or start_col + len(word) > grid_size):
            return False
        
        # Check each letter position
        for i, char in enumerate(word):
            current_cell = grid[start_row][start_col + i]
            # Can place if empty or same letter (intersection)
            if current_cell != '.' and current_cell != char:
                return False
        
        # Crossword rule: can't have adjacent words
        # Check before and after
        if start_col > 0 and grid[start_row][start_col - 1] != '.':
            return False
        if start_col + len(word) < grid_size and grid[start_row][start_col + len(word)] != '.':
            return False
        
        return True
    
    else:  # down
        # Check bounds
        if (start_col < 0 or start_col >= grid_size or 
            start_row < 0 or start_row + len(word) > grid_size):
            return False
        
        # Check each letter position
        for i, char in enumerate(word):
            current_cell = grid[start_row + i][start_col]
            if current_cell != '.' and current_cell != char:
                return False
        
        # Check before and after
        if start_row > 0 and grid[start_row - 1][start_col] != '.':
            return False
        if start_row + len(word) < grid_size and grid[start_row + len(word)][start_col] != '.':
            return False
        
        return True

def place_word_in_grid(grid, word, start_row, start_col, direction):
    """Place word in the grid"""
    if direction == "across":
        for i, char in enumerate(word):
            grid[start_row][start_col + i] = char
    else:  # down
        for i, char in enumerate(word):
            grid[start_row + i][start_col] = char

def find_strategic_placement(grid, word, grid_size, existing_words):
    """Find good placement that maintains crossword structure"""
    
    # Define strategic positions that create a nice crossword pattern
    strategic_positions = [
        # Horizontal placements
        (3, 1, "across"), (3, 8, "across"),
        (5, 2, "across"), (5, 9, "across"), 
        (9, 1, "across"), (9, 8, "across"),
        (11, 3, "across"), (11, 7, "across"),
        
        # Vertical placements  
        (1, 3, "down"), (1, 7, "down"), (1, 11, "down"),
        (2, 5, "down"), (2, 9, "down"),
        (8, 2, "down"), (8, 6, "down"), (8, 12, "down"),
    ]
    
    # Try each strategic position
    for start_row, start_col, direction in strategic_positions:
        if is_valid_crossword_placement(grid, word, start_row, start_col, direction, grid_size):
            return (start_row, start_col, direction)
    
    # Fallback: try any valid position
    for row in range(1, grid_size - 1):
        for col in range(1, grid_size - 1):
            for direction in ["across", "down"]:
                if is_valid_crossword_placement(grid, word, row, col, direction, grid_size):
                    return (row, col, direction)
    
    return None

def check_adjacency_rules(grid, word, start_row, start_col, direction, grid_size):
    """Ensure words don't create invalid adjacencies"""
    if direction == "across":
        # Check cells before and after word
        if start_col > 0 and grid[start_row][start_col - 1] != '.':
            return False
        if start_col + len(word) < grid_size and grid[start_row][start_col + len(word)] != '.':
            return False
        
        # Check cells above and below (only where word doesn't intersect)
        for i, char in enumerate(word):
            col = start_col + i
            if grid[start_row][col] == '.':  # New placement, not intersection
                # Check above
                if start_row > 0 and grid[start_row - 1][col] != '.':
                    return False
                # Check below
                if start_row < grid_size - 1 and grid[start_row + 1][col] != '.':
                    return False
    
    else:  # down
        # Check cells before and after word
        if start_row > 0 and grid[start_row - 1][start_col] != '.':
            return False
        if start_row + len(word) < grid_size and grid[start_row + len(word)][start_col] != '.':
            return False
        
        # Check cells left and right (only where word doesn't intersect)
        for i, char in enumerate(word):
            row = start_row + i
            if grid[row][start_col] == '.':  # New placement, not intersection
                # Check left
                if start_col > 0 and grid[row][start_col - 1] != '.':
                    return False
                # Check right
                if start_col < grid_size - 1 and grid[row][start_col + 1] != '.':
                    return False
    
    return True



def create_crossword_word(word, word_info, start_row, start_col, direction, number):
    """Helper to create CrosswordWord object"""
    return CrosswordWord(
        word=word,
        clue_original=word_info["clue_original"],
        clue_english=word_info["clue_english"],
        hints=word_info["hints"],
        difficulty=word_info["difficulty"],
        start_row=start_row,
        start_col=start_col,
        direction=direction,
        number=number
    )

# Add this endpoint to your main.py
@app.post("/crossword/daily")
async def get_daily_crossword(body: CrosswordRequest):
    """Get or generate daily crossword puzzle"""
    
    try:
        print(f"🧩 Crossword request - UID: {body.uid}, Date: {body.date}, Language: {body.language}")
        
        # 1. Select words for the day
        words = select_daily_words(body.language, body.date)
        print(f"📝 Selected words: {words}")
        
        # 2. Generate clues and hints
        word_data = await generate_crossword_clues(words, body.language)
        print(f"🎯 Generated clues for {len(word_data)} words")
        
        # 3. Create crossword grid
        grid, placed_words = create_crossword_grid(word_data)
        print(f"🎲 Placed {len(placed_words)} words in grid")
        
        # 4. Return puzzle
        puzzle = CrosswordPuzzle(
            id=f"{body.date}_{body.language}",
            date=body.date,
            language=body.language,
            words=placed_words,
            grid_size=[len(grid), len(grid[0])],
            grid=grid
        )
        
        return puzzle.dict()
        
    except Exception as e:
        print(f"❌ Crossword generation error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to generate crossword: {str(e)}"}
        )