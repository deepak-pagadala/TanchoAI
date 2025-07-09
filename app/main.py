# app/main.py
from typing import Literal, List, Dict
import os, sys

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

from app.dummy_store import get_history, write_turns   # in-memory store
from app.prompts import PROMPTS
from app.resources import match_resources
from app.dummy_store import inc_topic, topic_hits
# ──────────────────────────── config ────────────────────────────
HISTORY_WINDOW = 6       # how many recent turns to send verbatim
SUMMARY_THRESHOLD = 40   # how many total turns before we start summarising
MODEL_NAME = "gpt-4o"    # or "gpt-4o-mini"

# ──────────────────────────── helpers ───────────────────────────
def format_last_turns(turns: List[Dict]) -> str:
    """
    Convert the last N turns into:
    USER: …
    ASSISTANT: …
    """
    lines = []
    for t in turns[-HISTORY_WINDOW:]:
        who = "USER" if t["role"] == "user" else "ASSISTANT"
        lines.append(f"{who}: {t['content']}")
    return "Last turns:\n" + "\n".join(lines) if lines else ""

# -------------- OPTIONAL long-memory summary stubs ---------------
def read_summary(uid: str) -> str | None:
    """TODO: load a saved summary for this chat (e.g. from Firestore)."""
    return None

def write_summary(uid: str, summary: str) -> None:
    """TODO: persist the summary somewhere durable."""
    pass

async def summarise_history(turns: List[Dict]) -> str:
    """
    OPTIONAL: use the model itself to compress >SUMMARY_THRESHOLD turns into
    2-3 polite sentences.  For now this is a stub returning an empty string.
    """
    return ""  # implement later if desired
# -----------------------------------------------------------------

# ──────────────────────── pydantic model ─────────────────────────
class ChatRequest(BaseModel):
    uid: str
    mode: Literal["convFormal", "convCasual"]
    userMessage: str

# ───────────────────────── FastAPI init ──────────────────────────
load_dotenv()
client = AsyncOpenAI()
app = FastAPI()

# ─────────────────────────── /chat route ─────────────────────────
@app.post("/chat", response_class=StreamingResponse)
async def chat(body: ChatRequest):
    uid, mode, user_msg = body.uid, body.mode, body.userMessage

    # 1️⃣  Load history and prepare system prompt + context
    history = get_history(uid)
    system_prompt = PROMPTS[mode]

    # add short context block
    context_block = format_last_turns(history)
    if context_block:
        system_prompt += "\n\n" + context_block

    # add long-memory summary if we have one
    if (summary := read_summary(uid)):
        system_prompt += "\n\nChat summary so far:\n" + summary

    messages = (
        [{"role": "system", "content": system_prompt}]
        + history[-HISTORY_WINDOW:]                # recent real turns
        + [{"role": "user", "content": user_msg}]
    )

    # 2️⃣  SSE generator — stream tokens back to client
    async def event_stream():
        assistant_text = ""
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                stream=True,
                messages=messages,
            )
            async for chunk in response:
                token = chunk.choices[0].delta.content or ""
                assistant_text += token
                yield f"data: {token}\n\n"

            yield "event: done\ndata:[DONE]\n\n"
            write_turns(uid, user_msg, assistant_text)

            # 3️⃣  OPTIONAL: create/update long-memory summary
            if len(history) + 1 > SUMMARY_THRESHOLD:
                # only summarise when assistant stops talking
                summary_text = await summarise_history(history + [{
                    "role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_text}
                ])
                if summary_text:
                    write_summary(uid, summary_text)

        except Exception as e:
            print("ERROR:", e, file=sys.stderr)
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


class MentorReq(BaseModel):
    uid: str
    question: str

@app.post("/mentor", response_class=StreamingResponse)
async def mentor(body: MentorReq):

    uid, q = body.uid, body.question
    topic = q.split()[0]          # naïve topic extractor
    inc_topic(uid, topic)

    resources = match_resources(topic)
    res_lines = "\n".join(
        f"- {r['title']} ({r['type']})" for r in resources
    ) or "NONE"

    system_prompt = PROMPTS["mentor"] + f"\n\nAVAILABLE_RESOURCES:\n{res_lines}"

    # include history to keep context
    messages = (
        [{"role": "system", "content": system_prompt}]
        + get_history(uid)
        + [{"role": "user", "content": q}]
    )

    # same streaming generator pattern as /chat
    async def event_stream():
        answer_text = ""
        try:
            stream = await client.chat.completions.create(
                model="gpt-4o",
                stream=True,
                messages=messages,
            )
            async for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                answer_text += token
                yield f"data: {token}\n\n"
            yield "event: done\ndata:[DONE]\n\n"
            write_turns(uid, q, answer_text)

        except Exception as e:
            yield f"event: error\ndata: {e}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")