# app/main.py
from typing import Literal
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from app.dummy_store import get_history, write_turns
from app.prompts import PROMPTS

# ---------- Pydantic schema ----------
from pydantic import BaseModel


class ChatRequest(BaseModel):
    uid: str
    mode: Literal["convFormal", "convCasual"]
    userMessage: str


# ---------- Init ----------
load_dotenv()  # loads OPENAI_API_KEY from .env during local runs
client = AsyncOpenAI()  # picks up OPENAI_API_KEY automatically
app = FastAPI()


# ---------- Route ----------
@app.post("/chat", response_class=StreamingResponse)
async def chat(body: ChatRequest):
    uid, mode, user_msg = body.uid, body.mode, body.userMessage

    # Build full message list: system prompt + history + new user turn
    messages = (
        [{"role": "system", "content": PROMPTS[mode]}]
        + get_history(uid)
        + [{"role": "user", "content": user_msg}]
    )

    # SSE generator ----------------------------------------------------------
    async def event_stream():
        assistant_text = ""
        try:
            response = await client.chat.completions.create(
                model="gpt-4o",
                stream=True,
                messages=messages,
            )
            async for chunk in response:  # async iterator yields token deltas
                token = chunk.choices[0].delta.content or ""
                assistant_text += token
                yield f"data: {token}\n\n"

            yield "event: done\ndata:[DONE]\n\n"

            # Persist history after the stream finishes
            write_turns(uid, user_msg, assistant_text)

        except Exception as e:
            # Surface errors to the client & log them
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
