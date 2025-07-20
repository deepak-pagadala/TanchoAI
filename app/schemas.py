# app/schemas.py

from typing import Literal, Optional
from pydantic import BaseModel

class ChatRequest(BaseModel):
    uid: str
    mode: Literal["convFormal", "convCasual"]
    userMessage: str
    name: Optional[str] = None

class MentorRequest(BaseModel):
    uid: str
    question: str
    freeSlot: Optional[str] = None   # ISO-8601 datetime string for client-computed slot
    name: Optional[str] = None  