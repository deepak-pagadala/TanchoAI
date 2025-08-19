# app/schemas.py

from typing import Literal, Optional
from pydantic import BaseModel

class ChatRequest(BaseModel):
    uid: str
    mode: Literal["convFormal", "convCasual"]
    userMessage: str
    name: Optional[str] = None
    language: Literal["japanese", "korean"] = "japanese"  # NEW
     
class MentorRequest(BaseModel):
    uid: str
    question: str
    freeSlot: Optional[str] = None
    name: Optional[str] = None
    language: Literal["japanese", "korean"] = "japanese"  # NEW