# app/schemas.py  (new file)
from typing import Literal
from pydantic import BaseModel

class ChatRequest(BaseModel):
    uid: str
    mode: Literal["convFormal", "convCasual"]
    userMessage: str
