from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    message: str
    profile: Optional[str] = "General"
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    profile: str

