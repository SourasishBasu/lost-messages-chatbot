from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(default="user")
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gemini-1.5-flash")
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=0.9)
    max_tokens: Optional[int] = Field(default=150)

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"

class ChatCompletion(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Dict[str, int]