from typing import Any, Literal

from pydantic import BaseModel, Field

Intent = Literal[
    "product_inquiry",
    "order_query",
    "logistics_tracking",
    "inventory_query",
    "return_refund",
    "policy_qa",
    "small_talk",
]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = Field(default="demo-session")
    user_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    source: str
    chunk_id: str
    score: float
    text: str


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    ok: bool = True
    attempts: int = 1
    idempotency_key: str | None = None
    cached: bool = False
    error: str | None = None


class ChatResponse(BaseModel):
    answer: str
    intent: Intent
    session_id: str
    trace_id: str
    case: dict[str, Any] = Field(default_factory=dict)
    memory: dict[str, Any] = Field(default_factory=dict)
    observability: dict[str, Any] = Field(default_factory=dict)
    citations: list[Citation] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    reflection: dict[str, Any] = Field(default_factory=dict)
    trace: list[str] = Field(default_factory=list)
