"""
FastAPI backend for the cross-border ecommerce customer service agent.
Exposes the agent as REST API with proper error handling and logging.

Run with: uvicorn api:app --reload --port 8000
"""

import os
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.graph import run_agent, run_agent_with_memory

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent-api")


# ---------- Lifespan ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Agent API starting up...")
    yield
    logger.info("🛑 Agent API shutting down...")


app = FastAPI(
    title="跨境电商智能客服 Agent API",
    description="LangGraph-powered multi-agent customer service system with RAG + Reflection",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Schemas ----------
class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息", min_length=1, max_length=2000)
    session_id: str | None = Field(None, description="会话ID，用于多轮对话")
    language: str | None = Field(None, description="用户语言偏好 zh/en")


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Agent 回复")
    intent: str = Field(..., description="识别的意图")
    reflection_passed: bool = Field(..., description="自检是否通过")
    latency_ms: float = Field(..., description="响应延迟(毫秒)")
    session_id: str | None = Field(None, description="会话ID")


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float


# ---------- Health check ----------
_start_time = time.time()


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=round(time.time() - _start_time, 1),
    )


# ---------- Chat endpoint ----------
@app.post("/agent/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """多 Agent 智能客服对话接口"""
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")

    start = time.time()

    try:
        if req.session_id:
            result = run_agent_with_memory(req.message, req.session_id)
        else:
            result = run_agent(req.message)

        elapsed = (time.time() - start) * 1000

        return ChatResponse(
            reply=result.get("final_response", ""),
            intent=result.get("intent", "unknown"),
            reflection_passed=result.get("reflection_passed", True),
            latency_ms=round(elapsed, 1),
            session_id=req.session_id,
        )
    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Docs ----------
@app.get("/")
async def root():
    return {
        "service": "跨境电商智能客服 Agent API",
        "docs": "/docs",
        "endpoints": {
            "chat": "POST /agent/chat",
            "health": "GET /health",
        },
    }
