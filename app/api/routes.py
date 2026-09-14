from fastapi import APIRouter, Request

from app.core.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.get("/health")
def health(request: Request):
    return {"status": "ok", "rag_backend": request.app.state.agent.settings.rag_backend}


@router.get("/ready")
def ready(request: Request):
    agent = request.app.state.agent
    if agent.settings.rag_backend == "production":
        agent.retriever.stores.check()
    return {"status": "ready", "rag_backend": agent.settings.rag_backend}


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, http_request: Request):
    return http_request.app.state.agent.chat(request)
