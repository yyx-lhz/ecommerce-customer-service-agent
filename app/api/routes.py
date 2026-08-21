from pathlib import Path

from fastapi import APIRouter, Depends

from app.agent.workflow import CustomerServiceAgent
from app.core.config import Settings, get_settings
from app.core.schemas import ChatRequest, ChatResponse

router = APIRouter()
SettingsDep = Depends(get_settings)


def get_agent(settings: Settings = SettingsDep) -> CustomerServiceAgent:
    return CustomerServiceAgent(settings=settings, knowledge_dir=Path("data/knowledge"))


AgentDep = Depends(get_agent)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, agent: CustomerServiceAgent = AgentDep) -> ChatResponse:
    return agent.chat(request)
