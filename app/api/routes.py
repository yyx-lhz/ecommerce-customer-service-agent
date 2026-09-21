from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from app.agent.workflow import CustomerServiceAgent
from app.core.config import Settings, get_settings
from app.core.schemas import ChatRequest, ChatResponse
from app.inquiry.repository import InMemoryInquiryRepository
from app.inquiry.schemas import InquiryInput, InquiryRecord
from app.inquiry.parser import OpenAIInquiryParser, RuleBasedInquiryParser
from app.inquiry.completeness import CompletenessResult, InquiryCompletenessChecker
from app.inquiry.service import InquiryIntakeService
from app.inquiry.workflow import InquiryProcessingWorkflow, InquiryWorkflowResult
from app.inquiry.langgraph_workflow import LangGraphInquiryWorkflow

router = APIRouter()
SettingsDep = Depends(get_settings)


def get_agent(settings: Settings = SettingsDep) -> CustomerServiceAgent:
    return CustomerServiceAgent(settings=settings, knowledge_dir=Path("data/knowledge"))


AgentDep = Depends(get_agent)

# The storage dependency is intentionally isolated here.  Replace this single
# instance with a database repository when durable persistence is introduced.
inquiry_intake = InquiryIntakeService(InMemoryInquiryRepository())
def build_inquiry_parser():
    settings = get_settings()
    if settings.inquiry_parser_provider.lower() == "openai" and settings.openai_api_key:
        return OpenAIInquiryParser(settings.openai_api_key, settings.inquiry_parser_model)
    return RuleBasedInquiryParser()


inquiry_workflow = InquiryProcessingWorkflow(
    inquiry_intake.repository,
    parser=build_inquiry_parser(),
)
inquiry_graph = LangGraphInquiryWorkflow(inquiry_workflow)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, agent: CustomerServiceAgent = AgentDep) -> ChatResponse:
    return agent.chat(request)


@router.post("/inquiries", response_model=InquiryRecord, status_code=status.HTTP_201_CREATED)
def create_inquiry(request: InquiryInput) -> InquiryRecord:
    """Store an original overseas-buyer inquiry before agent processing."""
    return inquiry_intake.create(request)


@router.get("/inquiries/{inquiry_id}", response_model=InquiryRecord)
def get_inquiry(inquiry_id: str) -> InquiryRecord:
    record = inquiry_intake.get(inquiry_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Inquiry not found")
    return record


@router.post("/inquiries/{inquiry_id}/parse", response_model=InquiryRecord)
def parse_inquiry(inquiry_id: str) -> InquiryRecord:
    """Parse an inquiry into the shared schema (offline fallback for now)."""
    record = inquiry_intake.parse(inquiry_id, build_inquiry_parser())
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Inquiry not found")
    return record


@router.get("/inquiries/{inquiry_id}/completeness", response_model=CompletenessResult)
def check_inquiry_completeness(inquiry_id: str) -> CompletenessResult:
    record = inquiry_intake.get(inquiry_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Inquiry not found")
    if record.structured is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Parse the inquiry before checking completeness",
        )
    return InquiryCompletenessChecker().check(record.structured)


@router.post("/inquiries/{inquiry_id}/process", response_model=InquiryWorkflowResult)
def process_inquiry(inquiry_id: str) -> InquiryWorkflowResult:
    """Run parsing, validation, retrieval, matching, tools, and reply drafting."""
    result = inquiry_graph.invoke(inquiry_id)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Inquiry not found")
    return result
