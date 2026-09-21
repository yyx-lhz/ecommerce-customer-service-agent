from app.inquiry.repository import InMemoryInquiryRepository
from app.inquiry.schemas import InquiryInput
from app.inquiry.service import InquiryIntakeService
from app.inquiry.workflow import InquiryProcessingWorkflow


def test_complete_rfq_retrieves_product_calls_tools_and_drafts_quote():
    repository = InMemoryInquiryRepository()
    intake = InquiryIntakeService(repository)
    record = intake.create(
        InquiryInput(
            content=(
                "Please quote 1,000 pcs cotton canvas tote bags shipping to Australia, FOB."
            )
        )
    )

    result = InquiryProcessingWorkflow(repository).run(record.inquiry_id)

    assert result is not None
    assert result.status == "ready_for_reply"
    assert result.match is not None and result.match.sku == "BAG-TOTE-CAN"
    assert set(result.tool_results) == {"price", "inventory", "lead_time", "specifications"}
    assert "indicative EXW price" in result.reply_draft


def test_incomplete_rfq_returns_clarification_without_tool_calls():
    repository = InMemoryInquiryRepository()
    record = InquiryIntakeService(repository).create(
        InquiryInput(content="Could you send your mug catalog?")
    )

    result = InquiryProcessingWorkflow(repository).run(record.inquiry_id)

    assert result is not None
    assert result.status == "waiting_for_information"
    assert result.tool_results == {}
    assert "To prepare an accurate quotation" in result.reply_draft
