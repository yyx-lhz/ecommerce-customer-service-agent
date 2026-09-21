"""End-to-end foreign-trade inquiry workflow."""

from pathlib import Path

from pydantic import BaseModel, Field

from app.inquiry.business_tools import InquiryBusinessTools
from app.inquiry.catalog import ProductKnowledgeBase
from app.inquiry.completeness import CompletenessResult, InquiryCompletenessChecker
from app.inquiry.matching import ProductMatch, ProductMatcher
from app.inquiry.parser import InquiryParser, RuleBasedInquiryParser
from app.inquiry.repository import InquiryRepository
from app.inquiry.response import InquiryResponseGenerator
from app.inquiry.schemas import InquiryRecord


class InquiryWorkflowResult(BaseModel):
    inquiry: InquiryRecord
    status: str
    completeness: CompletenessResult
    candidates: list[dict] = Field(default_factory=list)
    match: ProductMatch | None = None
    tool_results: dict[str, dict] = Field(default_factory=dict)
    reply_draft: str
    trace: list[str] = Field(default_factory=list)


class InquiryProcessingWorkflow:
    def __init__(
        self,
        repository: InquiryRepository,
        parser: InquiryParser | None = None,
        knowledge_base: ProductKnowledgeBase | None = None,
    ) -> None:
        self.repository = repository
        self.parser = parser or RuleBasedInquiryParser()
        data_path = Path("data/inquiry/products.json")
        self.knowledge_base = knowledge_base or ProductKnowledgeBase.from_json(data_path)
        self.checker = InquiryCompletenessChecker()
        self.matcher = ProductMatcher()
        self.tools = InquiryBusinessTools()
        self.responses = InquiryResponseGenerator()

    def run(self, inquiry_id: str) -> InquiryWorkflowResult | None:
        record = self.repository.get(inquiry_id)
        if record is None:
            return None
        trace = ["inquiry_parser"]
        record.structured = self.parser.parse(record.raw.content)
        self.repository.save(record)
        completeness = self.checker.check(record.structured)
        trace.append("completeness_checker")
        if not completeness.is_complete:
            return InquiryWorkflowResult(
                inquiry=record,
                status="waiting_for_information",
                completeness=completeness,
                reply_draft=self.responses.generate_clarification(completeness),
                trace=[*trace, "clarification_generator"],
            )

        query = " ".join(filter(None, [record.structured.product_type, record.raw.content]))
        candidates = self.knowledge_base.search(query)
        trace.append("product_retrieval")
        match = self.matcher.select(record.structured, candidates)
        if match is None:
            return InquiryWorkflowResult(
                inquiry=record,
                status="no_product_match",
                completeness=completeness,
                candidates=[],
                reply_draft=(
                    "Dear Customer,\n\nThank you for your inquiry. We are reviewing the requested "
                    "product and will revert shortly.\n\nBest regards,\nSales Team"
                ),
                trace=[*trace, "product_matching"],
            )
        product = candidates[[item.product.sku for item in candidates].index(match.sku)].product
        tool_results = {
            "price": self.tools.query_price(product, record.structured.quantity),
            "inventory": self.tools.query_inventory(product),
            "lead_time": self.tools.query_lead_time(product),
            "specifications": self.tools.query_specifications(product),
        }
        trace.extend(["product_matching", "business_tools", "response_generator"])
        return InquiryWorkflowResult(
            inquiry=record,
            status="ready_for_reply",
            completeness=completeness,
            candidates=[
                {
                    "sku": item.product.sku,
                    "name": item.product.name,
                    "score": round(item.score, 4),
                }
                for item in candidates
            ],
            match=match,
            tool_results=tool_results,
            reply_draft=self.responses.generate_quote(record, match, tool_results),
            trace=trace,
        )
