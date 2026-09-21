"""Professional English reply drafting grounded only in workflow evidence."""

from app.inquiry.completeness import CompletenessResult
from app.inquiry.matching import ProductMatch
from app.inquiry.schemas import InquiryRecord


class InquiryResponseGenerator:
    def generate_clarification(self, result: CompletenessResult) -> str:
        questions = "\n".join(f"- {question}" for question in result.clarification_questions)
        return (
            "Dear Customer,\n\nThank you for your inquiry. To prepare an accurate "
            f"quotation, could you please confirm:\n{questions}\n\nBest regards,\nSales Team"
        )

    def generate_quote(
        self,
        record: InquiryRecord,
        match: ProductMatch,
        tools: dict[str, dict],
    ) -> str:
        inquiry = record.structured
        assert inquiry is not None
        price = tools["price"]
        inventory = tools["inventory"]
        lead_time = tools["lead_time"]
        specs = tools["specifications"]["specifications"]
        quantity = f"{inquiry.quantity:,} {inquiry.quantity_unit or 'pcs'}"
        destination = inquiry.country_or_region
        spec_summary = ", ".join(
            f"{key.replace('_', ' ')}: {value}" for key, value in specs.items()
        )
        return (
            "Dear Customer,\n\n"
            f"Thank you for your inquiry for {quantity} of {match.name} for delivery "
            f"to {destination}. "
            f"We recommend SKU {match.sku}. The current indicative EXW price is "
            f"{price['currency']} {price['unit_price']:.2f}/pc, subject to final "
            "specification and packing confirmation.\n\n"
            f"Key specifications: {spec_summary}.\n"
            f"Availability: {inventory['available_quantity']:,} pcs in stock "
            f"(MOQ: {inventory['moq']:,} pcs).\n"
            "Production lead time: approximately "
            f"{lead_time['production_lead_time_days']} days after order confirmation.\n\n"
            "Please let us know your preferred packaging, destination port, and required "
            "delivery date, "
            "and we will issue a formal quotation with freight options.\n\n"
            "Best regards,\nSales Team"
        )
