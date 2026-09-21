"""Business completeness rules and buyer-facing clarification questions."""

from enum import StrEnum

from pydantic import BaseModel, Field

from app.inquiry.schemas import StructuredInquiry


class MissingField(StrEnum):
    PRODUCT_TYPE = "product_type"
    QUANTITY = "quantity"
    COUNTRY_OR_REGION = "country_or_region"


class CompletenessResult(BaseModel):
    is_complete: bool
    missing_fields: list[MissingField] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)


class InquiryCompletenessChecker:
    """Checks the minimum facts required before product matching and pricing."""

    _QUESTIONS = {
        MissingField.PRODUCT_TYPE: (
            "Could you please confirm the product name or share a product photo/specification?"
        ),
        MissingField.QUANTITY: (
            "What quantity do you require, including the unit (for example, pcs or sets)?"
        ),
        MissingField.COUNTRY_OR_REGION: "Which country or region should we quote delivery to?",
    }

    def check(self, inquiry: StructuredInquiry) -> CompletenessResult:
        missing: list[MissingField] = []
        if not inquiry.product_type:
            missing.append(MissingField.PRODUCT_TYPE)
        if inquiry.quantity is None:
            missing.append(MissingField.QUANTITY)
        if not inquiry.country_or_region:
            missing.append(MissingField.COUNTRY_OR_REGION)

        return CompletenessResult(
            is_complete=not missing,
            missing_fields=missing,
            clarification_questions=[self._QUESTIONS[field] for field in missing],
        )
