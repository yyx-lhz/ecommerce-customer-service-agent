from app.inquiry.completeness import InquiryCompletenessChecker, MissingField
from app.inquiry.schemas import StructuredInquiry


def test_checker_returns_targeted_clarification_questions_for_missing_facts():
    result = InquiryCompletenessChecker().check(StructuredInquiry(product_type="water bottle"))

    assert result.is_complete is False
    assert result.missing_fields == [MissingField.QUANTITY, MissingField.COUNTRY_OR_REGION]
    assert len(result.clarification_questions) == 2
    assert "quantity" in result.clarification_questions[0].lower()


def test_checker_accepts_minimum_quotable_inquiry():
    result = InquiryCompletenessChecker().check(
        StructuredInquiry(product_type="water bottle", quantity=500, country_or_region="Australia")
    )

    assert result.is_complete is True
    assert result.missing_fields == []
    assert result.clarification_questions == []
