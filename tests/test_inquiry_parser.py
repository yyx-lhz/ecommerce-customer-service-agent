from app.inquiry.parser import RuleBasedInquiryParser
from app.inquiry.repository import InMemoryInquiryRepository
from app.inquiry.schemas import InquiryInput
from app.inquiry.service import InquiryIntakeService


def test_rule_parser_extracts_core_rfq_fields():
    parser = RuleBasedInquiryParser()

    parsed = parser.parse(
        "Hello, please quote USD 4.20 for 500 pcs insulated water bottles "
        "shipping to Australia, FOB."
    )

    assert parsed.quantity == 500
    assert parsed.quantity_unit == "pcs"
    assert parsed.target_price == 4.2
    assert parsed.price_currency == "USD"
    assert parsed.country_or_region == "Australia"
    assert parsed.delivery.incoterm == "FOB"
    assert parsed.intent == "rfq"


def test_parse_attaches_schema_without_changing_raw_content():
    service = InquiryIntakeService(InMemoryInquiryRepository())
    original = "We need 1,000 pcs mugs. Please send a quotation."
    record = service.create(InquiryInput(content=original))

    parsed_record = service.parse(record.inquiry_id, RuleBasedInquiryParser())

    assert parsed_record is not None
    assert parsed_record.raw.content == original
    assert parsed_record.structured is not None
    assert parsed_record.structured.quantity == 1000


def test_rule_parser_removes_quantity_and_shipping_words_from_product_name():
    parsed = RuleBasedInquiryParser().parse(
        "Need 500 pcs stainless steel bottles shipping to Canada. Please send your quotation."
    )

    assert parsed.product_type == "stainless steel bottles"
    assert parsed.country_or_region == "Canada"
