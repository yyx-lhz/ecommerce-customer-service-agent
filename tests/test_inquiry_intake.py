from app.inquiry.repository import InMemoryInquiryRepository
from app.inquiry.schemas import InquiryInput, InquirySource
from app.inquiry.service import InquiryIntakeService


def test_intake_preserves_raw_overseas_inquiry():
    service = InquiryIntakeService(InMemoryInquiryRepository())
    raw = InquiryInput(
        content="Hello, please quote 500 stainless steel bottles for Sydney.",
        source=InquirySource.EMAIL,
        customer_name="Alex Smith",
        customer_email="alex@example.com",
        external_reference="RFQ-2026-08",
    )

    saved = service.create(raw)

    assert saved.inquiry_id.startswith("inq_")
    assert saved.raw == raw
    assert saved.structured is None
    assert service.get(saved.inquiry_id) == saved


def test_structured_schema_accepts_future_parser_output():
    from app.inquiry.schemas import StructuredInquiry

    inquiry = StructuredInquiry(
        product_type="stainless steel water bottle",
        quantity=500,
        quantity_unit="pieces",
        country_or_region="Australia",
        target_price=4.2,
        price_currency="USD",
        specifications={"attributes": {"capacity_ml": 750, "material": "304 stainless steel"}},
        delivery={"destination_port": "Sydney", "incoterm": "FOB"},
        intent="rfq",
    )

    assert inquiry.quantity == 500
    assert inquiry.delivery.incoterm == "FOB"
