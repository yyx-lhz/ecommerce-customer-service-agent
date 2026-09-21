"""Application service for the Inquiry Input module."""

from datetime import UTC, datetime
from uuid import uuid4

from app.inquiry.parser import InquiryParser
from app.inquiry.repository import InquiryRepository
from app.inquiry.schemas import InquiryInput, InquiryRecord


class InquiryIntakeService:
    def __init__(self, repository: InquiryRepository) -> None:
        self.repository = repository

    def create(self, raw_inquiry: InquiryInput) -> InquiryRecord:
        record = InquiryRecord(
            inquiry_id=f"inq_{uuid4().hex}",
            created_at=datetime.now(UTC),
            raw=raw_inquiry,
        )
        return self.repository.save(record)

    def get(self, inquiry_id: str) -> InquiryRecord | None:
        return self.repository.get(inquiry_id)

    def parse(self, inquiry_id: str, parser: InquiryParser) -> InquiryRecord | None:
        """Attach structured extraction without altering the original inquiry."""
        record = self.repository.get(inquiry_id)
        if record is None:
            return None
        record.structured = parser.parse(record.raw.content)
        return self.repository.save(record)
