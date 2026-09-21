"""Repository boundary for inquiry persistence.

The in-memory implementation makes the first module runnable locally.  A
database-backed repository can replace it later without changing the API or
workflow contracts.
"""

from typing import Protocol

from app.inquiry.schemas import InquiryRecord


class InquiryRepository(Protocol):
    def save(self, record: InquiryRecord) -> InquiryRecord: ...

    def get(self, inquiry_id: str) -> InquiryRecord | None: ...


class InMemoryInquiryRepository:
    def __init__(self) -> None:
        self._records: dict[str, InquiryRecord] = {}

    def save(self, record: InquiryRecord) -> InquiryRecord:
        self._records[record.inquiry_id] = record
        return record

    def get(self, inquiry_id: str) -> InquiryRecord | None:
        return self._records.get(inquiry_id)
