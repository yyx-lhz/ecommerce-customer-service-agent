"""Contracts for the inquiry-processing workflow.

The raw-input contracts are implemented in the first module.  The structured
schema is deliberately defined here too, so later parser, completeness, and
matching modules share one stable vocabulary.
"""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class InquirySource(StrEnum):
    EMAIL = "email"
    MARKETPLACE = "marketplace"
    WHATSAPP = "whatsapp"
    OTHER = "other"


class InquiryInput(BaseModel):
    """Raw message received from an overseas buyer."""

    content: str = Field(..., min_length=1, max_length=20_000)
    source: InquirySource = InquirySource.EMAIL
    customer_name: str | None = Field(default=None, max_length=200)
    customer_email: str | None = Field(default=None, max_length=320)
    external_reference: str | None = Field(default=None, max_length=200)


class ProductSpecification(BaseModel):
    """Open-ended product attributes, e.g. material, size, voltage, colour."""

    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict)
    free_text: str | None = None


class DeliveryRequirement(BaseModel):
    destination_port: str | None = None
    incoterm: str | None = None
    required_date: str | None = None
    shipping_method: str | None = None


class StructuredInquiry(BaseModel):
    """Target output of the future LLM-based inquiry parser."""

    product_type: str | None = None
    quantity: int | None = Field(default=None, ge=1)
    quantity_unit: str | None = None
    country_or_region: str | None = None
    specifications: ProductSpecification = Field(default_factory=ProductSpecification)
    target_price: float | None = Field(default=None, ge=0)
    price_currency: str | None = None
    delivery: DeliveryRequirement = Field(default_factory=DeliveryRequirement)
    intent: Literal["rfq", "sample_request", "catalog_request", "general_inquiry"] | None = None


class InquiryRecord(BaseModel):
    """Persisted inquiry and its untouched original content."""

    inquiry_id: str
    created_at: datetime
    raw: InquiryInput
    structured: StructuredInquiry | None = None
