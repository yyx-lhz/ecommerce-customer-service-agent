"""Inquiry-to-schema parsing boundary.

`InquiryParser` is intentionally provider-agnostic: production can plug in an
LLM with structured output, while the rule parser keeps local development and
tests deterministic.
"""

import re
from typing import Protocol

from app.inquiry.schemas import DeliveryRequirement, ProductSpecification, StructuredInquiry


class InquiryParser(Protocol):
    def parse(self, inquiry_text: str) -> StructuredInquiry: ...


class RuleBasedInquiryParser:
    """Small offline fallback, not a replacement for the production LLM parser."""

    _QUANTITY = re.compile(
        r"\b(\d{1,3}(?:,\d{3})*|\d{1,9})\s*(pcs?|pieces?|units?|sets?|cartons?)\b",
        re.IGNORECASE,
    )
    _PRICE = re.compile(
        r"(?:\b(?:usd|eur|gbp)\s*|[$€£])\s*(\d+(?:\.\d{1,2})?)", re.IGNORECASE
    )
    _COUNTRY = re.compile(
        r"\b(?:ship(?:ping)?\s+to|destination(?:\s+is)?|delivery\s+to)\s+([A-Z][A-Za-z ]{2,40})",
        re.IGNORECASE,
    )
    _INCOTERM = re.compile(r"\b(EXW|FCA|FOB|CFR|CIF|DAP|DDP)\b", re.IGNORECASE)
    _PRODUCT = re.compile(
        r"(?:quote(?:\s+us|\s+me)?\s+for|need|looking for|interested in)"
        r"\s+(?:about\s+)?(.+?)(?:\s+(?:for|to|at|with|under)\b|[,.?\n]|$)",
        re.IGNORECASE,
    )
    _PRODUCT_AFTER_QUANTITY = re.compile(
        r"\b\d{1,3}(?:,\d{3})*\s*(?:pcs?|pieces?|units?|sets?|cartons?)"
        r"\s+(.+?)(?:\s+shipping\b|[,.?\n]|$)",
        re.IGNORECASE,
    )
    _PRODUCT_AFTER_QUOTE = re.compile(r"\bquote\s+(?:a|an)\s+(.+?)(?:[,.?\n]|$)", re.IGNORECASE)

    def parse(self, inquiry_text: str) -> StructuredInquiry:
        text = " ".join(inquiry_text.split())
        quantity_match = self._QUANTITY.search(text)
        price_match = self._PRICE.search(text)
        country_match = self._COUNTRY.search(text)
        incoterm_match = self._INCOTERM.search(text)
        product_match = self._PRODUCT.search(text)

        currency = None
        if price_match:
            token = price_match.group(0).upper()
            if "EUR" in token or "€" in token:
                currency = "EUR"
            elif "GBP" in token or "£" in token:
                currency = "GBP"
            else:
                currency = "USD"

        lower = text.lower()
        if "sample" in lower:
            intent = "sample_request"
        elif "catalog" in lower:
            intent = "catalog_request"
        elif any(word in lower for word in ("quote", "price", "quotation")):
            intent = "rfq"
        else:
            intent = "general_inquiry"
        quantity_product_match = self._PRODUCT_AFTER_QUANTITY.search(text)
        product_contains_quantity = product_match and self._QUANTITY.search(product_match.group(1))
        if product_contains_quantity and quantity_product_match:
            product_match = quantity_product_match
        elif product_match is None and quantity_product_match:
            quantity_candidate = quantity_product_match.group(1).strip().lower()
            if not quantity_candidate.startswith(("shipping", "delivery", "to ", "for ")):
                product_match = quantity_product_match
        product_match = product_match or self._PRODUCT_AFTER_QUOTE.search(text)
        product_type = product_match.group(1).strip(" ,.") if product_match else None
        return StructuredInquiry(
            product_type=product_type,
            quantity=int(quantity_match.group(1).replace(",", "")) if quantity_match else None,
            quantity_unit=quantity_match.group(2).lower() if quantity_match else None,
            country_or_region=country_match.group(1).strip(" ,.") if country_match else None,
            specifications=ProductSpecification(free_text=text),
            target_price=float(price_match.group(1)) if price_match else None,
            price_currency=currency,
            delivery=DeliveryRequirement(
                incoterm=incoterm_match.group(1).upper() if incoterm_match else None
            ),
            intent=intent,
        )


class OpenAIInquiryParser:
    """Production parser using OpenAI structured output.

    The SDK is imported only when this parser is selected, keeping offline
    demos free from an API-key or optional dependency requirement.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self.api_key = api_key
        self.model = model

    def parse(self, inquiry_text: str) -> StructuredInquiry:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        completion = client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract a foreign-trade buyer inquiry. Do not invent values; "
                        "use null for unknowns. "
                        "Return the supplied structured schema."
                    ),
                },
                {"role": "user", "content": inquiry_text},
            ],
            response_format=StructuredInquiry,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("LLM returned no structured inquiry")
        return parsed
