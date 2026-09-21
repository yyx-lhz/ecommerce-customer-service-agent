"""Ranks retrieved products against a structured buyer requirement."""

from pydantic import BaseModel

from app.inquiry.catalog import Product, ProductCandidate
from app.inquiry.schemas import StructuredInquiry


class ProductMatch(BaseModel):
    sku: str
    name: str
    score: float
    reasons: list[str]
    product: dict


class ProductMatcher:
    def select(
        self,
        inquiry: StructuredInquiry,
        candidates: list[ProductCandidate],
    ) -> ProductMatch | None:
        if not candidates:
            return None
        scored = [(self._score(inquiry, candidate), candidate) for candidate in candidates]
        score, candidate = max(scored, key=lambda item: item[0])
        product = candidate.product
        reasons = ["Best semantic match for the requested product description."]
        if inquiry.quantity and inquiry.quantity >= product.moq:
            reasons.append(f"Requested quantity meets the MOQ of {product.moq:,} pcs.")
        if inquiry.quantity and inquiry.quantity <= product.stock:
            reasons.append(f"Current stock can support the requested {inquiry.quantity:,} pcs.")
        if inquiry.target_price is not None and product.unit_price <= inquiry.target_price:
            reasons.append("Listed unit price is within the buyer's target price.")
        return ProductMatch(
            sku=product.sku,
            name=product.name,
            score=round(score, 4),
            reasons=reasons,
            product=_product_dict(product),
        )

    @staticmethod
    def _score(inquiry: StructuredInquiry, candidate: ProductCandidate) -> float:
        product = candidate.product
        score = candidate.score
        desired = {str(value).lower() for value in inquiry.specifications.attributes.values()}
        actual = {str(value).lower() for value in product.specifications.values()}
        score += 0.15 * len(desired & actual)
        if inquiry.quantity and product.moq <= inquiry.quantity <= product.stock:
            score += 0.1
        if inquiry.target_price is not None and product.unit_price <= inquiry.target_price:
            score += 0.05
        return score


def _product_dict(product: Product) -> dict:
    return {field: getattr(product, field) for field in Product.__dataclass_fields__}
