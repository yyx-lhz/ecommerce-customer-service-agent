"""Read-only foreign-trade business tools used by the inquiry workflow."""

from app.inquiry.catalog import Product


class InquiryBusinessTools:
    def query_price(self, product: Product, quantity: int | None) -> dict:
        unit_price = product.unit_price
        if quantity and quantity >= 5000:
            unit_price = round(unit_price * 0.92, 2)
        elif quantity and quantity >= 1000:
            unit_price = round(unit_price * 0.96, 2)
        return {
            "sku": product.sku,
            "unit_price": unit_price,
            "currency": product.currency,
            "quote_basis": "EXW",
        }

    @staticmethod
    def query_inventory(product: Product) -> dict:
        return {"sku": product.sku, "available_quantity": product.stock, "moq": product.moq}

    @staticmethod
    def query_lead_time(product: Product) -> dict:
        return {"sku": product.sku, "production_lead_time_days": product.lead_time_days}

    @staticmethod
    def query_specifications(product: Product) -> dict:
        return {"sku": product.sku, "specifications": product.specifications}
