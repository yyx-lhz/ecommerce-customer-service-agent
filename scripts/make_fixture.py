"""Generate the committed synthetic PDF; no external/business data."""

from pathlib import Path

import pymupdf

POLICIES = [
    ("Warranty", "AX900 adapters have a two-year warranty. Water damage is excluded."),
    (
        "Battery transport",
        "EB200 earbuds contain lithium batteries and cannot ship by untracked mail.",
    ),
    ("Gift wrapping", "Gift wrapping costs 4 dollars per item and is non-refundable."),
    ("Address changes", "Delivery addresses can be changed only before warehouse dispatch."),
    ("Loyalty points", "Loyalty points expire after twelve months of account inactivity."),
    (
        "Discount codes",
        "Only one discount code can be applied per order. Codes cannot be combined.",
    ),
    ("Invoice", "Tax invoices are downloadable from the order details page after payment."),
    ("Preorders", "Preorder items ship on their listed release date and may ship separately."),
]


def main():
    with pymupdf.open() as doc:
        for title, body in POLICIES:
            page = doc.new_page()
            page.insert_text((50, 60), title + "\n\n" + body, fontsize=11)
        doc.save(Path("data/knowledge/sample_policies.pdf"), deflate=True)


if __name__ == "__main__":
    main()
