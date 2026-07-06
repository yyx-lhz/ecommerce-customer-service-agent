"""
Mock APIs for ecommerce customer service — simulates order lookup, logistics tracking,
and product inventory queries.
"""

import json
import random
import time
from datetime import datetime, timedelta
from pathlib import Path


DATA_PATH = Path(__file__).parent.parent / "rag" / "data" / "products.json"

with open(DATA_PATH) as f:
    _data = json.load(f)

_products = {p["id"]: p for p in _data["products"]}

# Simulated order database
_ORDERS: dict[str, dict] = {}


def _seed_orders():
    """Generate some fake orders for demo purposes."""
    if _ORDERS:
        return
    statuses = ["pending", "processing", "shipped", "in_transit", "delivered", "cancelled"]
    for i in range(1, 8):
        pid = random.choice(list(_products.keys()))
        created = datetime.now() - timedelta(days=random.randint(1, 30))
        _ORDERS[f"ORD-{1000 + i}"] = {
            "order_id": f"ORD-{1000 + i}",
            "product_id": pid,
            "product_name": _products[pid]["name"],
            "quantity": random.randint(1, 3),
            "status": random.choice(statuses[:5]),
            "total": round(_products[pid]["price"] * random.randint(1, 3), 2),
            "currency": "USD",
            "created_at": created.strftime("%Y-%m-%d"),
            "estimated_delivery": (created + timedelta(days=random.randint(7, 20))).strftime("%Y-%m-%d"),
            "tracking_number": f"YT{random.randint(200000000, 299999999)}",
            "carrier": random.choice(["YunExpress", "4PX", "DHL", "FedEx"]),
        }


_seed_orders()


def lookup_order(order_id: str) -> dict | None:
    """
    Look up an order by ID.
    Returns order details dict or None if not found.
    """
    time.sleep(0.3)  # simulate API latency
    return _ORDERS.get(order_id.upper().strip())


def lookup_orders_by_email(email: str) -> list[dict]:
    """Find all orders associated with an email address (simulated)."""
    time.sleep(0.3)
    # In demo mode, return all orders for any email
    return list(_ORDERS.values())


def track_shipment(tracking_number: str) -> dict | None:
    """
    Track a shipment by tracking number.
    Returns tracking info or None.
    """
    time.sleep(0.5)
    for order in _ORDERS.values():
        if order.get("tracking_number") == tracking_number:
            tracking_events = [
                {"date": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d %H:%M"),
                 "location": "Shenzhen, CN", "status": "Package received by carrier"},
                {"date": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d %H:%M"),
                 "location": "Hong Kong, HK", "status": "Departed origin facility"},
                {"date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M"),
                 "location": "In Transit", "status": "Clearing customs"},
                {"date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                 "location": "Destination Country", "status": "Arrived at destination hub"},
            ]
            return {
                "tracking_number": tracking_number,
                "carrier": order["carrier"],
                "status": order["status"],
                "estimated_delivery": order["estimated_delivery"],
                "events": tracking_events,
            }
    return None


def check_stock(product_id: str) -> dict | None:
    """Check current stock level for a product."""
    product = _products.get(product_id.upper().strip())
    if not product:
        return None
    return {
        "product_id": product["id"],
        "product_name": product["name"],
        "stock": product["stock"],
        "in_stock": product["stock"] > 0,
        "restock_eta": "5-10 business days" if product["stock"] == 0 else None,
    }


def get_product_info(product_id: str) -> dict | None:
    """Get full product information."""
    return _products.get(product_id.upper().strip())


# Define the tool schemas for OpenAI Function Calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order by its order ID (e.g., ORD-1001). Returns order status, items, total, tracking number, and estimated delivery date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to look up, e.g., ORD-1001"
                    }
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "track_shipment",
            "description": "Track a shipment by its tracking number. Returns current status, location history, carrier info, and estimated delivery date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tracking_number": {
                        "type": "string",
                        "description": "The tracking number, e.g., YT200123456"
                    }
                },
                "required": ["tracking_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Check current stock availability for a product by product ID. Returns stock count and in-stock status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The product ID, e.g., P001"
                    }
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_info",
            "description": "Get detailed product information including price, shipping info, description, and FAQ by product ID or by searching product names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The product ID, e.g., P001"
                    }
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_return_policy",
            "description": "Get the store's return and refund policy information.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
]

# Map function names to actual callables
TOOL_MAP = {
    "lookup_order": lookup_order,
    "track_shipment": track_shipment,
    "check_stock": check_stock,
    "get_product_info": get_product_info,
    "get_return_policy": lambda: _data["return_policy"],
}
