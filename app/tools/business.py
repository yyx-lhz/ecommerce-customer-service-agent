from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.tools.reliability import ToolReliabilityLayer

ORDERS = {
    "OD1001": {"status": "paid", "items": ["noise-canceling earbuds"], "tracking_no": "TRK9001"},
    "OD1002": {"status": "shipped", "items": ["travel adapter"], "tracking_no": "TRK9002"},
    "OD1003": {"status": "refund_pending", "items": ["smart watch strap"], "tracking_no": None},
}

LOGISTICS = {
    "TRK9001": {"carrier": "DHL", "stage": "customs_clearance", "eta": "2026-08-24"},
    "TRK9002": {"carrier": "FedEx", "stage": "in_transit", "eta": "2026-08-23"},
}

INVENTORY = {
    "earbuds": {"sku": "SKU-EAR-01", "available": 42, "warehouse": "CN-SH"},
    "adapter": {"sku": "SKU-ADT-02", "available": 8, "warehouse": "AU-SYD"},
    "watch strap": {"sku": "SKU-WST-03", "available": 0, "warehouse": "CN-SZ"},
}


@dataclass
class ToolResult:
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    ok: bool
    attempts: int = 1
    idempotency_key: str | None = None
    cached: bool = False
    error: str | None = None


class BusinessToolExecutor:
    def __init__(self):
        self.reliability = ToolReliabilityLayer()

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        handlers = {
            "get_order": self.get_order,
            "track_logistics": self.track_logistics,
            "check_inventory": self.check_inventory,
            "create_return_case": self.create_return_case,
            "create_return": self.create_return_case,
            "refund": self.refund,
            "cancel_order": self.cancel_order,
        }
        if name not in handlers:
            return ToolResult(name, arguments, {"error": "unknown tool"}, False)
        execution = self.reliability.execute(name, arguments, handlers[name])
        return ToolResult(
            name=name,
            arguments=execution.arguments,
            result=execution.result,
            ok=execution.ok,
            attempts=execution.attempts,
            idempotency_key=execution.idempotency_key,
            cached=execution.cached,
            error=execution.error,
        )

    def get_order(self, order_id: str) -> dict[str, Any]:
        order = ORDERS.get(order_id.upper())
        if not order:
            return {"found": False, "message": "Order not found"}
        return {"found": True, "order_id": order_id.upper(), **order}

    def track_logistics(self, tracking_no: str) -> dict[str, Any]:
        logistics = LOGISTICS.get(tracking_no.upper())
        if not logistics:
            return {"found": False, "message": "Tracking number not found"}
        return {"found": True, "tracking_no": tracking_no.upper(), **logistics}

    def check_inventory(self, keyword: str) -> dict[str, Any]:
        keyword = keyword.lower()
        for name, item in INVENTORY.items():
            if keyword in name or name in keyword:
                return {"found": True, "keyword": keyword, **item}
        return {"found": False, "keyword": keyword, "available": 0}

    def create_return_case(self, order_id: str, reason: str = "customer_request") -> dict[str, Any]:
        if order_id.upper() not in ORDERS:
            return {"created": False, "message": "Order not found"}
        return {"created": True, "case_id": f"RT-{order_id.upper()}", "reason": reason}

    def refund(
        self,
        order_id: str,
        amount: float | None = None,
        reason: str = "customer_request",
    ) -> dict[str, Any]:
        order = ORDERS.get(order_id.upper())
        if not order:
            return {"refunded": False, "message": "Order not found"}
        if order["status"] not in {"paid", "refund_pending"}:
            return {"refunded": False, "message": f"Order status {order['status']} is not refundable"}
        return {
            "refunded": True,
            "refund_id": f"RF-{order_id.upper()}",
            "amount": amount,
            "reason": reason,
        }

    def cancel_order(self, order_id: str, reason: str = "customer_request") -> dict[str, Any]:
        order = ORDERS.get(order_id.upper())
        if not order:
            return {"cancelled": False, "message": "Order not found"}
        if order["status"] != "paid":
            return {"cancelled": False, "message": f"Order status {order['status']} cannot be cancelled"}
        return {"cancelled": True, "cancel_id": f"CN-{order_id.upper()}", "reason": reason}
