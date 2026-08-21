from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class OrderArgs(BaseModel):
    order_id: str = Field(..., pattern=r"^OD\d{4}$")


class TrackingArgs(BaseModel):
    tracking_no: str = Field(..., pattern=r"^TRK\d{4}$")


class InventoryArgs(BaseModel):
    keyword: str = Field(..., min_length=1, max_length=80)


class ReturnArgs(BaseModel):
    order_id: str = Field(..., pattern=r"^OD\d{4}$")
    reason: str = Field(default="customer_request", min_length=1, max_length=120)


class RefundArgs(BaseModel):
    order_id: str = Field(..., pattern=r"^OD\d{4}$")
    amount: float | None = Field(default=None, ge=0)
    reason: str = Field(default="customer_request", min_length=1, max_length=120)


class CancelOrderArgs(BaseModel):
    order_id: str = Field(..., pattern=r"^OD\d{4}$")
    reason: str = Field(default="customer_request", min_length=1, max_length=120)


@dataclass(frozen=True)
class ToolSpec:
    schema: type[BaseModel]
    side_effect: bool = False
    timeout_seconds: float = 2.0
    retries: int = 1


@dataclass
class ReliableExecution:
    arguments: dict[str, Any]
    result: dict[str, Any]
    ok: bool
    attempts: int
    error: str | None = None
    idempotency_key: str | None = None
    cached: bool = False


class ToolReliabilityLayer:
    def __init__(self):
        self.specs = {
            "get_order": ToolSpec(OrderArgs, retries=1),
            "track_logistics": ToolSpec(TrackingArgs, retries=1),
            "check_inventory": ToolSpec(InventoryArgs, retries=1),
            "create_return_case": ToolSpec(ReturnArgs, side_effect=True, retries=0),
            "create_return": ToolSpec(ReturnArgs, side_effect=True, retries=0),
            "refund": ToolSpec(RefundArgs, side_effect=True, retries=0),
            "cancel_order": ToolSpec(CancelOrderArgs, side_effect=True, retries=0),
        }
        self._idempotency_cache: dict[str, dict[str, Any]] = {}

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        handler: Callable[..., dict[str, Any]],
    ) -> ReliableExecution:
        if name not in self.specs:
            return ReliableExecution(arguments, {"error": "unknown tool"}, False, attempts=0)

        spec = self.specs[name]
        try:
            validated = spec.schema(**self._normalize(name, arguments))
        except ValidationError as exc:
            return ReliableExecution(
                arguments,
                {"error": "schema_validation_failed", "details": exc.errors()},
                False,
                attempts=0,
                error="schema_validation_failed",
            )

        validated_args = validated.model_dump(exclude_none=True)
        idempotency_key = self._idempotency_key(name, validated_args) if spec.side_effect else None
        if idempotency_key and idempotency_key in self._idempotency_cache:
            return ReliableExecution(
                validated_args,
                self._idempotency_cache[idempotency_key],
                True,
                attempts=0,
                idempotency_key=idempotency_key,
                cached=True,
            )

        attempts = 0
        last_error: str | None = None
        for attempt in range(spec.retries + 1):
            attempts = attempt + 1
            started = time.monotonic()
            try:
                result = handler(**validated_args)
                elapsed = time.monotonic() - started
                if elapsed > spec.timeout_seconds:
                    raise TimeoutError(f"{name} exceeded {spec.timeout_seconds}s")
                if idempotency_key:
                    self._idempotency_cache[idempotency_key] = result
                return ReliableExecution(
                    validated_args,
                    result,
                    True,
                    attempts=attempts,
                    idempotency_key=idempotency_key,
                )
            except (TimeoutError, TypeError, ValueError, RuntimeError) as exc:
                last_error = str(exc)

        return ReliableExecution(
            validated_args,
            {"error": "tool_execution_failed", "details": last_error},
            False,
            attempts=attempts,
            error=last_error,
            idempotency_key=idempotency_key,
        )

    @staticmethod
    def _normalize(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(arguments)
        if "order_id" in normalized:
            normalized["order_id"] = str(normalized["order_id"]).upper()
        if "tracking_no" in normalized:
            normalized["tracking_no"] = str(normalized["tracking_no"]).upper()
        if name == "create_return":
            normalized.setdefault("reason", "customer_request")
        return normalized

    @staticmethod
    def _idempotency_key(name: str, arguments: dict[str, Any]) -> str:
        parts = [name]
        for key in sorted(arguments):
            parts.append(f"{key}={arguments[key]}")
        return "|".join(parts)
