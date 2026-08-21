from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4

from app.core.schemas import Intent, ToolCall


class CaseStatus(StrEnum):
    CREATED = "CREATED"
    ANALYZING = "ANALYZING"
    WAITING_FOR_INFORMATION = "WAITING_FOR_INFORMATION"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class CustomerServiceCase:
    case_id: str
    user_query: str
    intent: Intent
    order_id: str | None = None
    current_step: str = "created"
    executed_actions: list[str] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    case_status: CaseStatus = CaseStatus.CREATED

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "user_query": self.user_query,
            "intent": self.intent,
            "order_id": self.order_id,
            "current_step": self.current_step,
            "executed_actions": self.executed_actions,
            "tool_results": self.tool_results,
            "case_status": self.case_status.value,
        }


class CaseWorkflow:
    def create_case(self, user_query: str, intent: Intent, order_id: str | None) -> CustomerServiceCase:
        return CustomerServiceCase(
            case_id=f"CS-{uuid4().hex[:12].upper()}",
            user_query=user_query,
            intent=intent,
            order_id=order_id,
        )

    def analyze(self, case: CustomerServiceCase, planned_actions: list[str]) -> None:
        case.case_status = CaseStatus.ANALYZING
        case.current_step = "analyzing_request"
        case.executed_actions.append(f"plan:{','.join(planned_actions) or 'answer_only'}")

        if self._requires_order_id(case.intent) and not case.order_id:
            case.case_status = CaseStatus.WAITING_FOR_INFORMATION
            case.current_step = "waiting_for_order_id"

    def before_tool_execution(self, case: CustomerServiceCase) -> None:
        if case.case_status == CaseStatus.WAITING_FOR_INFORMATION:
            return
        case.case_status = CaseStatus.EXECUTING
        case.current_step = "executing_tools"

    def record_tool_call(self, case: CustomerServiceCase, tool_call: ToolCall) -> None:
        case.executed_actions.append(tool_call.name)
        case.tool_results.append(
            {
                "tool": tool_call.name,
                "arguments": tool_call.arguments,
                "ok": tool_call.ok,
                "result": tool_call.result,
            }
        )
        if not tool_call.ok:
            case.case_status = CaseStatus.FAILED
            case.current_step = f"failed:{tool_call.name}"

    def complete(self, case: CustomerServiceCase) -> None:
        if case.case_status in {CaseStatus.WAITING_FOR_INFORMATION, CaseStatus.FAILED}:
            return
        case.case_status = CaseStatus.COMPLETED
        case.current_step = "completed"

    @staticmethod
    def _requires_order_id(intent: Intent) -> bool:
        return intent in {"order_query", "return_refund"}
