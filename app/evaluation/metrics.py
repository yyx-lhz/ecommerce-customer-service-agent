from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.schemas import ChatResponse


@dataclass
class EvalCase:
    id: str
    question: str
    expected_intent: str
    expected_tool: str | None
    expected_source: str | None
    category: str = "general"
    session_id: str | None = None
    expected_tool_args: dict[str, Any] | None = None
    expected_case_status: str | None = None


def intent_accuracy(cases: list[EvalCase], responses: list[ChatResponse]) -> float:
    return _mean(case.expected_intent == response.intent for case, response in zip(cases, responses))


def tool_accuracy(cases: list[EvalCase], responses: list[ChatResponse]) -> float:
    checks = []
    for case, response in zip(cases, responses):
        actual = response.tool_calls[0].name if response.tool_calls else None
        checks.append(actual == case.expected_tool)
    return _mean(checks)


def tool_argument_accuracy(cases: list[EvalCase], responses: list[ChatResponse]) -> float:
    checks = []
    for case, response in zip(cases, responses):
        if not case.expected_tool_args:
            checks.append(True)
            continue
        actual = response.tool_calls[0].arguments if response.tool_calls else {}
        checks.append(all(actual.get(key) == value for key, value in case.expected_tool_args.items()))
    return _mean(checks)


def task_completion_rate(cases: list[EvalCase], responses: list[ChatResponse]) -> float:
    checks = []
    for case, response in zip(cases, responses):
        expected = case.expected_case_status or "COMPLETED"
        checks.append(response.case.get("case_status") == expected)
    return _mean(checks)


def recall_at_k(cases: list[EvalCase], responses: list[ChatResponse], k: int = 5) -> float:
    checks = []
    for case, response in zip(cases, responses):
        if not case.expected_source:
            checks.append(True)
            continue
        checks.append(any(citation.source == case.expected_source for citation in response.citations[:k]))
    return _mean(checks)


def faithfulness(responses: list[ChatResponse]) -> float:
    return _mean(response.reflection.get("faithful", False) for response in responses)


def category_breakdown(cases: list[EvalCase], responses: list[ChatResponse]) -> dict[str, dict[str, float]]:
    categories = sorted({case.category for case in cases})
    report = {}
    for category in categories:
        pairs = [(case, response) for case, response in zip(cases, responses) if case.category == category]
        category_cases = [case for case, _ in pairs]
        category_responses = [response for _, response in pairs]
        report[category] = {
            "case_count": len(category_cases),
            "intent_accuracy": intent_accuracy(category_cases, category_responses),
            "tool_accuracy": tool_accuracy(category_cases, category_responses),
            "task_completion_rate": task_completion_rate(category_cases, category_responses),
        }
    return report


def _mean(values) -> float:
    values = list(values)
    return round(sum(bool(value) for value in values) / max(len(values), 1), 4)
