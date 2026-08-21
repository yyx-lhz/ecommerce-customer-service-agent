from __future__ import annotations

from dataclasses import dataclass

from app.core.schemas import ChatResponse


@dataclass
class EvalCase:
    id: str
    question: str
    expected_intent: str
    expected_tool: str | None
    expected_source: str | None


def intent_accuracy(cases: list[EvalCase], responses: list[ChatResponse]) -> float:
    return _mean(case.expected_intent == response.intent for case, response in zip(cases, responses))


def tool_accuracy(cases: list[EvalCase], responses: list[ChatResponse]) -> float:
    checks = []
    for case, response in zip(cases, responses):
        actual = response.tool_calls[0].name if response.tool_calls else None
        checks.append(actual == case.expected_tool)
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


def _mean(values) -> float:
    values = list(values)
    return round(sum(bool(value) for value in values) / max(len(values), 1), 4)
