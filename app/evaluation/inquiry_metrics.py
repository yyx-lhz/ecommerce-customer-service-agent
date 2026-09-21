"""Metrics required for the foreign-trade inquiry benchmark."""

from collections.abc import Iterable


def ratio(checks: Iterable[bool]) -> float:
    values = list(checks)
    return round(sum(values) / len(values), 4) if values else 0.0


def extraction_accuracy(cases: list[dict], results: list[dict]) -> float:
    checks: list[bool] = []
    fields = ("product_type", "quantity", "country_or_region", "target_price")
    for case, result in zip(cases, results):
        expected = case["expected"]
        actual = result["inquiry"]["structured"]
        for field in fields:
            if field in expected:
                checks.append(actual.get(field) == expected[field])
    return ratio(checks)


def intent_classification_accuracy(cases: list[dict], results: list[dict]) -> float:
    return ratio(
        result["inquiry"]["structured"].get("intent") == case["expected"].get("intent")
        for case, result in zip(cases, results)
    )


def retrieval_recall_at_k(cases: list[dict], results: list[dict], k: int = 3) -> float:
    checks = []
    for case, result in zip(cases, results):
        sku = case["expected"].get("sku")
        if sku is None:
            continue
        checks.append(sku in [candidate["sku"] for candidate in result["candidates"][:k]])
    return ratio(checks)


def tool_calling_accuracy(cases: list[dict], results: list[dict]) -> float:
    required = {"price", "inventory", "lead_time", "specifications"}
    return ratio(
        required <= set(result["tool_results"])
        if case["expected"].get("sku")
        else not result["tool_results"]
        for case, result in zip(cases, results)
    )


def end_to_end_task_completion_rate(cases: list[dict], results: list[dict]) -> float:
    return ratio(
        result["status"] == case["expected"]["status"]
        for case, result in zip(cases, results)
    )
