from app.tools.business import BusinessToolExecutor


def test_tool_schema_validation_rejects_bad_order_id():
    executor = BusinessToolExecutor()

    result = executor.execute("get_order", {"order_id": "bad-id"})

    assert result.ok is False
    assert result.result["error"] == "schema_validation_failed"
    assert result.attempts == 0


def test_side_effect_tool_is_idempotent():
    executor = BusinessToolExecutor()

    first = executor.execute("create_return_case", {"order_id": "OD1001", "reason": "wrong_item"})
    second = executor.execute("create_return_case", {"order_id": "OD1001", "reason": "wrong_item"})

    assert first.ok is True
    assert second.ok is True
    assert first.idempotency_key == second.idempotency_key
    assert second.cached is True
    assert second.result == first.result


def test_cancel_order_business_rule():
    executor = BusinessToolExecutor()

    result = executor.execute("cancel_order", {"order_id": "OD1002", "reason": "changed_mind"})

    assert result.ok is True
    assert result.result["cancelled"] is False
    assert "cannot be cancelled" in result.result["message"]
