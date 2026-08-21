from pathlib import Path

from app.agent.workflow import CustomerServiceAgent
from app.core.config import Settings
from app.core.schemas import ChatRequest


def test_return_request_completes_case_with_tool_result():
    agent = CustomerServiceAgent(Settings(), Path("data/knowledge"))

    response = agent.chat(
        ChatRequest(
            message="I want to return order OD1001 because I bought the wrong product.",
            session_id="case-return",
        )
    )

    assert response.case["case_status"] == "COMPLETED"
    assert response.case["order_id"] == "OD1001"
    assert "create_return_case" in response.case["executed_actions"]
    assert response.case["tool_results"][0]["ok"] is True


def test_return_request_without_order_waits_for_information():
    agent = CustomerServiceAgent(Settings(), Path("data/knowledge"))

    response = agent.chat(
        ChatRequest(message="I want to return a product.", session_id="case-waiting")
    )

    assert response.case["case_status"] == "WAITING_FOR_INFORMATION"
    assert response.case["current_step"] == "waiting_for_order_id"
    assert response.tool_calls == []
