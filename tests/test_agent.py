from pathlib import Path

from app.agent.workflow import CustomerServiceAgent
from app.core.config import Settings
from app.core.schemas import ChatRequest


def build_agent() -> CustomerServiceAgent:
    return CustomerServiceAgent(Settings(), Path("data/knowledge"))


def test_order_query_calls_order_tool():
    agent = build_agent()
    response = agent.chat(ChatRequest(message="Please check order OD1002", session_id="test-order"))

    assert response.intent == "order_query"
    assert response.tool_calls[0].name == "get_order"
    assert response.tool_calls[0].result["found"] is True
    assert response.reflection["faithful"] is True


def test_hybrid_retrieval_returns_return_policy():
    agent = build_agent()
    response = agent.chat(
        ChatRequest(
            message="How many days does a refund take after warehouse inspection?",
            session_id="test-rag",
        )
    )

    assert any(citation.source == "return_policy.md" for citation in response.citations)


def test_inventory_tool_low_stock():
    agent = build_agent()
    response = agent.chat(
        ChatRequest(message="Do you have adapter inventory?", session_id="test-inventory")
    )

    assert response.intent == "inventory_query"
    assert response.tool_calls[0].name == "check_inventory"
    assert response.tool_calls[0].result["available"] == 8
