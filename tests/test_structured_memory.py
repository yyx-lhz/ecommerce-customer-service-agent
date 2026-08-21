from pathlib import Path

from app.agent.workflow import CustomerServiceAgent
from app.core.config import Settings
from app.core.schemas import ChatRequest


def test_structured_memory_keeps_order_context_and_actions():
    agent = CustomerServiceAgent(Settings(), Path("data/knowledge"))

    first = agent.chat(
        ChatRequest(
            message="Can you check my order OD1002?",
            session_id="memory-session",
            user_id="customer-1",
        )
    )
    second = agent.chat(
        ChatRequest(
            message="What is the current status again?",
            session_id="memory-session",
            user_id="customer-1",
        )
    )

    assert first.memory["customer_profile"]["user_id"] == "customer-1"
    assert "OD1002" in first.memory["order_context"]
    assert "OD1002" in second.memory["order_context"]
    assert any(action["tool"] == "get_order" for action in second.memory["previous_actions"])
