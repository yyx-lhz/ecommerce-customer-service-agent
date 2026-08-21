from pathlib import Path

from app.agent.workflow import CustomerServiceAgent
from app.core.config import Settings
from app.core.schemas import ChatRequest


def test_agent_trace_contains_key_execution_stages():
    agent = CustomerServiceAgent(Settings(), Path("data/knowledge"))

    response = agent.chat(ChatRequest(message="Can you check my order OD1002?", session_id="trace"))
    stages = [event["stage"] for event in response.observability["events"]]

    assert response.trace_id.startswith("TR-")
    assert response.observability["trace_id"] == response.trace_id
    assert "user_input" in stages
    assert "intent_router" in stages
    assert "planner" in stages
    assert "retriever" in stages
    assert "tool_executor" in stages
    assert "final_response" in stages
