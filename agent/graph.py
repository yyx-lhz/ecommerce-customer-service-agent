"""
LangGraph state graph definition — orchestrates the multi-agent pipeline.

Flow: Classify Intent → Retrieve Knowledge → Execute Tools → Generate Response → Reflect

Supports both single-turn (run_agent) and multi-turn (run_agent_with_memory) modes.
"""

import os
from collections import defaultdict

from langgraph.graph import StateGraph, END
from openai import OpenAI

from agent.nodes import (
    AgentState,
    classify_intent,
    retrieve_knowledge,
    execute_tools,
    generate_response,
    reflect_and_refine,
)
from rag.knowledge_base import ProductKnowledgeBase

# Lazy initialization
_client: OpenAI | None = None
_kb: ProductKnowledgeBase | None = None

# Session memory: session_id → list of {role, content}
_session_store: dict[str, list[dict]] = defaultdict(list)
MAX_HISTORY = 20  # keep last 20 messages per session


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


def get_kb() -> ProductKnowledgeBase:
    global _kb
    if _kb is None:
        _kb = ProductKnowledgeBase()
    return _kb


# ---------- Graph builder ----------

def build_graph() -> StateGraph:
    """Build and compile the LangGraph state graph."""

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify_intent", lambda s: classify_intent(s, get_client()))
    graph.add_node("retrieve_knowledge", lambda s: retrieve_knowledge(s, get_kb()))
    graph.add_node("execute_tools", lambda s: execute_tools(s, get_client()))
    graph.add_node("generate_response", lambda s: generate_response(s, get_client()))
    graph.add_node("reflect", lambda s: reflect_and_refine(s, get_client()))

    # Define edges
    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "retrieve_knowledge")
    graph.add_edge("retrieve_knowledge", "execute_tools")
    graph.add_edge("execute_tools", "generate_response")
    graph.add_edge("generate_response", "reflect")
    graph.add_edge("reflect", END)

    return graph.compile()


# Singleton compiled graph
_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_agent(user_message: str) -> AgentState:
    """
    Run the full agent pipeline on a single user message.
    Returns the final AgentState with the response and debug info.
    """
    state = AgentState(user_message=user_message)
    graph = get_graph()
    result = graph.invoke(state)
    return result


def run_agent_with_memory(user_message: str, session_id: str) -> AgentState:
    """
    Run the agent with multi-turn conversation memory.

    Args:
        user_message: The current user message
        session_id: Unique session identifier for tracking conversation state

    Returns:
        AgentState with response and debug info

    How memory works:
    1. Retrieves past messages for this session_id from _session_store
    2. Passes chat_history into AgentState so the response generator can reference it
    3. After generating a response, appends both the user msg and agent reply to the store
    4. Caps history at MAX_HISTORY entries to prevent unbounded growth
    """
    history = _session_store.get(session_id, [])

    state = AgentState(
        user_message=user_message,
        chat_history=list(history),  # pass a copy to avoid mutation
    )

    graph = get_graph()
    result = graph.invoke(state)

    # Update session memory — store the Q&A pair
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": result["final_response"]})

    # Trim old history if too long
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

    _session_store[session_id] = history

    return result


def clear_session(session_id: str):
    """Clear conversation history for a session."""
    _session_store.pop(session_id, None)
