"""
LangGraph state graph definition — orchestrates the multi-agent pipeline.

Flow: Classify Intent → Retrieve Knowledge → Execute Tools → Generate Response → Reflect
"""

import os

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
    Run the full agent pipeline on a user message.
    Returns the final AgentState with the response and debug info.
    """
    state = AgentState(user_message=user_message)
    graph = get_graph()
    result = graph.invoke(state)
    return result
