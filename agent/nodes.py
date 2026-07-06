"""
LangGraph agent nodes for the ecommerce customer service system.

Pipeline: Intent Classification → Tool Execution → Response Generation → Reflection
"""

import json
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from utils.mock_apis import TOOLS, TOOL_MAP
from utils.reflection import reflect, quick_reflect


# ---------- State definition ----------

@dataclass
class AgentState:
    """State that flows through the LangGraph pipeline."""
    messages: list[dict] = field(default_factory=list)
    user_message: str = ""
    chat_history: list[dict] = field(default_factory=list)  # multi-turn memory
    intent: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    rag_context: str = ""
    draft_response: str = ""
    final_response: str = ""
    reflection_passed: bool = True
    reflection_issues: list[str] = field(default_factory=list)


# ---------- Node implementations ----------

INTENT_CLASSIFIER_PROMPT = """You are an intent classifier for a cross-border ecommerce customer service system.

Classify the user's message into EXACTLY ONE of these intents:
- product_inquiry: Asking about product details, features, price, stock, or general product questions
- order_status: Asking about an order's current status, tracking, or delivery timeline
- shipping: Asking about shipping methods, costs, delivery time, or shipping policies
- return_refund: Asking about returns, refunds, warranty, or exchange policies
- complaint: Expressing dissatisfaction, reporting a problem with a product or service
- other: Anything that doesn't fit the above categories

User message: {user_message}

Return a JSON object with:
- "intent": one of the intent labels above
- "language": "zh" or "en" or "mixed"
- "entities": {{extracted entities like order_id, product_name, tracking_number}}
- "confidence": 0.0 to 1.0
"""


def classify_intent(state: AgentState, client: OpenAI) -> AgentState:
    """Node 1: Classify user intent."""
    prompt = INTENT_CLASSIFIER_PROMPT.format(user_message=state.user_message)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=300,
    )
    result = json.loads(response.choices[0].message.content)

    state.intent = result.get("intent", "other")
    state.messages.append({
        "role": "system",
        "content": f"Intent classified as: {state.intent} (confidence: {result.get('confidence', 0)})"
    })
    return state


def retrieve_knowledge(state: AgentState, kb) -> AgentState:
    """Retrieve relevant knowledge from RAG store."""
    results = kb.search(state.user_message, n_results=5)
    if results:
        state.rag_context = "\n\n".join(
            f"[{r['metadata'].get('product_id', r['id'])}] {r['content']}"
            for r in results
        )
    state.messages.append({
        "role": "system",
        "content": f"Retrieved {len(results)} knowledge chunks from RAG."
    })
    return state


def execute_tools(state: AgentState, client: OpenAI) -> AgentState:
    """Node 2: Decide and execute tool calls (Function Calling)."""
    # Build messages for the LLM to decide which tools to call
    system_msg = {
        "role": "system",
        "content": (
            "You are an ecommerce customer service assistant. Use the available functions "
            "to look up order information, track shipments, check stock, or get product details.\n\n"
            f"Relevant knowledge from our product database:\n{state.rag_context[:3000]}\n\n"
            "If the user provides an order ID, look it up. "
            "If they provide a tracking number, track it. "
            "If they ask about a product, get product info or check stock. "
            "If they ask about returns, get the return policy. "
            "Call the appropriate functions and present the results."
        )
    }

    conv_messages = [system_msg, {"role": "user", "content": state.user_message}]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conv_messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.2,
        max_tokens=800,
    )

    tool_results = []
    msg = response.choices[0].message

    if msg.tool_calls:
        for tc in msg.tool_calls:
            func_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if func_name in TOOL_MAP:
                try:
                    result = TOOL_MAP[func_name](**args)
                except Exception as e:
                    result = {"error": str(e)}
            else:
                result = {"error": f"Unknown function: {func_name}"}

            tool_results.append({
                "tool_call_id": tc.id,
                "function_name": func_name,
                "arguments": args,
                "result": result,
            })

    state.tool_results = tool_results
    state.messages.append({
        "role": "system",
        "content": f"Executed {len(tool_results)} tool call(s): "
                   f"{[t['function_name'] for t in tool_results]}"
    })
    return state


RESPONSE_GENERATOR_PROMPT = """You are a helpful and professional customer service agent for a cross-border ecommerce store.

Use the following context to answer the user's question accurately and helpfully.

## Conversation History:
{chat_history}

## Retrieved Knowledge Base:
{rag_context}

## Tool Results:
{tool_results}

## Guidelines:
- Be concise but complete — answer the question directly
- If order/tracking info was found, present it clearly
- If a product is out of stock, inform the user and suggest alternatives if possible
- Be empathetic — cross-border customers care about shipping time and costs
- Match the language of the user's question (Chinese → reply in Chinese, English → reply in English)
- If you don't have enough information, be honest about it
- Refer to previous conversation context when relevant to maintain coherence

User question: {user_message}

Please provide a helpful response:"""


def generate_response(state: AgentState, client: OpenAI) -> AgentState:
    """Node 3: Generate the final response using all gathered context."""
    tool_results_text = json.dumps(
        [{"fn": t["function_name"], "result": t["result"]} for t in state.tool_results],
        indent=2, ensure_ascii=False,
    )

    # Format chat history
    history_text = ""
    if state.chat_history:
        history_text = "\n".join(
            f"{h['role']}: {h['content']}" for h in state.chat_history[-6:]  # last 3 turns
        )

    prompt = RESPONSE_GENERATOR_PROMPT.format(
        chat_history=history_text or "(this is the first message in the conversation)",
        rag_context=state.rag_context[:3000],
        tool_results=tool_results_text,
        user_message=state.user_message,
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1000,
    )

    state.draft_response = response.choices[0].message.content
    return state


def reflect_and_refine(state: AgentState, client: OpenAI) -> AgentState:
    """Node 4: Self-check the response and refine if needed."""
    # Quick pre-check
    quick = quick_reflect(state.draft_response)
    if not quick.passed:
        state.reflection_issues = quick.issues

    # Full LLM-based reflection
    result = reflect(
        user_message=state.user_message,
        agent_response=state.draft_response,
        context=state.rag_context,
        client=client,
    )

    state.reflection_passed = result.passed
    state.reflection_issues.extend(result.issues)
    state.final_response = result.fixed_response

    return state
