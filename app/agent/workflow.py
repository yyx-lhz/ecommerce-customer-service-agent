from __future__ import annotations

import re
from dataclasses import asdict

from app.core.config import Settings
from app.core.memory import ConversationMemory
from app.core.schemas import ChatRequest, ChatResponse, Citation, Intent, ToolCall
from app.rag.documents import load_markdown_chunks
from app.rag.hybrid import HybridRetriever, RetrievalResult
from app.tools.business import BusinessToolExecutor

ORDER_RE = re.compile(r"\bOD\d{4}\b", re.IGNORECASE)
TRACKING_RE = re.compile(r"\bTRK\d{4}\b", re.IGNORECASE)


class CustomerServiceAgent:
    def __init__(self, settings: Settings, knowledge_dir):
        self.settings = settings
        self.memory = ConversationMemory(settings)
        self.retriever = HybridRetriever(load_markdown_chunks(knowledge_dir))
        self.tools = BusinessToolExecutor()

    def chat(self, request: ChatRequest) -> ChatResponse:
        trace: list[str] = []
        message = request.message.strip()
        history = self.memory.load(request.session_id)

        intent = self._route_intent(message)
        trace.append(f"intent_router:{intent}")

        plan = self._plan(intent, message)
        trace.append(f"planner:{','.join(plan) or 'answer_only'}")

        retrievals = self.retriever.search(message, top_k=5)
        trace.append(f"retriever:{len(retrievals)}")

        tool_calls = []
        for tool_name, args in self._tool_plan(intent, message):
            result = self.tools.execute(tool_name, args)
            tool_calls.append(ToolCall(**asdict(result)))
            trace.append(f"tool_executor:{tool_name}:{result.ok}")

        answer = self._compose_answer(message, intent, retrievals, tool_calls, len(history))
        reflection = self._reflect(answer, retrievals, tool_calls, intent)
        trace.append(f"reflection:{reflection['status']}")

        self.memory.append(request.session_id, "user", message)
        self.memory.append(request.session_id, "assistant", answer)

        return ChatResponse(
            answer=answer,
            intent=intent,
            session_id=request.session_id,
            citations=[
                Citation(
                    source=item.chunk.source,
                    chunk_id=item.chunk.chunk_id,
                    score=round(item.score, 4),
                    text=item.chunk.text,
                )
                for item in retrievals
            ],
            tool_calls=tool_calls,
            reflection=reflection,
            trace=trace,
        )

    def _route_intent(self, message: str) -> Intent:
        text = message.lower()
        if "return" in text or "refund" in text or "退货" in text or "退款" in text:
            return "return_refund"
        if ORDER_RE.search(message) or "order" in text or "订单" in text:
            return "order_query"
        if (
            TRACKING_RE.search(message)
            or "tracking" in text
            or "logistics" in text
            or "物流" in text
            or "快递" in text
        ):
            return "logistics_tracking"
        if "stock" in text or "inventory" in text or "库存" in text:
            return "inventory_query"
        if "product" in text or "商品" in text or "adapter" in text or "earbuds" in text:
            return "product_inquiry"
        if "hi" in text or "hello" in text or "你好" in text:
            return "small_talk"
        return "policy_qa"

    @staticmethod
    def _plan(intent: Intent, message: str) -> list[str]:
        base = ["retrieve_policy"]
        if intent == "order_query":
            return ["extract_order_id", "get_order", *base]
        if intent == "logistics_tracking":
            return ["extract_tracking_no", "track_logistics", *base]
        if intent == "inventory_query":
            return ["extract_product_keyword", "check_inventory", *base]
        if intent == "return_refund":
            return ["check_return_policy", "maybe_create_return_case", *base]
        if intent == "small_talk":
            return []
        return base

    def _tool_plan(self, intent: Intent, message: str) -> list[tuple[str, dict]]:
        if intent == "order_query" and (match := ORDER_RE.search(message)):
            return [("get_order", {"order_id": match.group(0)})]
        if intent == "logistics_tracking" and (match := TRACKING_RE.search(message)):
            return [("track_logistics", {"tracking_no": match.group(0)})]
        if intent == "inventory_query":
            keyword = "earbuds" if "earbud" in message.lower() else "adapter"
            if "watch" in message.lower() or "strap" in message.lower():
                keyword = "watch strap"
            return [("check_inventory", {"keyword": keyword})]
        if intent == "return_refund" and (match := ORDER_RE.search(message)):
            return [("create_return_case", {"order_id": match.group(0), "reason": "customer_request"})]
        return []

    def _compose_answer(
        self,
        message: str,
        intent: Intent,
        retrievals: list[RetrievalResult],
        tool_calls: list[ToolCall],
        history_count: int,
    ) -> str:
        context = retrievals[0].chunk.text if retrievals else "No policy context found."
        prefix = "I checked the latest available order data and policy knowledge."
        if history_count:
            prefix += " I also considered the previous conversation context."

        if intent == "small_talk":
            return "Hello! I can help with product questions, order status, logistics, inventory, returns, and refunds."

        if tool_calls:
            call = tool_calls[0]
            return f"{prefix} Tool result: {call.result}. Relevant policy: {context}"

        return f"{prefix} Relevant policy: {context} Customer question: {message}"

    @staticmethod
    def _reflect(
        answer: str,
        retrievals: list[RetrievalResult],
        tool_calls: list[ToolCall],
        intent: Intent,
    ) -> dict:
        grounded = bool(retrievals) or intent == "small_talk"
        tool_ok = all(call.ok for call in tool_calls)
        has_order_claim = "OD" in answer
        if has_order_claim and not tool_calls:
            return {"status": "needs_review", "faithful": False, "reason": "order claim without tool evidence"}
        return {
            "status": "pass" if grounded and tool_ok else "needs_review",
            "faithful": grounded and tool_ok,
            "retrieval_evidence": len(retrievals),
            "tool_ok": tool_ok,
        }
