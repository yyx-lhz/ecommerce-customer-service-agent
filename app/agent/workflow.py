from __future__ import annotations

import re
from dataclasses import asdict

from app.case.workflow import CaseWorkflow
from app.core.config import Settings
from app.core.memory import ConversationMemory
from app.core.schemas import ChatRequest, ChatResponse, Citation, Intent, ToolCall
from app.observability.trace import AgentTrace
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
        self.case_workflow = CaseWorkflow()

    def chat(self, request: ChatRequest) -> ChatResponse:
        trace: list[str] = []
        observability = AgentTrace()
        message = request.message.strip()
        history = self.memory.load(request.session_id)
        structured_memory = self.memory.load_structured(request.session_id)
        observability.record("user_input", message=message, session_id=request.session_id, user_id=request.user_id)

        intent = self._route_intent(message)
        trace.append(f"intent_router:{intent}")
        observability.record("intent_router", intent=intent)

        plan = self._plan(intent, message)
        trace.append(f"planner:{','.join(plan) or 'answer_only'}")
        observability.record("planner", decision=plan)
        order_id = self._extract_order_id(message)
        case = self.case_workflow.create_case(message, intent, order_id)
        self.case_workflow.analyze(case, plan)
        trace.append(f"case:{case.case_id}:{case.case_status.value}")
        observability.record("case_workflow", case=case.to_dict())

        retrievals = self.retriever.search(message, top_k=5)
        trace.append(f"retriever:{len(retrievals)}")
        observability.record(
            "retriever",
            documents=[
                {
                    "source": item.chunk.source,
                    "chunk_id": item.chunk.chunk_id,
                    "score": round(item.score, 4),
                    "vector_score": round(item.vector_score, 4),
                    "bm25_score": round(item.bm25_score, 4),
                }
                for item in retrievals
            ],
        )

        tool_calls = []
        if case.case_status.value != "WAITING_FOR_INFORMATION":
            self.case_workflow.before_tool_execution(case)
            for tool_name, args in self._tool_plan(intent, message):
                result = self.tools.execute(tool_name, args)
                tool_call = ToolCall(**asdict(result))
                tool_calls.append(tool_call)
                self.case_workflow.record_tool_call(case, tool_call)
                trace.append(f"tool_executor:{tool_name}:{result.ok}")
                observability.record(
                    "tool_executor",
                    selected_tool=tool_name,
                    tool_arguments=tool_call.arguments,
                    tool_results=tool_call.result,
                    ok=tool_call.ok,
                    attempts=tool_call.attempts,
                    idempotency_key=tool_call.idempotency_key,
                    cached=tool_call.cached,
                )
        self.case_workflow.complete(case)
        observability.record("case_completed", case=case.to_dict())

        answer = self._compose_answer(
            message,
            intent,
            retrievals,
            tool_calls,
            len(history),
            structured_memory.order_context,
        )
        reflection = self._reflect(
            answer,
            retrievals,
            tool_calls,
            intent,
            case.case_status.value,
            has_memory_evidence=bool(structured_memory.order_context),
        )
        trace.append(f"reflection:{reflection['status']}")
        observability.record("reflection", result=reflection)

        self.memory.append(request.session_id, "user", message)
        self.memory.append(request.session_id, "assistant", answer)
        structured_memory = self._update_structured_memory(
            request.session_id,
            request.user_id,
            case.to_dict(),
            tool_calls,
            retrievals,
        )
        observability.record("final_response", answer=answer)

        return ChatResponse(
            answer=answer,
            intent=intent,
            session_id=request.session_id,
            trace_id=observability.trace_id,
            case=case.to_dict(),
            memory=asdict(structured_memory),
            observability=observability.to_dict(),
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

    @staticmethod
    def _extract_order_id(message: str) -> str | None:
        match = ORDER_RE.search(message)
        return match.group(0).upper() if match else None

    def _compose_answer(
        self,
        message: str,
        intent: Intent,
        retrievals: list[RetrievalResult],
        tool_calls: list[ToolCall],
        history_count: int,
        order_context: dict,
    ) -> str:
        context = retrievals[0].chunk.text if retrievals else "No policy context found."
        prefix = "I checked the latest available order data and policy knowledge."
        if history_count:
            prefix += " I also considered the previous conversation context."
        if order_context and not any(call.name == "get_order" for call in tool_calls):
            prefix += f" Current order context: {order_context}."

        if intent == "small_talk":
            return "Hello! I can help with product questions, order status, logistics, inventory, returns, and refunds."

        if tool_calls:
            call = tool_calls[0]
            return f"{prefix} Tool result: {call.result}. Relevant policy: {context}"

        return f"{prefix} Relevant policy: {context} Customer question: {message}"

    def _update_structured_memory(
        self,
        session_id: str,
        user_id: str | None,
        case: dict,
        tool_calls: list[ToolCall],
        retrievals: list[RetrievalResult],
    ):
        order_context = {}
        previous_actions = []
        facts = []
        for call in tool_calls:
            previous_actions.append(
                {
                    "tool": call.name,
                    "arguments": call.arguments,
                    "ok": call.ok,
                    "idempotency_key": call.idempotency_key,
                }
            )
            if call.name == "get_order" and call.result.get("found"):
                order_context[call.result["order_id"]] = call.result
            if call.name in {"create_return_case", "create_return"} and call.result.get("created"):
                facts.append(f"Return case created for {call.arguments.get('order_id')}")
            if call.name == "track_logistics" and call.result.get("found"):
                facts.append(
                    f"Tracking {call.result['tracking_no']} is {call.result['stage']} with ETA {call.result['eta']}"
                )

        facts.extend([f"Retrieved {item.chunk.source}:{item.chunk.chunk_id}" for item in retrievals[:2]])
        profile = {"user_id": user_id} if user_id else {}
        return self.memory.update_structured(
            session_id,
            customer_profile=profile,
            order_context=order_context,
            current_case=case,
            previous_actions=previous_actions,
            important_business_facts=facts,
        )

    @staticmethod
    def _reflect(
        answer: str,
        retrievals: list[RetrievalResult],
        tool_calls: list[ToolCall],
        intent: Intent,
        case_status: str = "COMPLETED",
        has_memory_evidence: bool = False,
    ) -> dict:
        grounded = bool(retrievals) or intent == "small_talk"
        waiting_for_info = case_status == "WAITING_FOR_INFORMATION"
        tool_ok = all(call.ok for call in tool_calls) or waiting_for_info
        has_order_claim = "OD" in answer
        if has_order_claim and not tool_calls and not has_memory_evidence:
            return {"status": "needs_review", "faithful": False, "reason": "order claim without tool evidence"}
        return {
            "status": "pass" if grounded and tool_ok else "needs_review",
            "faithful": grounded and tool_ok,
            "retrieval_evidence": len(retrievals),
            "tool_ok": tool_ok,
        }
