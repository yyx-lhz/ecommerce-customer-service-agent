"""LangGraph adapter for the inquiry-processing workflow.

The graph exposes the business stages for observability and future insertion
of human approval nodes. The domain workflow remains the single owner of
business decisions and response payloads.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.inquiry.workflow import InquiryProcessingWorkflow, InquiryWorkflowResult


class InquiryGraphState(TypedDict, total=False):
    inquiry_id: str
    result: InquiryWorkflowResult
    status: str


class LangGraphInquiryWorkflow:
    def __init__(self, workflow: InquiryProcessingWorkflow) -> None:
        self.workflow = workflow
        graph = StateGraph(InquiryGraphState)
        graph.add_node("inquiry_parser", self._process)
        graph.add_node("completeness_checker", self._route)
        graph.add_node("clarification", self._finish)
        graph.add_node("retrieval_matching_tools_response", self._finish)
        graph.add_edge(START, "inquiry_parser")
        graph.add_edge("inquiry_parser", "completeness_checker")
        graph.add_conditional_edges(
            "completeness_checker",
            lambda state: (
                "clarification"
                if state["status"] == "waiting_for_information"
                else "retrieval_matching_tools_response"
            ),
        )
        graph.add_edge("clarification", END)
        graph.add_edge("retrieval_matching_tools_response", END)
        self.graph = graph.compile()

    def invoke(self, inquiry_id: str) -> InquiryWorkflowResult | None:
        state = self.graph.invoke({"inquiry_id": inquiry_id})
        return state.get("result")

    def _process(self, state: InquiryGraphState) -> InquiryGraphState:
        result = self.workflow.run(state["inquiry_id"])
        return {"result": result, "status": result.status if result else "not_found"}

    @staticmethod
    def _route(state: InquiryGraphState) -> InquiryGraphState:
        return state

    @staticmethod
    def _finish(state: InquiryGraphState) -> InquiryGraphState:
        return state
