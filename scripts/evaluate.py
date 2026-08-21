import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.agent.workflow import CustomerServiceAgent
from app.core.config import get_settings
from app.core.schemas import ChatRequest
from app.evaluation.metrics import (
    EvalCase,
    faithfulness,
    intent_accuracy,
    recall_at_k,
    tool_accuracy,
)


def main() -> None:
    cases = [
        EvalCase(**item)
        for item in json.loads(Path("data/eval/customer_service_cases.json").read_text(encoding="utf-8"))
    ]
    agent = CustomerServiceAgent(get_settings(), Path("data/knowledge"))
    responses = [
        agent.chat(ChatRequest(message=case.question, session_id=f"eval-{case.id}")) for case in cases
    ]
    report = {
        "case_count": len(cases),
        "intent_accuracy": intent_accuracy(cases, responses),
        "tool_accuracy": tool_accuracy(cases, responses),
        "recall_at_5": recall_at_k(cases, responses, k=5),
        "faithfulness": faithfulness(responses),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
