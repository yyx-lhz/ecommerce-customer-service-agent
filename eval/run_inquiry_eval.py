"""Run the fixed benchmark for the foreign-trade inquiry workflow."""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.evaluation.inquiry_metrics import (
    end_to_end_task_completion_rate,
    extraction_accuracy,
    intent_classification_accuracy,
    retrieval_recall_at_k,
    tool_calling_accuracy,
)
from app.inquiry.repository import InMemoryInquiryRepository
from app.inquiry.schemas import InquiryInput
from app.inquiry.service import InquiryIntakeService
from app.inquiry.workflow import InquiryProcessingWorkflow


def main() -> None:
    cases = json.loads(Path("data/eval/inquiry_benchmark.json").read_text(encoding="utf-8"))
    repository = InMemoryInquiryRepository()
    intake = InquiryIntakeService(repository)
    workflow = InquiryProcessingWorkflow(repository)
    results = []
    for case in cases:
        record = intake.create(InquiryInput(content=case["text"]))
        result = workflow.run(record.inquiry_id)
        assert result is not None
        results.append(result.model_dump(mode="json"))
    report = {
        "case_count": len(cases),
        "information_extraction_accuracy": extraction_accuracy(cases, results),
        "intent_classification_accuracy": intent_classification_accuracy(cases, results),
        "product_retrieval_recall_at_3": retrieval_recall_at_k(cases, results),
        "tool_calling_accuracy": tool_calling_accuracy(cases, results),
        "end_to_end_task_completion_rate": end_to_end_task_completion_rate(cases, results),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
