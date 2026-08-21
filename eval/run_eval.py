"""
Evaluation framework for the customer service agent.
Measures intent classification accuracy, response quality, and latency.

Usage: python eval/run_eval.py
"""

import json
import time
import sys
import os
from dataclasses import dataclass, field

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import run_agent


@dataclass
class TestCase:
    id: str
    question: str
    expected_intent: str
    expected_keywords: list[str]  # keywords that should appear in the answer
    forbidden_keywords: list[str] = field(default_factory=list)  # words that should NOT appear


@dataclass
class EvalResult:
    test_id: str
    question: str
    expected_intent: str
    actual_intent: str
    intent_match: bool
    reply: str
    keywords_found: list[str]
    keywords_missed: list[str]
    forbidden_found: list[str]
    latency_ms: float
    passed: bool


# ── 50 test cases covering all intents ──

TEST_CASES = [
    # === product_inquiry (10 cases) ===
    TestCase("P01", "蓝牙耳机多少钱？", "product_inquiry", ["29.99", "耳机", "蓝牙"], []),
    TestCase("P02", "What is the price of the fitness watch?", "product_inquiry", ["49.99", "watch", "fitness"], []),
    TestCase("P03", "充电宝还有货吗？", "product_inquiry", ["充电宝", "stock", "20000"], ["没有货", "缺货"]),
    TestCase("P04", "Do you have silk pillowcases in stock?", "product_inquiry", ["silk", "pillowcase", "24.99"], []),
    TestCase("P05", "这个蓝牙耳机防水吗？", "product_inquiry", ["IPX5", "waterproof", "防水"], []),
    TestCase("P06", "Is the smart watch compatible with iPhone?", "product_inquiry", ["iOS", "compatible", "watch"], []),
    TestCase("P07", "环形灯带三脚架吗？", "product_inquiry", ["ring light", "tripod", "三脚架"], []),
    TestCase("P08", "What products do you have for streaming?", "product_inquiry", ["ring light", "环形"], []),
    TestCase("P09", "有适合送女生的产品吗？", "product_inquiry", ["", "silk", "pillowcase"], []),
    TestCase("P10", "充电宝能带上飞机吗？", "product_inquiry", ["airplane", "airline", "飞机", "carry-on"], []),

    # === order_status (10 cases) ===
    TestCase("O01", "帮我查一下订单ORD-1001", "order_status", ["ORD-1001", ""], []),
    TestCase("O02", "Where is my order ORD-1002?", "order_status", ["ORD-1002", "status"], []),
    TestCase("O03", "我的订单什么时候到？订单号ORD-1003", "order_status", ["ORD-1003", "estimated_delivery", "delivery"], []),
    TestCase("O04", "ORD-1005状态", "order_status", ["ORD-1005", ""], []),
    TestCase("O05", "Can you check ORD-1007 for me?", "order_status", ["ORD-1007", ""], []),
    TestCase("O06", "订单ORD-1004退款了吗？", "order_status", ["ORD-1004", ""], []),
    TestCase("O07", "我的包裹是不是丢了？ORD-1006", "order_status", ["ORD-1006", ""], []),
    TestCase("O08", "I haven't received ORD-1001 yet, where is it?", "order_status", ["ORD-1001", ""], []),
    TestCase("O09", "ORD-1008状态", "order_status", ["", ""], []),  # non-existent
    TestCase("O10", "能帮我确认一下订单吗？下单后没收到确认邮件", "order_status", ["confirm", "order"], ["I don't"]),

    # === shipping (10 cases) ===
    TestCase("S01", "包邮吗？", "shipping", ["free shipping", "包邮", "39.99"], []),
    TestCase("S02", "How long does shipping take?", "shipping", ["days", "shipping", "delivery"], []),
    TestCase("S03", "你们用什么快递？", "shipping", ["carrier", "YunExpress", "4PX", "DHL", "FedEx"], []),
    TestCase("S04", "发货到美国要多久？", "shipping", ["days", "US", "shipping"], []),
    TestCase("S05", "Do you ship to Europe?", "shipping", ["EU", "Europe", "warehouse"], []),
    TestCase("S06", "运费多少钱？", "shipping", ["shipping", "free", "4.99", "5.99"], []),
    TestCase("S07", "下单后多久发货？", "shipping", ["processing", "1-3", "business days"], []),
    TestCase("S08", "What is the free shipping threshold?", "shipping", ["39.99", "free shipping"], []),
    TestCase("S09", "可以加急吗？", "shipping", ["shipping", "delivery", "express"], []),
    TestCase("S10", "从哪个国家发货？", "shipping", ["warehouse", "CN", "US", "warehouses"], []),

    # === return_refund (10 cases) ===
    TestCase("R01", "可以退货吗？", "return_refund", ["return", "30", "day"], []),
    TestCase("R02", "What is your refund policy?", "return_refund", ["refund", "return", "30"], []),
    TestCase("R03", "收到货坏了怎么退？", "return_refund", ["return", "defective", "contact"], []),
    TestCase("R04", "退货邮费谁出？", "return_refund", ["return", "buyer pays", "defective", "shipping"], []),
    TestCase("R05", "退款多久到账？", "return_refund", ["refund", "5-7", "business days", "timeline"], []),
    TestCase("R06", "How do I return an item?", "return_refund", ["return", "original packaging", "unused"], []),
    TestCase("R07", "过了30天还能退吗？", "return_refund", ["30", "day", "period"], []),
    TestCase("R08", "可以换货吗？", "return_refund", ["exchange", "return"], []),
    TestCase("R09", "What condition must items be in for return?", "return_refund", ["unused", "original packaging"], []),
    TestCase("R10", "退货地址是什么？", "return_refund", ["return", ""], []),

    # === complaint (5 cases) ===
    TestCase("C01", "收到的产品坏了！", "complaint", ["sorry", "defective", "replace", "return"], []),
    TestCase("C02", "The earbuds stopped working after 2 days!", "complaint", ["sorry", "earbuds", "problem"], []),
    TestCase("C03", "物流太慢了，等的受不了了", "complaint", ["sorry", "delay", "shipping"], []),
    TestCase("C04", "I want to speak to a manager!", "complaint", ["", "help"], []),
    TestCase("C05", "收到的颜色不对，发错了", "complaint", ["wrong", "return", "exchange", "help"], []),

    # === other (5 cases) ===
    TestCase("X01", "你好", "other", ["hello", "help", "你好", "assist"], []),
    TestCase("X02", "你们公司在哪里？", "other", ["", ""], []),
    TestCase("X03", "现在有什么优惠吗？", "other", ["", ""], []),
    TestCase("X04", "How do I contact customer support?", "other", ["contact", "email", "support"], []),
    TestCase("X05", "谢谢！", "other", ["welcome", "help", "assist"], []),
]


def run_evaluation(test_cases: list[TestCase], verbose: bool = True) -> tuple[list[EvalResult], dict]:
    """Run all test cases and return results + summary stats."""
    results: list[EvalResult] = []

    for tc in test_cases:
        start = time.time()
        try:
            state = run_agent(tc.question)
            reply = state.get("final_response", "")
            latency = (time.time() - start) * 1000
        except Exception as e:
            reply = f"[ERROR: {e}]"
            latency = (time.time() - start) * 1000

        # Check intent match
        intent_match = state.get("intent", "") == tc.expected_intent

        # Check keywords
        keywords_found = []
        keywords_missed = []
        reply_lower = reply.lower()

        for kw in tc.expected_keywords:
            if kw and kw.lower() in reply_lower:
                keywords_found.append(kw)
            elif kw:
                keywords_missed.append(kw)

        # Check forbidden keywords
        forbidden_found = []
        for kw in tc.forbidden_keywords:
            if kw and kw.lower() in reply_lower:
                forbidden_found.append(kw)

        # Pass criteria: intent matches AND at least 1 keyword found (if keywords specified) AND no forbidden
        has_keywords = len(keywords_missed) == 0 if tc.expected_keywords else True
        passed = intent_match and has_keywords and len(forbidden_found) == 0

        result = EvalResult(
            test_id=tc.id,
            question=tc.question,
            expected_intent=tc.expected_intent,
            actual_intent=state.get("intent", "unknown"),
            intent_match=intent_match,
            reply=reply[:300],
            keywords_found=keywords_found,
            keywords_missed=keywords_missed,
            forbidden_found=forbidden_found,
            latency_ms=round(latency, 1),
            passed=passed,
        )
        results.append(result)

        if verbose:
            status = "✅" if passed else "❌"
            print(f"  {status} {tc.id}: {tc.question[:50]}... | intent={tc.expected_intent}/{result.actual_intent} | {latency:.0f}ms")

    # Summary stats
    intent_correct = sum(1 for r in results if r.intent_match)
    overall_passed = sum(1 for r in results if r.passed)
    avg_latency = sum(r.latency_ms for r in results) / len(results)

    summary = {
        "total": len(results),
        "intent_accuracy": round(intent_correct / len(results) * 100, 1),
        "overall_pass_rate": round(overall_passed / len(results) * 100, 1),
        "avg_latency_ms": round(avg_latency, 1),
    }

    return results, summary


# ── Intent-only accuracy (already known from test cases above) ──


if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Set OPENAI_API_KEY to run evaluation")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("📊 Running Eval: 50 cross-border ecommerce test cases")
    print(f"{'='*60}\n")

    results, summary = run_evaluation(TEST_CASES, verbose=True)

    print(f"\n{'='*60}")
    print("📈 SUMMARY")
    print(f"{'='*60}")
    print(f"  Total cases:        {summary['total']}")
    print(f"  Intent accuracy:    {summary['intent_accuracy']}% ({int(summary['total'] * summary['intent_accuracy']/100)}/{summary['total']})")
    print(f"  Overall pass rate:  {summary['overall_pass_rate']}% ({int(summary['total'] * summary['overall_pass_rate']/100)}/{summary['total']})")
    print(f"  Avg latency:        {summary['avg_latency_ms']}ms")

    # Intent-level breakdown
    from collections import Counter
    intent_stats = Counter()
    intent_correct_stats = Counter()
    for r in results:
        intent_stats[r.expected_intent] += 1
        if r.intent_match:
            intent_correct_stats[r.expected_intent] += 1

    print(f"\n  Intent breakdown:")
    for intent in sorted(intent_stats.keys()):
        correct = intent_correct_stats[intent]
        total = intent_stats[intent]
        rate = round(correct/total*100, 1)
        names = {
            "product_inquiry": "产品咨询",
            "order_status": "订单状态",
            "shipping": "物流",
            "return_refund": "退换货",
            "complaint": "投诉",
            "other": "其他",
        }
        label = names.get(intent, intent)
        bar = "█" * int(rate / 10) + "░" * (10 - int(rate / 10))
        print(f"    {label:8s} {rate:5.1f}% [{bar}] ({correct}/{total})")

    # Failed cases
    failed = [r for r in results if not r.passed]
    if failed:
        print(f"\n  Failed cases ({len(failed)}):")
        for r in failed:
            print(f"    ❌ {r.test_id}: {r.question[:60]}")
            if not r.intent_match:
                print(f"       Intent mismatch: expected={r.expected_intent} got={r.actual_intent}")
            if r.keywords_missed:
                print(f"       Missing keywords: {r.keywords_missed}")
            if r.forbidden_found:
                print(f"       Forbidden found: {r.forbidden_found}")

    # Save detailed results
    with open("eval/results.json", "w") as f:
        json.dump({
            "summary": summary,
            "results": [{"id": r.test_id, "question": r.question, "passed": r.passed,
                         "intent_match": r.intent_match, "latency_ms": r.latency_ms} for r in results],
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Detailed results saved to eval/results.json")
