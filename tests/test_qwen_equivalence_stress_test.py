from __future__ import annotations

from eval_dataset.mermaid_td_f1.qwen_equivalence_stress_test import (
    parse_qwen_case,
    should_stop,
)


def test_parse_qwen_case_accepts_fenced_json() -> None:
    payload = """```json
{
  "scenario": "采购流程",
  "gold_mermaid": "flowchart TD\\nA[开始] --> B[审批]",
  "pred_mermaid": "flowchart TD\\nX[开始] --> Y[审批]",
  "equivalence_reason": ["节点文字一致", "仅 ID 重命名"]
}
```"""

    case = parse_qwen_case(payload)

    assert case["scenario"] == "采购流程"
    assert case["gold_mermaid"].startswith("flowchart TD")
    assert case["pred_mermaid"].startswith("flowchart TD")
    assert case["equivalence_reason"] == ["节点文字一致", "仅 ID 重命名"]


def test_parse_qwen_case_tolerates_unescaped_multiline_mermaid_strings() -> None:
    payload = """{
  "scenario": "贷款流程",
  "gold_mermaid": "flowchart TD
A[开始] --> B[结束]",
  "pred_mermaid": "```mermaid
flowchart TD
X[开始] --> Y[结束]
```",
  "equivalence_reason": ["起止节点一致"]
}"""

    case = parse_qwen_case(payload)

    assert case["gold_mermaid"] == "flowchart TD\nA[开始] --> B[结束]"
    assert case["pred_mermaid"] == "flowchart TD\nX[开始] --> Y[结束]"


def test_should_stop_when_scores_below_threshold() -> None:
    result = {
        "parse_valid": True,
        "structure_f1": 1.0,
        "semantic_f1": 0.8,
        "binding_f1": 1.0,
        "final_td_f1": 0.9,
    }

    assert should_stop(result=result, error_message="", fail_threshold=0.999999) is True
