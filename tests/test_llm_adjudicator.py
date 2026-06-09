from __future__ import annotations

from src.pipeline.llm_adjudicator import _parse_patch_decision, build_issue_prompt_payload
from src.schema import Issue, ModelOutput


def test_build_issue_prompt_payload_keeps_full_mermaid_for_small_flowcharts() -> None:
    small_mermaid = """flowchart TD
Start["开始"] --> Risk["家族史和/或高危因素"]
Risk -->|无| CT_Q["是否可以进行增强CT?"]
Risk -->|有| Genetic["遗传咨询"]
Genetic --> CT_Q
CT_Q -->|是| CT["CT"]
CT_Q -->|否| MRI["MRI"]
MRI --> Chest["胸腹部CT"]
Chest --> Local["局部进展期"]
Chest --> Met["合并转移"]
Local --> EUS["通过EUS活检"]
EUS --> Treat["局部/局部区域PC的治疗（图2）"]
Met --> Biopsy["转移灶或原发肿瘤的活检"]
Biopsy --> Adv["进展期或转移性PC的治疗（图3）"]"""
    issue = Issue(
        issue_id="flowchart-diff-m1-missing-node-1",
        issue_type="flowchart_graph_conflict",
        page_idx=0,
        target_block_id="m1",
        candidate_payload={
            "review_mode": "disagreement",
            "current_mermaid": small_mermaid,
            "reference_mermaid": small_mermaid,
            "graph_diff": {
                "diff_kind": "missing_node",
                "reference_node": {"text": "遗传咨询"},
            },
        },
        reasons=["flowchart_graph_conflict_detected"],
    )

    prompt_payload = build_issue_prompt_payload(issue, "flowchart_adjudication")

    assert prompt_payload["current_excerpt"] == small_mermaid
    assert prompt_payload["reference_excerpt"] == small_mermaid


def test_parse_patch_decision_converts_overreaching_flowchart_merge_to_qwen_fields_on_disagreement() -> None:
    issue = Issue(
        issue_id="flowchart-diff-m1-missing-node-1",
        issue_type="flowchart_graph_conflict",
        page_idx=0,
        target_block_id="m1",
        candidate_payload={
            "review_mode": "disagreement",
            "current_mermaid": 'flowchart TD\nA["开始"] --> B["判断"]\nB --> C["结束"]',
            "reference_mermaid": (
                'flowchart TD\nA["开始"] --> B["判断"]\n'
                'B --> C["结束"]\nB --> D["补充分支"]'
            ),
            "graph_diff": {
                "diff_kind": "missing_node",
                "reference_node": {"text": "补充分支"},
            },
        },
        reasons=["flowchart_graph_conflict_detected"],
    )
    output = ModelOutput(
        image_id="img-flow-overreach",
        model_name="qwen-test",
        success=True,
        raw_text="""{
  "issue_id": "flowchart-diff-m1-missing-node-1",
  "target_block_id": "m1",
  "decision": "merge",
  "patch": {
    "type": "chart",
    "sub_type": "flowchart",
    "content": {
      "content": "flowchart TD\\nA[\\"开始\\"] --> B[\\"判断\\"]\\nB --> C[\\"结束\\"]\\nB --> D[\\"补充分支\\"]\\nD --> E[\\"远端1\\"]\\nE --> F[\\"远端2\\"]\\nF --> G[\\"远端3\\"]"
    }
  },
  "reason": "rebuild"
}""",
    )

    decision = _parse_patch_decision(issue=issue, output=output)

    assert decision.decision == "use_qwen_fields"
    assert decision.patch == {}
    assert decision.reason == "qwen_flowchart_preferred_on_conflict"


def test_parse_patch_decision_prefers_qwen_on_flowchart_disagreement() -> None:
    issue = Issue(
        issue_id="flowchart-diff-m1-missing-node-2",
        issue_type="flowchart_graph_conflict",
        page_idx=0,
        target_block_id="m1",
        candidate_payload={
            "review_mode": "disagreement",
            "current_mermaid": (
                'flowchart TD\nA["开始"] --> B["判断"]\n'
                'B --> C["结束"]'
            ),
            "reference_mermaid": (
                'flowchart TD\nA["开始"] --> B["判断"]\n'
                'B --> C["结束"]\nB --> D["补充分支"]'
            ),
            "graph_diff": {
                "diff_kind": "missing_node",
                "reference_node": {"text": "补充分支"},
            },
        },
        reasons=["flowchart_graph_conflict_detected"],
    )
    output = ModelOutput(
        image_id="img-flow-qwen-first",
        model_name="qwen-test",
        success=True,
        raw_text="""{
  "issue_id": "flowchart-diff-m1-missing-node-2",
  "target_block_id": "m1",
  "decision": "keep_mineru",
  "patch": {},
  "reason": "reference excerpt incomplete"
}""",
    )

    decision = _parse_patch_decision(issue=issue, output=output)

    assert decision.decision == "use_qwen_fields"
    assert decision.patch == {}
    assert decision.reason == "qwen_flowchart_preferred_on_conflict"


def test_parse_patch_decision_rejects_false_positive_helper_node_conflict() -> None:
    issue = Issue(
        issue_id="flowchart-diff-m1-helper-node-1",
        issue_type="flowchart_graph_conflict",
        page_idx=0,
        target_block_id="m1",
        candidate_payload={
            "review_mode": "disagreement",
            "current_mermaid": (
                'flowchart TD\nCT["CT"] --> Other["其他发现或鉴别诊断"]\n'
                'CT --> Resectable["无转移，肿瘤可切除或临界可切除"]'
            ),
            "reference_mermaid": (
                'flowchart TD\nCTAlias["CT"] --> H[]:::hidden\n'
                'H --> OtherAlias["其他发现或鉴别诊断"]\n'
                'H --> ResectableAlias["无转移，肿瘤可切除或临界可切除"]\n'
                'classDef hidden display:none;'
            ),
            "graph_diff": {
                "diff_kind": "missing_node",
                "reference_node": {"text": "H"},
            },
        },
        reasons=["flowchart_graph_conflict_detected"],
    )
    output = ModelOutput(
        image_id="img-flow-helper-node",
        model_name="qwen-test",
        success=True,
        raw_text="""{
  "issue_id": "flowchart-diff-m1-helper-node-1",
  "target_block_id": "m1",
  "decision": "keep_mineru",
  "patch": {},
  "reason": "reference excerpt incomplete"
}""",
    )

    decision = _parse_patch_decision(issue=issue, output=output)

    assert decision.decision == "reject_issue"
    assert decision.patch == {}
    assert decision.reason == "flowchart_conflict_false_positive"
