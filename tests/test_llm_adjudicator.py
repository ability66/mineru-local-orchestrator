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

    assert prompt_payload["current_mermaid"] == small_mermaid
    assert prompt_payload["reference_mermaid"] == small_mermaid


def test_parse_patch_decision_rejects_overreaching_flowchart_merge_on_disagreement() -> None:
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

    assert decision.decision == "keep_mineru"
    assert decision.patch == {}
    assert decision.reason == "llm_patch_overreach_on_disagreement"


def test_parse_patch_decision_keeps_model_choice_on_flowchart_disagreement() -> None:
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

    assert decision.decision == "keep_mineru"
    assert decision.patch == {}
    assert decision.reason == "reference excerpt incomplete"


def test_parse_patch_decision_accepts_safe_flowchart_merge_on_disagreement() -> None:
    issue = Issue(
        issue_id="flowchart-diff-m1-safe-merge-1",
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
        image_id="img-flow-safe-merge",
        model_name="qwen-test",
        success=True,
        raw_text="""{
  "issue_id": "flowchart-diff-m1-safe-merge-1",
  "target_block_id": "m1",
  "decision": "merge",
  "patch": {
    "type": "chart",
    "sub_type": "flowchart",
    "content": {
      "content": "flowchart TD\\nA[\\"开始\\"] --> B[\\"判断\\"]\\nB --> C[\\"结束\\"]\\nB --> D[\\"补充分支\\"]"
    }
  },
  "reason": "merge two correct branches"
}""",
    )

    decision = _parse_patch_decision(issue=issue, output=output)

    assert decision.decision == "merge"
    assert decision.patch["content"]["content"].startswith("flowchart TD")
    assert decision.reason == "merge two correct branches"


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


def test_build_issue_prompt_payload_for_table_contains_similarity_context() -> None:
    issue = Issue(
        issue_id="table-m1",
        issue_type="table_conflict",
        page_idx=0,
        target_block_id="m1",
        candidate_payload={
            "candidates": [
                {
                    "candidate_id": "mineru",
                    "table_format": "markdown",
                    "table_content": "| A |\n| --- |\n| 1 |",
                },
                {
                    "candidate_id": "paddle",
                    "table_format": "markdown",
                    "table_content": "| A |\n| --- |\n| 1 |",
                },
            ],
            "pairwise_scores": [
                {"left": "mineru", "right": "paddle", "score": 0.97, "metrics": {}}
            ],
            "pairwise_matrix": {
                "mineru": {"mineru": 1.0, "paddle": 0.97},
                "paddle": {"mineru": 0.97, "paddle": 1.0},
            },
            "consensus_diagnostics": {
                "stable_consensus": False,
                "consensus_kind": "none",
            },
        },
        reasons=["no_stable_table_consensus"],
    )

    prompt_payload = build_issue_prompt_payload(issue, "table_adjudication")

    assert prompt_payload["review_mode"] == "table_disagreement"
    assert len(prompt_payload["candidates"]) == 2
    assert prompt_payload["pairwise_matrix"]["mineru"]["paddle"] == 0.97
    assert prompt_payload["consensus_diagnostics"]["consensus_kind"] == "none"


def test_build_issue_prompt_payload_for_chart_table_second_pass_requires_final_table() -> None:
    issue = Issue(
        issue_id="table-m1",
        issue_type="table_conflict",
        page_idx=0,
        target_block_id="m1",
        candidate_payload={
            "review_mode": "chart_table_second_pass",
            "branch_mode": "chart_table",
            "forced_second_pass": True,
            "must_output_final_table": True,
            "must_include_caption": True,
            "final_table_target": {
                "type": "table",
                "content_key": "table_body",
                "caption_key": "table_caption",
            },
            "task_instruction": "请直接输出最终表格与 caption。",
            "candidates": [
                {
                    "candidate_id": "mineru",
                    "table_format": "markdown",
                    "table_content": "| A | B |\n| --- | --- |\n| 1 | 2 |",
                    "caption": "候选表格",
                }
            ],
        },
        reasons=["chart_table_requires_qwen_second_pass"],
    )

    prompt_payload = build_issue_prompt_payload(issue, "table_adjudication")

    assert prompt_payload["review_mode"] == "chart_table_second_pass"
    assert prompt_payload["branch_mode"] == "chart_table"
    assert prompt_payload["forced_second_pass"] is True
    assert prompt_payload["must_output_final_table"] is True
    assert prompt_payload["must_include_caption"] is True
    assert prompt_payload["final_table_target"]["content_key"] == "table_body"
