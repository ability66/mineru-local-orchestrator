from __future__ import annotations

from src.pipeline.issues import (
    detect_flowchart_issues,
    detect_flowchart_second_pass_issues,
    detect_seal_issues,
)
from src.pipeline.llm_adjudicator import build_issue_prompt_payload
from src.schema import (
    CanonicalBlock,
    CanonicalDocument,
    CaptionStructured,
    ImageTask,
    OcrRegion,
    ParsedLabel,
    StructuredLabel,
)


def test_detect_seal_issue_when_qwen_supplies_missing_ocr() -> None:
    image_task = ImageTask(
        image_id="img-1",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    mineru_document = CanonicalDocument(
        document_id="img-1",
        source="mineru",
        blocks=[
            CanonicalBlock(
                block_id="m1",
                page_idx=0,
                order_index=1,
                type="image",
                sub_type="seal",
                bbox=[100, 100, 300, 300],
                text="",
                content={"img_path": "data/demo.png"},
                source="mineru",
                caption_structured=CaptionStructured(brief=""),
            )
        ],
    )
    qwen_document = CanonicalDocument(
        document_id="img-1",
        source="qwen",
        blocks=[
            CanonicalBlock(
                block_id="q1",
                page_idx=0,
                order_index=1,
                type="image",
                sub_type="seal",
                bbox=[100, 100, 300, 300],
                text="某某公司印章",
                content={
                    "img_path": "data/demo.png",
                    "image_caption": ["某某公司印章"],
                },
                source="qwen",
                caption_structured=CaptionStructured(brief="某某公司印章"),
                ocr_regions=[
                    OcrRegion(role="seal", text="某某公司", confidence="high")
                ],
            )
        ],
    )

    issues = detect_seal_issues(
        image_task=image_task,
        mineru_document=mineru_document,
        qwen_document=qwen_document,
    )

    assert len(issues) == 1
    assert issues[0].issue_type == "seal_missing_ocr"
    assert issues[0].target_block_id == "m1"


def test_detect_seal_issue_when_qwen_marks_plain_image_as_seal() -> None:
    image_task = ImageTask(
        image_id="img-2",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    mineru_document = CanonicalDocument(
        document_id="img-2",
        source="mineru",
        blocks=[
            CanonicalBlock(
                block_id="m1",
                page_idx=0,
                order_index=1,
                type="image",
                bbox=[100, 100, 300, 300],
                text="",
                content={"img_path": "data/demo.png"},
                source="mineru",
                caption_structured=CaptionStructured(brief=""),
            )
        ],
    )
    qwen_document = CanonicalDocument(
        document_id="img-2",
        source="qwen",
        blocks=[
            CanonicalBlock(
                block_id="q1",
                page_idx=0,
                order_index=1,
                type="image",
                sub_type="seal",
                bbox=[100, 100, 300, 300],
                text="某某公司印章",
                content={
                    "img_path": "data/demo.png",
                    "image_caption": ["某某公司印章"],
                },
                source="qwen",
                caption_structured=CaptionStructured(brief="某某公司印章"),
            )
        ],
    )

    issues = detect_seal_issues(
        image_task=image_task,
        mineru_document=mineru_document,
        qwen_document=qwen_document,
    )

    assert len(issues) == 1
    assert issues[0].issue_type == "seal_type_disagreement"


def test_detect_flowchart_issue_reports_graph_conflicts_against_qwen_reference() -> (
    None
):
    image_task = ImageTask(
        image_id="img-flow-1",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    mineru_document = CanonicalDocument(
        document_id="img-flow-1",
        source="mineru",
        blocks=[
            CanonicalBlock(
                block_id="m1",
                page_idx=0,
                order_index=1,
                type="chart",
                sub_type="flowchart",
                bbox=[0, 0, 1000, 1000],
                text="流程图",
                content={"img_path": "data/demo.png", "content": "flowchart TD\nA-->B"},
                source="mineru",
                structured_label=StructuredLabel(
                    kind="mermaid",
                    content="flowchart TD\nA-->B",
                    format="mermaid",
                    source="mineru",
                ),
                caption_structured=CaptionStructured(brief="流程图"),
            )
        ],
    )
    qwen_document = CanonicalDocument(
        document_id="img-flow-1",
        source="qwen",
        blocks=[
            CanonicalBlock(
                block_id="q1",
                page_idx=0,
                order_index=1,
                type="chart",
                sub_type="flowchart",
                bbox=[0, 0, 1000, 1000],
                text="流程图",
                content={
                    "img_path": "data/demo.png",
                    "content": "flowchart TD\nA-->B\nB-->C",
                },
                source="qwen",
                structured_label=StructuredLabel(
                    kind="mermaid",
                    content="flowchart TD\nA-->B\nB-->C",
                    format="mermaid",
                    source="model",
                ),
                caption_structured=CaptionStructured(brief="流程图"),
            )
        ],
    )
    qwen_label = ParsedLabel(
        image_type="flowchart",
        caption="流程图",
        caption_structured=CaptionStructured(brief="流程图"),
        structured_label=StructuredLabel(
            kind="mermaid",
            content="flowchart TD\nA-->B\nB-->C",
            format="mermaid",
            source="model",
        ),
    )

    issues = detect_flowchart_issues(
        image_task=image_task,
        mineru_document=mineru_document,
        qwen_document=qwen_document,
        mineru_label=None,
        qwen_label=qwen_label,
    )

    assert issues
    assert all(issue.issue_type == "flowchart_graph_conflict" for issue in issues)
    assert all(issue.target_block_id == "m1" for issue in issues)
    assert any(
        issue.candidate_payload
        and issue.candidate_payload["reference_mermaid"] == "flowchart TD\nA-->B\nB-->C"
        for issue in issues
    )
    assert any(
        issue.candidate_payload
        and issue.candidate_payload["graph_diff"]["diff_kind"]
        in {"missing_node", "missing_edge"}
        for issue in issues
    )
    assert all(
        "current_graph" not in (issue.candidate_payload or {}) for issue in issues
    )
    assert all(
        "reference_graph" not in (issue.candidate_payload or {}) for issue in issues
    )
    assert all(
        "current_patch" not in (issue.candidate_payload or {}) for issue in issues
    )

    prompt_payload = build_issue_prompt_payload(issues[0], "flowchart_adjudication")
    assert "current_block" in prompt_payload
    assert "reference_block" in prompt_payload
    assert "current_excerpt" in prompt_payload
    assert "reference_excerpt" in prompt_payload
    assert "current_mermaid" not in prompt_payload
    assert "reference_mermaid" not in prompt_payload


def test_detect_flowchart_second_pass_issue_without_qwen_reference() -> None:
    image_task = ImageTask(
        image_id="img-flow-2",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    mineru_document = CanonicalDocument(
        document_id="img-flow-2",
        source="mineru",
        blocks=[
            CanonicalBlock(
                block_id="m1",
                page_idx=0,
                order_index=1,
                type="image",
                sub_type="flowchart",
                bbox=[0, 0, 1000, 1000],
                text="```mermaid\nflowchart TD\nA-->B\n```",
                content={"img_path": "data/demo.png"},
                source="mineru",
                caption_structured=CaptionStructured(brief="流程图"),
            )
        ],
    )

    issues = detect_flowchart_second_pass_issues(
        image_task=image_task,
        mineru_document=mineru_document,
        mineru_label=None,
    )

    assert len(issues) == 1
    assert issues[0].target_block_id == "m1"
    assert issues[0].candidate_payload is not None
    assert issues[0].candidate_payload["review_mode"] == "second_pass"
    assert (
        issues[0].candidate_payload["graph_diff"]["diff_kind"] == "second_pass_required"
    )

    prompt_payload = build_issue_prompt_payload(issues[0], "flowchart_adjudication")
    assert prompt_payload["review_mode"] == "second_pass"
    assert prompt_payload["reference_excerpt"] == ""
