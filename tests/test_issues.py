from __future__ import annotations

from src.graph_fusion import FusedGraphResult
from src.pipeline.issues import detect_flowchart_issues, detect_seal_issues
from src.schema import CanonicalBlock, CanonicalDocument, CaptionStructured, ImageTask, OcrRegion, ParsedLabel, StructuredLabel


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
                content={"img_path": "data/demo.png", "image_caption": ["某某公司印章"]},
                source="qwen",
                caption_structured=CaptionStructured(brief="某某公司印章"),
                ocr_regions=[OcrRegion(role="seal", text="某某公司", confidence="high")],
            )
        ],
    )

    issues = detect_seal_issues(image_task=image_task, mineru_document=mineru_document, qwen_document=qwen_document)

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
                content={"img_path": "data/demo.png", "image_caption": ["某某公司印章"]},
                source="qwen",
                caption_structured=CaptionStructured(brief="某某公司印章"),
            )
        ],
    )

    issues = detect_seal_issues(image_task=image_task, mineru_document=mineru_document, qwen_document=qwen_document)

    assert len(issues) == 1
    assert issues[0].issue_type == "seal_type_disagreement"


def test_detect_flowchart_issue_includes_graph_fusion_candidate() -> None:
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
                content={"img_path": "data/demo.png", "chart_caption": ["流程图"]},
                source="mineru",
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
                content={"img_path": "data/demo.png", "content": "flowchart TD\nA-->B"},
                source="qwen",
                structured_label=StructuredLabel(
                    kind="mermaid",
                    content="flowchart TD\nA-->B",
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
            content="flowchart TD\nA-->B",
            format="mermaid",
            source="model",
        ),
    )
    graph_fusion_result = FusedGraphResult(
        mermaid="flowchart TD\nA-->B",
        fusion_method="visual_order",
        fusion_status="fused",
        graph_confidence=0.92,
    )

    issues = detect_flowchart_issues(
        image_task=image_task,
        mineru_document=mineru_document,
        qwen_document=qwen_document,
        mineru_label=None,
        qwen_label=qwen_label,
        graph_fusion_result=graph_fusion_result,
    )

    assert len(issues) == 1
    assert issues[0].issue_type == "flowchart_candidate_review"
    assert issues[0].target_block_id == "m1"
    assert issues[0].candidate_payload is not None
    assert issues[0].candidate_payload["candidate_mermaid"] == "flowchart TD\nA-->B"
