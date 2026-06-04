from __future__ import annotations

from src.pipeline.adjudicator import adjudicate_documents
from src.schema import (
    CanonicalBlock,
    CanonicalDocument,
    CaptionStructured,
    ImageTask,
    Issue,
    ModelOutput,
    OcrRegion,
    PatchDecision,
    ParsedLabel,
    StructuredLabel,
)


def test_adjudicator_injects_seal_from_qwen_label() -> None:
    image_task = ImageTask(
        image_id="img-1",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    mineru_document = CanonicalDocument(
        document_id="img-1",
        source="mineru",
        backend="mineru",
        page_count=1,
        blocks=[
            CanonicalBlock(
                block_id="b1",
                page_idx=0,
                order_index=1,
                type="image",
                bbox=[100, 100, 300, 300],
                text="",
                content={"img_path": "data/demo.png"},
                source="mineru",
            )
        ],
    )
    qwen_document = CanonicalDocument(
        document_id="img-1",
        source="qwen",
        backend="qwen",
        page_count=1,
        blocks=[
            CanonicalBlock(
                block_id="q1",
                page_idx=0,
                order_index=1,
                type="image",
                sub_type="seal",
                bbox=[110, 110, 310, 310],
                text="某某公司印章",
                content={
                    "img_path": "data/demo.png",
                    "image_caption": ["某某公司印章"],
                },
                source="qwen",
                ocr_regions=[
                    OcrRegion(role="seal", text="某某公司", confidence="high")
                ],
            )
        ],
    )
    qwen_label = ParsedLabel(
        image_type="document",
        caption="某某公司印章",
        caption_structured=CaptionStructured(
            brief="某某公司印章",
            visual_type="document",
            main_subject="某某公司印章",
            confidence="high",
        ),
        structured_label=StructuredLabel(
            kind="none", content="", format="none", source="none"
        ),
        ocr_regions=[OcrRegion(role="seal", text="某某公司", confidence="high")],
    )

    artifact = adjudicate_documents(
        image_task=image_task,
        mineru_document=mineru_document,
        qwen_document=qwen_document,
        mineru_label=None,
        qwen_label=qwen_label,
        mineru_output=ModelOutput(
            image_id="img-1", model_name="mineru", success=True, raw_text="{}"
        ),
        qwen_output=ModelOutput(
            image_id="img-1", model_name="qwen", success=True, raw_text="{}"
        ),
    )

    block = artifact.final_document.blocks[0]
    assert block.type == "image"
    assert block.sub_type == "seal"
    assert any(region.role == "seal" for region in block.ocr_regions)
    assert artifact.review_required is True


def test_adjudicator_keeps_mineru_primary_structure_when_reviewing_conflict() -> None:
    image_task = ImageTask(
        image_id="img-2",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    mineru_document = CanonicalDocument(
        document_id="img-2",
        source="mineru",
        backend="mineru",
        page_count=1,
        blocks=[
            CanonicalBlock(
                block_id="m1",
                page_idx=0,
                order_index=1,
                type="table",
                bbox=[0, 0, 1000, 1000],
                text="表格标题",
                content={
                    "table_body": "|a|b|",
                    "table_caption": ["表格标题"],
                    "img_path": "data/demo.png",
                },
                source="mineru",
                structured_label=StructuredLabel(
                    kind="table",
                    content="|a|b|",
                    format="markdown",
                    source="mineru",
                ),
                caption_structured=CaptionStructured(
                    brief="表格标题", visual_type="table"
                ),
            )
        ],
    )
    qwen_document = CanonicalDocument(
        document_id="img-2",
        source="qwen",
        backend="qwen",
        page_count=1,
        blocks=[
            CanonicalBlock(
                block_id="q1",
                page_idx=0,
                order_index=1,
                type="chart",
                bbox=[0, 0, 1000, 1000],
                text="图表标题",
                content={
                    "content": "趋势说明",
                    "chart_caption": ["图表标题"],
                    "img_path": "data/demo.png",
                },
                source="qwen",
                caption_structured=CaptionStructured(
                    brief="图表标题", visual_type="chart"
                ),
            )
        ],
    )
    mineru_label = ParsedLabel(
        image_type="table",
        caption="表格标题",
        caption_structured=CaptionStructured(
            brief="表格标题",
            visual_type="table",
            main_subject="表格标题",
            confidence="high",
        ),
        structured_label=StructuredLabel(
            kind="table", content="|a|b|", format="markdown", source="mineru"
        ),
    )
    qwen_label = ParsedLabel(
        image_type="chart",
        caption="图表标题",
        caption_structured=CaptionStructured(
            brief="图表标题",
            visual_type="chart",
            main_subject="图表标题",
            confidence="high",
        ),
        structured_label=StructuredLabel(
            kind="text", content="趋势说明", format="plain_text", source="model"
        ),
    )

    artifact = adjudicate_documents(
        image_task=image_task,
        mineru_document=mineru_document,
        qwen_document=qwen_document,
        mineru_label=mineru_label,
        qwen_label=qwen_label,
        mineru_output=ModelOutput(
            image_id="img-2", model_name="mineru", success=True, raw_text="{}"
        ),
        qwen_output=ModelOutput(
            image_id="img-2", model_name="qwen", success=True, raw_text="{}"
        ),
    )

    assert artifact.review_required is True
    assert artifact.final_document.blocks[0].type == "table"
    assert artifact.final_document.blocks[0].structured_label.kind == "table"


def test_adjudicator_auto_accepts_stamp_mode_when_no_seal_issues() -> None:
    image_task = ImageTask(
        image_id="img-3",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    mineru_document = CanonicalDocument(
        document_id="img-3",
        source="mineru",
        backend="mineru",
        page_count=1,
        blocks=[
            CanonicalBlock(
                block_id="m1",
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
                source="mineru",
                ocr_regions=[
                    OcrRegion(role="seal", text="某某公司", confidence="high")
                ],
                caption_structured=CaptionStructured(brief="某某公司印章"),
            )
        ],
    )
    qwen_document = CanonicalDocument(
        document_id="img-3",
        source="qwen",
        backend="qwen",
        page_count=1,
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
                ocr_regions=[
                    OcrRegion(role="seal", text="某某公司", confidence="high")
                ],
                caption_structured=CaptionStructured(brief="某某公司印章"),
            )
        ],
    )
    mineru_label = ParsedLabel(
        image_type="document",
        caption="某某公司印章",
        caption_structured=CaptionStructured(
            brief="某某公司印章",
            visual_type="document",
            main_subject="某某公司印章",
            confidence="high",
        ),
        structured_label=StructuredLabel(
            kind="none", content="", format="none", source="none"
        ),
        ocr_regions=[OcrRegion(role="seal", text="某某公司", confidence="high")],
    )
    qwen_label = ParsedLabel(
        image_type="document",
        caption="某某公司印章",
        caption_structured=CaptionStructured(
            brief="某某公司印章",
            visual_type="document",
            main_subject="某某公司印章",
            confidence="high",
        ),
        structured_label=StructuredLabel(
            kind="none", content="", format="none", source="none"
        ),
        ocr_regions=[OcrRegion(role="seal", text="某某公司", confidence="high")],
    )

    artifact = adjudicate_documents(
        image_task=image_task,
        mineru_document=mineru_document,
        qwen_document=qwen_document,
        mineru_label=mineru_label,
        qwen_label=qwen_label,
        mineru_output=ModelOutput(
            image_id="img-3", model_name="mineru", success=True, raw_text="{}"
        ),
        qwen_output=ModelOutput(
            image_id="img-3", model_name="qwen", success=True, raw_text="{}"
        ),
        issues=[],
        patch_decisions=[],
    )

    assert artifact.consensus is not None
    assert artifact.consensus.decision == "accepted"
    assert artifact.review_required is False


def test_adjudicator_auto_accepts_flowchart_mode_when_no_graph_issues() -> None:
    image_task = ImageTask(
        image_id="img-flow-accept",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    mineru_document = CanonicalDocument(
        document_id="img-flow-accept",
        source="mineru",
        backend="mineru",
        page_count=1,
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
                    source="fused_graph",
                ),
                caption_structured=CaptionStructured(brief="流程图"),
            )
        ],
    )
    qwen_document = CanonicalDocument(
        document_id="img-flow-accept",
        source="qwen",
        backend="qwen",
        page_count=1,
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
    mineru_label = ParsedLabel(
        image_type="flowchart",
        caption="流程图",
        caption_structured=CaptionStructured(brief="流程图"),
        structured_label=StructuredLabel(
            kind="mermaid",
            content="flowchart TD\nA-->B",
            format="mermaid",
            source="fused_graph",
        ),
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
    artifact = adjudicate_documents(
        image_task=image_task,
        mineru_document=mineru_document,
        qwen_document=qwen_document,
        mineru_label=mineru_label,
        qwen_label=qwen_label,
        mineru_output=ModelOutput(
            image_id="img-flow-accept", model_name="mineru", success=True, raw_text="{}"
        ),
        qwen_output=ModelOutput(
            image_id="img-flow-accept", model_name="qwen", success=True, raw_text="{}"
        ),
        issues=[],
        patch_decisions=[],
    )

    assert artifact.consensus is not None
    assert artifact.consensus.decision == "accepted"
    assert artifact.review_required is False


def test_adjudicator_reviews_flowchart_mode_when_final_mermaid_missing() -> None:
    image_task = ImageTask(
        image_id="img-flow-review",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    mineru_document = CanonicalDocument(
        document_id="img-flow-review",
        source="mineru",
        backend="mineru",
        page_count=1,
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
        document_id="img-flow-review",
        source="qwen",
        backend="qwen",
        page_count=1,
        blocks=[
            CanonicalBlock(
                block_id="q1",
                page_idx=0,
                order_index=1,
                type="chart",
                sub_type="flowchart",
                bbox=[0, 0, 1000, 1000],
                text="流程图",
                content={"img_path": "data/demo.png", "chart_caption": ["流程图说明"]},
                source="qwen",
                caption_structured=CaptionStructured(brief="流程图"),
            )
        ],
    )
    mineru_label = ParsedLabel(
        image_type="flowchart",
        caption="流程图",
        caption_structured=CaptionStructured(brief="流程图"),
    )
    qwen_label = ParsedLabel(
        image_type="flowchart",
        caption="流程图",
        caption_structured=CaptionStructured(brief="流程图"),
    )
    artifact = adjudicate_documents(
        image_task=image_task,
        mineru_document=mineru_document,
        qwen_document=qwen_document,
        mineru_label=mineru_label,
        qwen_label=qwen_label,
        mineru_output=ModelOutput(
            image_id="img-flow-review", model_name="mineru", success=True, raw_text="{}"
        ),
        qwen_output=ModelOutput(
            image_id="img-flow-review", model_name="qwen", success=True, raw_text="{}"
        ),
        issues=[
            Issue(
                issue_id="flowchart-review-m1",
                issue_type="flowchart_graph_conflict",
                page_idx=0,
                target_block_id="m1",
                reasons=["flowchart_graph_conflict_detected"],
            )
        ],
        patch_decisions=[
            PatchDecision(
                issue_id="flowchart-review-m1",
                target_block_id="m1",
                decision="keep_mineru",
                patch={},
                reason="当前无法确认候选",
            )
        ],
    )

    assert artifact.consensus is not None
    assert artifact.consensus.decision == "review"
    assert artifact.review_required is True


def test_adjudicator_accepts_flowchart_when_qwen_is_incomplete_and_mineru_remains_valid() -> (
    None
):
    image_task = ImageTask(
        image_id="img-flow-fallback",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    mineru_document = CanonicalDocument(
        document_id="img-flow-fallback",
        source="mineru",
        backend="mineru",
        page_count=1,
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
                    source="model",
                ),
                caption_structured=CaptionStructured(brief="流程图"),
            )
        ],
    )
    qwen_document = CanonicalDocument(
        document_id="img-flow-fallback",
        source="qwen",
        backend="qwen",
        page_count=1,
        blocks=[],
    )
    mineru_label = ParsedLabel(
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
    artifact = adjudicate_documents(
        image_task=image_task,
        mineru_document=mineru_document,
        qwen_document=qwen_document,
        mineru_label=mineru_label,
        qwen_label=None,
        mineru_output=ModelOutput(
            image_id="img-flow-fallback",
            model_name="mineru",
            success=True,
            raw_text="{}",
        ),
        qwen_output=ModelOutput(
            image_id="img-flow-fallback",
            model_name="qwen",
            success=True,
            raw_text="",
        ),
        issues=[
            Issue(
                issue_id="flowchart-diff-m1-missing-node-1",
                issue_type="flowchart_graph_conflict",
                page_idx=0,
                target_block_id="m1",
                reasons=["flowchart_graph_conflict_detected"],
            )
        ],
        patch_decisions=[
            PatchDecision(
                issue_id="flowchart-diff-m1-missing-node-1",
                target_block_id="m1",
                decision="keep_mineru",
                patch={},
                reason="qwen_flowchart_incomplete",
            )
        ],
    )

    assert artifact.consensus is not None
    assert artifact.consensus.decision == "accepted"
    assert artifact.review_required is False


def test_adjudicator_accepts_flowchart_with_stage2_patch_without_first_stage_qwen() -> (
    None
):
    image_task = ImageTask(
        image_id="img-flow-second-pass",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    mineru_document = CanonicalDocument(
        document_id="img-flow-second-pass",
        source="mineru",
        backend="mineru",
        page_count=1,
        blocks=[
            CanonicalBlock(
                block_id="m1",
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
                source="mineru",
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
    qwen_document = CanonicalDocument(
        document_id="img-flow-second-pass",
        source="qwen_second_pass",
        backend="qwen",
        page_count=1,
        blocks=[],
    )
    mineru_label = ParsedLabel(
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
    artifact = adjudicate_documents(
        image_task=image_task,
        mineru_document=mineru_document,
        qwen_document=qwen_document,
        mineru_label=mineru_label,
        qwen_label=None,
        mineru_output=ModelOutput(
            image_id="img-flow-second-pass",
            model_name="mineru",
            success=True,
            raw_text="{}",
        ),
        qwen_output=None,
        issues=[
            Issue(
                issue_id="flowchart-second-pass-m1",
                issue_type="flowchart_graph_conflict",
                page_idx=0,
                target_block_id="m1",
                reasons=["flowchart_requires_qwen_second_pass"],
            )
        ],
        patch_decisions=[
            PatchDecision(
                issue_id="flowchart-second-pass-m1",
                target_block_id="m1",
                decision="merge",
                patch={
                    "type": "chart",
                    "sub_type": "flowchart",
                    "content": {"content": "flowchart TD\nA-->B\nB-->D"},
                },
                reason="llm_patch_applied",
            )
        ],
    )

    assert artifact.consensus is not None
    assert artifact.consensus.decision == "accepted"
    assert artifact.review_required is False


def test_adjudicator_selects_qwen_flowchart_as_final_without_mutating_mineru() -> None:
    image_task = ImageTask(
        image_id="img-flow-select-qwen",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    mineru_document = CanonicalDocument(
        document_id="img-flow-select-qwen",
        source="mineru",
        backend="mineru",
        page_count=1,
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
                    source="model",
                ),
                caption_structured=CaptionStructured(brief="MinerU流程图"),
            )
        ],
    )
    qwen_document = CanonicalDocument(
        document_id="img-flow-select-qwen",
        source="qwen",
        backend="qwen",
        page_count=1,
        blocks=[
            CanonicalBlock(
                block_id="q1",
                page_idx=0,
                order_index=1,
                type="chart",
                sub_type="flowchart",
                bbox=[0, 0, 1000, 1000],
                text="流程图",
                content={"img_path": "data/demo.png", "content": "flowchart TD\nA-->B\nB-->C"},
                source="qwen",
                structured_label=StructuredLabel(
                    kind="mermaid",
                    content="flowchart TD\nA-->B\nB-->C",
                    format="mermaid",
                    source="model",
                ),
                caption_structured=CaptionStructured(brief="Qwen流程图"),
            )
        ],
    )
    mineru_label = ParsedLabel(
        image_type="flowchart",
        caption="MinerU流程图",
        caption_structured=CaptionStructured(brief="MinerU流程图"),
        structured_label=StructuredLabel(
            kind="mermaid",
            content="flowchart TD\nA-->B",
            format="mermaid",
            source="model",
        ),
    )
    qwen_label = ParsedLabel(
        image_type="flowchart",
        caption="Qwen流程图",
        caption_structured=CaptionStructured(brief="Qwen流程图"),
        structured_label=StructuredLabel(
            kind="mermaid",
            content="flowchart TD\nA-->B\nB-->C",
            format="mermaid",
            source="model",
        ),
    )

    artifact = adjudicate_documents(
        image_task=image_task,
        mineru_document=mineru_document,
        qwen_document=qwen_document,
        mineru_label=mineru_label,
        qwen_label=qwen_label,
        mineru_output=ModelOutput(
            image_id="img-flow-select-qwen",
            model_name="mineru",
            success=True,
            raw_text="{}",
        ),
        qwen_output=ModelOutput(
            image_id="img-flow-select-qwen",
            model_name="glm-as-qwen",
            success=True,
            raw_text="{}",
        ),
        issues=[
            Issue(
                issue_id="flowchart-diff-m1-missing-node-1",
                issue_type="flowchart_graph_conflict",
                page_idx=0,
                target_block_id="m1",
                reasons=["flowchart_graph_conflict_detected"],
            )
        ],
        patch_decisions=[
            PatchDecision(
                issue_id="flowchart-diff-m1-missing-node-1",
                target_block_id="m1",
                decision="use_qwen_fields",
                patch={},
                reason="qwen_flowchart_preferred_on_conflict",
            )
        ],
    )

    assert artifact.consensus is not None
    assert artifact.consensus.decision == "accepted"
    assert artifact.final_document.source == "qwen"
    assert artifact.final_document.raw_metadata["selected_output_role"] == "qwen"
    assert artifact.final_document.blocks[0].content["content"] == "flowchart TD\nA-->B\nB-->C"
    assert mineru_document.blocks[0].content["content"] == "flowchart TD\nA-->B"


def test_adjudicator_keeps_stamp_mode_in_review_when_issue_unresolved() -> None:
    image_task = ImageTask(
        image_id="img-4",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    mineru_document = CanonicalDocument(
        document_id="img-4",
        source="mineru",
        backend="mineru",
        page_count=1,
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
            )
        ],
    )
    qwen_document = CanonicalDocument(
        document_id="img-4",
        source="qwen",
        backend="qwen",
        page_count=1,
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
                ocr_regions=[
                    OcrRegion(role="seal", text="某某公司", confidence="high")
                ],
            )
        ],
    )
    mineru_label = ParsedLabel(
        image_type="document",
        caption="",
        structured_label=StructuredLabel(
            kind="none", content="", format="none", source="none"
        ),
    )
    qwen_label = ParsedLabel(
        image_type="document",
        caption="某某公司印章",
        caption_structured=CaptionStructured(
            brief="某某公司印章",
            visual_type="document",
            main_subject="某某公司印章",
            confidence="high",
        ),
        structured_label=StructuredLabel(
            kind="none", content="", format="none", source="none"
        ),
        ocr_regions=[OcrRegion(role="seal", text="某某公司", confidence="high")],
    )

    artifact = adjudicate_documents(
        image_task=image_task,
        mineru_document=mineru_document,
        qwen_document=qwen_document,
        mineru_label=mineru_label,
        qwen_label=qwen_label,
        mineru_output=ModelOutput(
            image_id="img-4", model_name="mineru", success=True, raw_text="{}"
        ),
        qwen_output=ModelOutput(
            image_id="img-4", model_name="qwen", success=True, raw_text="{}"
        ),
        issues=[
            Issue(
                issue_id="seal-missing-ocr-m1",
                issue_type="seal_missing_ocr",
                page_idx=0,
                target_block_id="m1",
            )
        ],
        patch_decisions=[
            PatchDecision(
                issue_id="seal-missing-ocr-m1",
                target_block_id="m1",
                decision="keep_mineru",
                patch={},
                reason="llm_patch_unavailable",
            )
        ],
    )

    assert artifact.consensus is not None
    assert artifact.consensus.decision == "review"
    assert artifact.review_required is True
