from __future__ import annotations

from src.pipeline.adjudicator import adjudicate_documents, analyze_table_bundles
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
    SealSelectionDecision,
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
        image_type="seal",
        caption="某某公司印章",
        caption_structured=CaptionStructured(
            brief="某某公司印章",
            visual_type="seal",
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
        image_type="seal",
        caption="某某公司印章",
        caption_structured=CaptionStructured(
            brief="某某公司印章",
            visual_type="seal",
            main_subject="某某公司印章",
            confidence="high",
        ),
        structured_label=StructuredLabel(
            kind="none", content="", format="none", source="none"
        ),
        ocr_regions=[OcrRegion(role="seal", text="某某公司", confidence="high")],
    )
    qwen_label = ParsedLabel(
        image_type="seal",
        caption="某某公司印章",
        caption_structured=CaptionStructured(
            brief="某某公司印章",
            visual_type="seal",
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
        image_type="seal",
        caption="某某公司印章",
        caption_structured=CaptionStructured(
            brief="某某公司印章",
            visual_type="seal",
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


def test_adjudicator_uses_projected_single_block_view_for_seal_comparison() -> None:
    image_task = ImageTask(
        image_id="img-seal-projected-compare",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    mineru_document = CanonicalDocument(
        document_id="img-seal-projected-compare",
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
                bbox=[0, 0, 999, 999],
                text="上海日轲电子有限公司",
                content={
                    "img_path": "data/demo.png",
                    "image_caption": ["上海日轲电子有限公司"],
                },
                source="mineru",
                caption_structured=CaptionStructured(brief="上海日轲电子有限公司"),
                ocr_regions=[OcrRegion(role="seal", text="上海日轲电子有限公司")],
            ),
            CanonicalBlock(
                block_id="m2",
                page_idx=0,
                order_index=2,
                type="paragraph",
                bbox=[194, 330, 543, 384],
                text="4541982082",
                content={
                    "paragraph_content": [{"type": "text", "content": "4541982082"}]
                },
                source="mineru",
            ),
            CanonicalBlock(
                block_id="m3",
                page_idx=0,
                order_index=3,
                type="image",
                sub_type="natural_image",
                bbox=[361, 390, 641, 644],
                text="Red hammer and sickle symbol on white background (no text or numbers)",
                content={
                    "img_path": "data/demo.png",
                    "image_caption": [
                        "Red hammer and sickle symbol on white background (no text or numbers)"
                    ],
                },
                source="mineru",
            ),
        ],
    )
    qwen_document = mineru_document.model_copy(deep=True)
    qwen_document.source = "qwen"
    qwen_document.backend = "qwen"
    for block in qwen_document.blocks:
        block.source = "qwen"

    mineru_label = ParsedLabel(
        image_type="seal",
        caption="上海日轲电子有限公司",
        caption_structured=CaptionStructured(
            brief="上海日轲电子有限公司",
            visual_type="seal",
            main_subject="上海日轲电子有限公司",
            confidence="high",
        ),
        structured_label=StructuredLabel(
            kind="none", content="", format="none", source="none"
        ),
        ocr_regions=[OcrRegion(role="seal", text="上海日轲电子有限公司")],
    )
    qwen_label = ParsedLabel(
        image_type="seal",
        caption="上海日轲电子有限公司",
        caption_structured=CaptionStructured(
            brief="上海日轲电子有限公司",
            visual_type="seal",
            main_subject="上海日轲电子有限公司",
            confidence="high",
        ),
        structured_label=StructuredLabel(
            kind="none", content="", format="none", source="none"
        ),
        ocr_regions=[OcrRegion(role="seal", text="上海日轲电子有限公司")],
    )

    artifact = adjudicate_documents(
        image_task=image_task,
        mineru_document=mineru_document,
        qwen_document=qwen_document,
        mineru_label=mineru_label,
        qwen_label=qwen_label,
        mineru_output=ModelOutput(
            image_id="img-seal-projected-compare",
            model_name="mineru",
            success=True,
            raw_text="{}",
        ),
        qwen_output=ModelOutput(
            image_id="img-seal-projected-compare",
            model_name="qwen",
            success=True,
            raw_text="{}",
        ),
        issues=[],
        patch_decisions=[],
    )

    assert artifact.matched_block_count == 1
    assert artifact.added_qwen_block_count == 0
    assert artifact.final_document.raw_metadata["comparison_view"] == "single_block_projection"
    assert len(artifact.final_document.blocks) == 3


def test_adjudicator_selects_auxiliary_seal_candidate_as_final() -> None:
    image_task = ImageTask(
        image_id="img-seal-select-aux",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    mineru_document = CanonicalDocument(
        document_id="img-seal-select-aux",
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
                bbox=[0, 0, 999, 999],
                text="上海日轲电子有限公司",
                content={"img_path": "data/demo.png", "image_caption": ["上海日轲电子有限公司"]},
                source="mineru",
                caption_structured=CaptionStructured(brief="上海日轲电子有限公司"),
            )
        ],
    )
    paddle_document = CanonicalDocument(
        document_id="img-seal-select-aux",
        source="paddle",
        backend="paddle",
        page_count=1,
        blocks=[
            CanonicalBlock(
                block_id="p1",
                page_idx=0,
                order_index=1,
                type="image",
                sub_type="seal",
                bbox=[0, 0, 999, 999],
                text="上海日轲电子有限公司\n\n4541982082",
                content={
                    "img_path": "data/demo.png",
                    "image_caption": ["上海日轲电子有限公司\n\n4541982082"],
                },
                source="paddle",
                caption_structured=CaptionStructured(brief="上海日轲电子有限公司"),
            )
        ],
    )
    paddle_label = ParsedLabel(
        image_type="seal",
        caption="上海日轲电子有限公司",
        caption_structured=CaptionStructured(
            brief="上海日轲电子有限公司",
            visual_type="seal",
            main_subject="上海日轲电子有限公司",
            confidence="high",
        ),
        structured_label=StructuredLabel(
            kind="none", content="", format="none", source="none"
        ),
    )

    artifact = adjudicate_documents(
        image_task=image_task,
        mineru_document=mineru_document,
        qwen_document=CanonicalDocument(
            document_id="img-seal-select-aux",
            source="qwen_judge_not_triggered",
            backend="empty",
            page_count=1,
            blocks=[],
        ),
        mineru_label=ParsedLabel(image_type="seal", caption="上海日轲电子有限公司"),
        qwen_label=None,
        mineru_output=ModelOutput(
            image_id="img-seal-select-aux",
            model_name="mineru",
            success=True,
            raw_text="{}",
        ),
        qwen_output=ModelOutput(
            image_id="img-seal-select-aux",
            model_name="qwen-judge",
            success=True,
            raw_text='{"selected_candidate":"paddle"}',
        ),
        seal_selection=SealSelectionDecision(
            selected_candidate="paddle",
            reason="paddle text is more complete",
            confidence="high",
        ),
        seal_selected_role="paddle",
        seal_selected_document=paddle_document,
        seal_selected_label=paddle_label,
        seal_selected_output=ModelOutput(
            image_id="img-seal-select-aux",
            model_name="paddle-local",
            success=True,
            raw_text="{}",
            vendor="paddleocr",
            source_type="local_service",
        ),
    )

    assert artifact.final_document.source == "paddle"
    assert artifact.final_document.raw_metadata["selected_output_role"] == "paddle"
    assert artifact.final_document.raw_metadata["selected_by"] == "seal_candidate_selection"
    assert artifact.final_label is not None
    assert artifact.final_label.image_type == "seal"
    assert artifact.review_required is False
    assert artifact.seal_selection is not None
    assert artifact.seal_selection.selected_candidate == "paddle"


def test_adjudicator_marks_review_when_seal_selection_requests_review() -> None:
    image_task = ImageTask(
        image_id="img-seal-select-review",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    mineru_document = CanonicalDocument(
        document_id="img-seal-select-review",
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
                bbox=[0, 0, 999, 999],
                text="上海日轲电子有限公司",
                content={"img_path": "data/demo.png", "image_caption": ["上海日轲电子有限公司"]},
                source="mineru",
            )
        ],
    )

    artifact = adjudicate_documents(
        image_task=image_task,
        mineru_document=mineru_document,
        qwen_document=CanonicalDocument(
            document_id="img-seal-select-review",
            source="qwen_judge_not_triggered",
            backend="empty",
            page_count=1,
            blocks=[],
        ),
        mineru_label=ParsedLabel(image_type="seal", caption="上海日轲电子有限公司"),
        qwen_label=None,
        mineru_output=ModelOutput(
            image_id="img-seal-select-review",
            model_name="mineru",
            success=True,
            raw_text="{}",
        ),
        qwen_output=ModelOutput(
            image_id="img-seal-select-review",
            model_name="qwen-judge",
            success=True,
            raw_text='{"selected_candidate":"review"}',
        ),
        seal_selection=SealSelectionDecision(
            selected_candidate="review",
            reason="all candidates are unreliable",
            confidence="low",
        ),
    )

    assert artifact.consensus is not None
    assert artifact.consensus.decision == "review"
    assert artifact.review_required is True
    assert artifact.final_document.source == "adjudicated"


def test_adjudicator_reviews_chart_table_second_pass_when_patch_is_not_adopted() -> None:
    image_task = ImageTask(
        image_id="img-chart-table-review",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    mineru_document = CanonicalDocument(
        document_id="img-chart-table-review",
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
                text="图表转表格",
                content={
                    "table_body": "| Sessions | Value |\n| --- | --- |\n| 1 | 62 |",
                    "table_caption": ["图表转表格"],
                    "img_path": "data/demo.png",
                },
                source="mineru",
                structured_label=StructuredLabel(
                    kind="table",
                    content="| Sessions | Value |\n| --- | --- |\n| 1 | 62 |",
                    format="markdown",
                    source="mineru",
                ),
                caption_structured=CaptionStructured(
                    brief="图表转表格", visual_type="table"
                ),
            )
        ],
    )

    artifact = adjudicate_documents(
        image_task=image_task,
        mineru_document=mineru_document,
        qwen_document=CanonicalDocument(
            document_id="img-chart-table-review",
            source="qwen_judge_not_triggered",
            backend="empty",
            page_count=1,
            blocks=[],
        ),
        mineru_label=ParsedLabel(image_type="table", caption="图表转表格"),
        qwen_label=None,
        mineru_output=ModelOutput(
            image_id="img-chart-table-review",
            model_name="mineru",
            success=True,
            raw_text="{}",
        ),
        qwen_output=ModelOutput(
            image_id="img-chart-table-review",
            model_name="qwen-judge",
            success=True,
            raw_text="not-json",
        ),
        issues=[
            Issue(
                issue_id="table-m1",
                issue_type="table_conflict",
                page_idx=0,
                target_block_id="m1",
                candidate_payload={
                    "review_mode": "chart_table_second_pass",
                    "must_output_final_table": True,
                },
                reasons=["chart_table_requires_qwen_second_pass"],
            )
        ],
        patch_decisions=[
            PatchDecision(
                issue_id="table-m1",
                target_block_id="m1",
                decision="keep_mineru",
                patch={},
                reason="llm_patch_invalid_json",
            )
        ],
    )

    assert artifact.consensus is not None
    assert artifact.consensus.decision == "review"
    assert artifact.review_required is True
    assert (
        "chart table second-stage adjudication did not produce an adoptable final table"
        in artifact.consensus.reasons
    )
    assert (
        artifact.final_document.raw_metadata.get("selected_by", "")
        != "chart_table_second_pass_adjudication"
    )


def _table_bundle(
    role: str,
    markdown_table: str,
    image_type: str = "table",
) -> dict[str, object]:
    block_type = "table" if image_type == "table" else "chart"
    block_sub_type = None
    content = (
        {"img_path": "data/demo.png", "table_body": markdown_table, "table_caption": ["表格"]}
        if block_type == "table"
        else {"img_path": "data/demo.png", "content": markdown_table, "chart_caption": ["图表"]}
    )
    return {
        "role": role,
        "output": ModelOutput(
            image_id=f"{role}-table",
            model_name=f"{role}-local",
            success=True,
            raw_text="{}",
        ),
        "document": CanonicalDocument(
            document_id=f"{role}-table",
            source=role,
            blocks=[
                CanonicalBlock(
                    block_id=f"{role}_b1",
                    page_idx=0,
                    order_index=1,
                    type=block_type,
                    sub_type=block_sub_type,
                    bbox=[0, 0, 1000, 1000],
                    text="表格",
                    content=content,
                    source=role,
                    structured_label=StructuredLabel(
                        kind="table",
                        content=markdown_table,
                        format="markdown",
                        source="model",
                    ),
                    caption_structured=CaptionStructured(brief="表格"),
                )
            ],
        ),
        "label": ParsedLabel(
            image_type=image_type,
            caption="表格",
            caption_structured=CaptionStructured(brief="表格", visual_type=image_type),
            structured_label=StructuredLabel(
                kind="table",
                content=markdown_table,
                format="markdown",
                source="model",
            ),
        ),
    }


def test_analyze_table_bundles_detects_all_candidate_consensus() -> None:
    markdown_table = "| 指标 | 值 |\n| --- | --- |\n| 增长率 | 12% |"

    analysis = analyze_table_bundles(
        mineru_bundle=_table_bundle("mineru", markdown_table),
        auxiliary_bundles=[
            _table_bundle("paddle", markdown_table),
            _table_bundle(
                "glm",
                markdown_table,
            ),
        ],
    )

    assert analysis is not None
    assert analysis["stable_consensus"] is True
    assert analysis["consensus_kind"] == "all"
    assert analysis["requires_qwen"] is False
    assert analysis["reference_role"] in {"paddle", "glm"}


def test_analyze_table_bundles_detects_mineru_pair_consensus_cluster() -> None:
    stable_markdown = "| 指标 | 值 |\n| --- | --- |\n| 增长率 | 12% |"
    divergent_markdown = "| 地区 | Q1 | Q2 |\n| --- | --- | --- |\n| 华东 | 10 | 20 |"

    analysis = analyze_table_bundles(
        mineru_bundle=_table_bundle("mineru", stable_markdown),
        auxiliary_bundles=[
            _table_bundle("paddle", stable_markdown),
            _table_bundle("glm", divergent_markdown),
        ],
    )

    assert analysis is not None
    assert analysis["stable_consensus"] is True
    assert analysis["consensus_kind"] == "pair"
    assert set(analysis["consensus_cluster"]) == {"mineru", "paddle"}
    assert analysis["reference_role"] == "paddle"
    assert analysis["requires_qwen"] is False
