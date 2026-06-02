from __future__ import annotations

from src.pipeline.adjudicator import adjudicate_documents
from src.schema import (
    CanonicalBlock,
    CanonicalDocument,
    CaptionStructured,
    ImageTask,
    ModelOutput,
    OcrRegion,
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
                content={"img_path": "data/demo.png", "image_caption": ["某某公司印章"]},
                source="qwen",
                ocr_regions=[OcrRegion(role="seal", text="某某公司", confidence="high")],
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
        structured_label=StructuredLabel(kind="none", content="", format="none", source="none"),
        ocr_regions=[OcrRegion(role="seal", text="某某公司", confidence="high")],
    )

    artifact = adjudicate_documents(
        image_task=image_task,
        mineru_document=mineru_document,
        qwen_document=qwen_document,
        mineru_label=None,
        qwen_label=qwen_label,
        mineru_output=ModelOutput(image_id="img-1", model_name="mineru", success=True, raw_text="{}"),
        qwen_output=ModelOutput(image_id="img-1", model_name="qwen", success=True, raw_text="{}"),
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
                content={"table_body": "|a|b|", "table_caption": ["表格标题"], "img_path": "data/demo.png"},
                source="mineru",
                structured_label=StructuredLabel(
                    kind="table",
                    content="|a|b|",
                    format="markdown",
                    source="mineru",
                ),
                caption_structured=CaptionStructured(brief="表格标题", visual_type="table"),
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
                content={"content": "趋势说明", "chart_caption": ["图表标题"], "img_path": "data/demo.png"},
                source="qwen",
                caption_structured=CaptionStructured(brief="图表标题", visual_type="chart"),
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
        structured_label=StructuredLabel(kind="table", content="|a|b|", format="markdown", source="mineru"),
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
        structured_label=StructuredLabel(kind="text", content="趋势说明", format="plain_text", source="model"),
    )

    artifact = adjudicate_documents(
        image_task=image_task,
        mineru_document=mineru_document,
        qwen_document=qwen_document,
        mineru_label=mineru_label,
        qwen_label=qwen_label,
        mineru_output=ModelOutput(image_id="img-2", model_name="mineru", success=True, raw_text="{}"),
        qwen_output=ModelOutput(image_id="img-2", model_name="qwen", success=True, raw_text="{}"),
    )

    assert artifact.review_required is True
    assert artifact.final_document.blocks[0].type == "table"
    assert artifact.final_document.blocks[0].structured_label.kind == "table"
