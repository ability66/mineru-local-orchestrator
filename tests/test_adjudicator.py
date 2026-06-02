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

