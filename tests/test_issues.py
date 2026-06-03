from __future__ import annotations

from src.pipeline.issues import detect_seal_issues
from src.schema import CanonicalBlock, CanonicalDocument, CaptionStructured, ImageTask, OcrRegion


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
