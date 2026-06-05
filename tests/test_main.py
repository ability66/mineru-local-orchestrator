from __future__ import annotations

from src.main import _pick_seal_reference_bundle
from src.schema import (
    CanonicalBlock,
    CanonicalDocument,
    CaptionStructured,
    ImageTask,
    ModelOutput,
    OcrRegion,
    ParsedLabel,
)


def test_pick_seal_reference_bundle_prefers_richer_auxiliary_result() -> None:
    image_task = ImageTask(
        image_id="seal-1",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    mineru_document = CanonicalDocument(
        document_id="seal-1",
        source="mineru",
        blocks=[
            CanonicalBlock(
                block_id="m1",
                page_idx=0,
                order_index=1,
                type="image",
                bbox=[0, 0, 100, 100],
                text="",
                content={"img_path": "data/demo.png"},
                source="mineru",
                caption_structured=CaptionStructured(brief=""),
            )
        ],
    )
    paddle_document = CanonicalDocument(
        document_id="seal-1",
        source="paddle",
        blocks=[
            CanonicalBlock(
                block_id="p1",
                page_idx=0,
                order_index=1,
                type="image",
                sub_type="seal",
                bbox=[0, 0, 100, 100],
                text="某某公司印章",
                content={
                    "img_path": "data/demo.png",
                    "image_caption": ["某某公司印章"],
                },
                source="paddle",
                caption_structured=CaptionStructured(brief="某某公司印章"),
                ocr_regions=[
                    OcrRegion(role="seal", text="某某公司", confidence="high")
                ],
            )
        ],
    )
    glm_document = CanonicalDocument(
        document_id="seal-1",
        source="glm",
        blocks=[
            CanonicalBlock(
                block_id="g1",
                page_idx=0,
                order_index=1,
                type="image",
                sub_type="seal",
                bbox=[0, 0, 100, 100],
                text="",
                content={"img_path": "data/demo.png"},
                source="glm",
                caption_structured=CaptionStructured(brief=""),
            )
        ],
    )

    bundle, issues = _pick_seal_reference_bundle(
        image_task=image_task,
        mineru_document=mineru_document,
        auxiliary_bundles=[
            {
                "role": "glm",
                "output": ModelOutput(
                    image_id="seal-1",
                    model_name="glm-local",
                    success=True,
                    raw_text="{}",
                ),
                "document": glm_document,
                "label": ParsedLabel(image_type="seal"),
            },
            {
                "role": "paddle",
                "output": ModelOutput(
                    image_id="seal-1",
                    model_name="paddle-local",
                    success=True,
                    raw_text="{}",
                ),
                "document": paddle_document,
                "label": ParsedLabel(image_type="seal"),
            },
        ],
    )

    assert bundle is not None
    assert bundle["role"] == "paddle"
    assert len(issues) == 1
    assert issues[0].issue_type == "seal_type_disagreement"
    assert issues[0].candidate_payload is not None
    assert issues[0].candidate_payload["reference_model_role"] == "paddle"
    assert issues[0].candidate_payload["reference_model_name"] == "paddle-local"
