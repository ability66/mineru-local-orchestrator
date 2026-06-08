from __future__ import annotations

from src.main import (
    _build_seal_adjudication_candidates,
    _pick_seal_reference_bundle,
    build_stage2_selection_record,
)
from src.schema import (
    CanonicalBlock,
    CanonicalDocument,
    CaptionStructured,
    ImageTask,
    ModelOutput,
    OcrRegion,
    ParsedLabel,
    SealSelectionDecision,
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


def test_build_seal_adjudication_candidates_returns_full_text_candidates() -> None:
    image_task = ImageTask(
        image_id="seal-select-1",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    mineru_document = CanonicalDocument(
        document_id="seal-select-1",
        source="mineru",
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
                ocr_regions=[OcrRegion(role="seal", text="上海日轲电子有限公司")],
            ),
            CanonicalBlock(
                block_id="m2",
                page_idx=0,
                order_index=2,
                type="paragraph",
                bbox=[194, 330, 543, 384],
                text="4541982082",
                content={"paragraph_content": [{"type": "text", "content": "4541982082"}]},
                source="mineru",
            ),
        ],
    )
    paddle_document = CanonicalDocument(
        document_id="seal-select-1",
        source="paddle",
        blocks=[
            CanonicalBlock(
                block_id="p1",
                page_idx=0,
                order_index=1,
                type="image",
                sub_type="seal",
                bbox=[0, 0, 999, 999],
                text="上海日轲电子有限公司",
                content={"img_path": "data/demo.png", "image_caption": ["上海日轲电子有限公司"]},
                source="paddle",
                caption_structured=CaptionStructured(brief="上海日轲电子有限公司"),
                ocr_regions=[OcrRegion(role="seal", text="上海日轲电子有限公司")],
            )
        ],
    )

    candidates, payload = _build_seal_adjudication_candidates(
        image_task=image_task,
        mineru_bundle={
            "output": ModelOutput(
                image_id="seal-select-1",
                model_name="mineru-local",
                success=True,
                raw_text="{}",
            ),
            "document": mineru_document,
            "label": ParsedLabel(image_type="seal", caption="上海日轲电子有限公司"),
        },
        auxiliary_bundles=[
            {
                "role": "paddle",
                "output": ModelOutput(
                    image_id="seal-select-1",
                    model_name="paddle-local",
                    success=True,
                    raw_text="{}",
                ),
                "document": paddle_document,
                "label": ParsedLabel(image_type="seal", caption="上海日轲电子有限公司"),
            }
        ],
    )

    assert len(candidates) == 2
    assert payload is not None
    mineru_payload = next(
        candidate for candidate in payload["candidates"] if candidate["candidate_id"] == "mineru"
    )
    assert mineru_payload["full_text"] == "上海日轲电子有限公司\n\n4541982082"
    assert payload["comparisons"][0]["candidate_id"] == "paddle"
    assert payload["comparisons"][0]["issue_types"] == ["seal_ocr_conflict"]


def test_build_stage2_selection_record_keeps_selection_payload() -> None:
    record = build_stage2_selection_record(
        selection_payload={
            "image_id": "seal-select-2",
            "candidates": [{"candidate_id": "mineru"}, {"candidate_id": "paddle"}],
        },
        output=ModelOutput(
            image_id="seal-select-2",
            model_name="qwen-judge",
            success=True,
            raw_text='{"selected_candidate":"paddle"}',
            parsed={"usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}},
        ),
        selection_decision=SealSelectionDecision(
            selected_candidate="paddle",
            reason="paddle text is more complete",
            confidence="high",
        ),
        prompt="prompt body",
        mode="seal_adjudication",
    )

    assert record["mode"] == "seal_adjudication"
    assert record["selection_payload"]["candidates"][1]["candidate_id"] == "paddle"
    assert record["selection_decision"]["selected_candidate"] == "paddle"
