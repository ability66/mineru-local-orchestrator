from __future__ import annotations

from src.pipeline.patches import apply_patch_decisions
from src.schema import CanonicalBlock, CanonicalDocument, CaptionStructured, Issue, PatchDecision


def test_apply_patch_decisions_merges_seal_fields_into_mineru_document() -> None:
    mineru_document = CanonicalDocument(
        document_id="img-1",
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
    issues = [
        Issue(
            issue_id="seal-type-m1",
            issue_type="seal_type_disagreement",
            page_idx=0,
            target_block_id="m1",
            reasons=["qwen_marks_block_as_seal"],
        )
    ]
    patch_decisions = [
        PatchDecision(
            issue_id="seal-type-m1",
            target_block_id="m1",
            decision="merge",
            patch={
                "type": "image",
                "sub_type": "seal",
                "text": "某某公司印章",
                "content": {"image_caption": ["某某公司印章"]},
                "ocr_regions": [
                    {"role": "seal", "text": "某某公司", "confidence": "high"}
                ],
            },
            reason="补印章 subtype 和 OCR",
        )
    ]

    patched = apply_patch_decisions(
        mineru_document=mineru_document,
        issues=issues,
        patch_decisions=patch_decisions,
    )

    block = patched.blocks[0]
    assert block.sub_type == "seal"
    assert block.text == "某某公司印章"
    assert block.content["image_caption"] == ["某某公司印章"]
    assert block.ocr_regions[0].role == "seal"
    assert block.ocr_regions[0].text == "某某公司"
    assert block.provenance["llm_patch_decision"] == "merge"
