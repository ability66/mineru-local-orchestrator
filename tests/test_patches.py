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


def test_apply_patch_decisions_supports_use_qwen_fields_for_flowchart_reference() -> None:
    mineru_document = CanonicalDocument(
        document_id="img-flow-1",
        source="mineru",
        blocks=[
            CanonicalBlock(
                block_id="m1",
                page_idx=0,
                order_index=1,
                type="chart",
                bbox=[0, 0, 1000, 1000],
                text="流程图",
                content={"img_path": "data/demo.png", "chart_caption": ["流程图"]},
                source="mineru",
                caption_structured=CaptionStructured(brief="流程图"),
            )
        ],
    )
    issues = [
        Issue(
            issue_id="flowchart-review-m1",
            issue_type="flowchart_graph_conflict",
            page_idx=0,
            target_block_id="m1",
            candidate_payload={
                "reference_patch": {
                    "type": "chart",
                    "sub_type": "flowchart",
                    "content": {"content": "flowchart TD\nA-->B"},
                    "flowchart_graph": {
                        "node_order_rule": "fused_visual_order",
                        "nodes": [
                            {"node_id": "N001", "order_index": 1, "shape": "rectangle", "text": "A"},
                            {"node_id": "N002", "order_index": 2, "shape": "rectangle", "text": "B"},
                        ],
                        "edges": [{"source": "N001", "target": "N002", "label": ""}],
                    },
                }
            },
            reasons=["flowchart_graph_conflict_detected"],
        )
    ]
    patch_decisions = [
        PatchDecision(
            issue_id="flowchart-review-m1",
            target_block_id="m1",
            decision="use_qwen_fields",
            patch={},
            reason="参考 Mermaid 可直接采用",
        )
    ]

    patched = apply_patch_decisions(
        mineru_document=mineru_document,
        issues=issues,
        patch_decisions=patch_decisions,
    )

    block = patched.blocks[0]
    assert block.sub_type == "flowchart"
    assert block.content["content"] == "flowchart TD\nA-->B"
    assert block.structured_label.kind == "mermaid"
    assert block.flowchart_graph is not None


def test_apply_patch_decisions_preserves_markdown_table_semantics_for_chart_block() -> None:
    markdown_table = "| 指标 | 值 |\n| --- | --- |\n| 增长率 | 12% |"
    mineru_document = CanonicalDocument(
        document_id="img-table-1",
        source="mineru",
        blocks=[
            CanonicalBlock(
                block_id="m1",
                page_idx=0,
                order_index=1,
                type="chart",
                bbox=[0, 0, 1000, 1000],
                text="图表标题",
                content={"img_path": "data/demo.png", "chart_caption": ["图表标题"]},
                source="mineru",
                caption_structured=CaptionStructured(brief="图表标题"),
            )
        ],
    )
    issues = [
        Issue(
            issue_id="table-m1",
            issue_type="table_conflict",
            page_idx=0,
            target_block_id="m1",
            candidate_payload={
                "reference_patch": {
                    "type": "chart",
                    "content": {
                        "content": markdown_table,
                        "img_path": "data/demo.png",
                        "chart_caption": ["图表标题"],
                    },
                }
            },
            reasons=["table_candidates_diverge"],
        )
    ]
    patch_decisions = [
        PatchDecision(
            issue_id="table-m1",
            target_block_id="m1",
            decision="use_qwen_fields",
            patch={},
            reason="采用参考 Markdown 表格",
        )
    ]

    patched = apply_patch_decisions(
        mineru_document=mineru_document,
        issues=issues,
        patch_decisions=patch_decisions,
    )

    block = patched.blocks[0]
    assert block.type == "table"
    assert block.sub_type is None
    assert block.content["table_body"] == markdown_table
    assert block.structured_label.kind == "table"
    assert block.structured_label.format == "markdown"
