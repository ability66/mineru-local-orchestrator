from __future__ import annotations

from src.schema import CanonicalBlock, CanonicalDocument, CaptionStructured, StructuredLabel
from src.writer import build_content_list, build_content_list_v2


def test_writer_preserves_flowchart_and_seal_blocks() -> None:
    document = CanonicalDocument(
        document_id="img-1",
        source="adjudicated",
        backend="mineru_plus_qwen",
        page_count=1,
        blocks=[
            CanonicalBlock(
                block_id="b1",
                page_idx=0,
                order_index=1,
                type="chart",
                sub_type="flowchart",
                bbox=[10, 10, 900, 900],
                text="流程图",
                content={"img_path": "/tmp/demo.png", "content": "flowchart TD\nA-->B"},
                source="adjudicated",
                structured_label=StructuredLabel(
                    kind="mermaid",
                    content="flowchart TD\nA-->B",
                    format="mermaid",
                    source="fused_graph",
                ),
                caption_structured=CaptionStructured(brief="流程图"),
            ),
            CanonicalBlock(
                block_id="b2",
                page_idx=0,
                order_index=2,
                type="image",
                sub_type="seal",
                bbox=[100, 100, 200, 200],
                text="某某公司",
                content={"img_path": "/tmp/demo.png", "image_caption": ["某某公司印章"]},
                source="adjudicated",
                caption_structured=CaptionStructured(brief="某某公司印章"),
            ),
        ],
    )

    content_list_v2 = build_content_list_v2(document)
    content_list = build_content_list(document)

    assert content_list_v2[0][0]["type"] == "chart"
    assert content_list_v2[0][0]["sub_type"] == "flowchart"
    assert content_list_v2[0][0]["content"]["content"] == "flowchart TD\nA-->B"
    assert content_list_v2[0][1]["type"] == "image"
    assert content_list_v2[0][1]["sub_type"] == "seal"

    assert content_list[0]["type"] == "chart"
    assert content_list[0]["sub_type"] == "flowchart"
    assert content_list[1]["type"] == "image"
    assert content_list[1]["sub_type"] == "seal"

