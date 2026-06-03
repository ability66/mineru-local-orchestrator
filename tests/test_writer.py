from __future__ import annotations

import json

from src.schema import (
    AdjudicationArtifact,
    CanonicalBlock,
    CanonicalDocument,
    CaptionStructured,
    ImageTask,
    ModelOutput,
    StructuredLabel,
)
from src.writer import build_content_list, build_content_list_v2, write_image_result


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


def test_writer_outputs_tmp_style_final_payload_and_removes_legacy_files(tmp_path) -> None:
    image_task = ImageTask(
        image_id="img-1",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    final_document = CanonicalDocument(
        document_id="img-1",
        source="mineru",
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
                provenance={"source_block_type": "chart", "source_angle": 15},
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
    artifact = AdjudicationArtifact(
        image_id="img-1",
        final_document=final_document,
    )
    mineru_output = ModelOutput(
        image_id="img-1",
        model_name="mineru",
        success=True,
        raw_text="",
        parsed={},
        latency_ms=120,
        vendor="mineru",
        source_type="local_api",
    )
    qwen_output = ModelOutput(
        image_id="img-1",
        model_name="qwen",
        success=True,
        raw_text="",
        parsed={},
        latency_ms=80,
    )

    final_dir = tmp_path / "final"
    final_dir.mkdir(parents=True)
    (final_dir / "img-1_content_list_v2.json").write_text("legacy", encoding="utf-8")
    (final_dir / "img-1_content_list.json").write_text("legacy", encoding="utf-8")

    summary = write_image_result(
        output_dir=tmp_path,
        image_task=image_task,
        mineru_output=mineru_output,
        qwen_output=qwen_output,
        mineru_document=final_document,
        qwen_document=final_document,
        mineru_label=None,
        qwen_label=None,
        artifact=artifact,
    )

    final_payload = json.loads((final_dir / "img-1.json").read_text(encoding="utf-8"))
    assert final_payload["image_id"] == "img-1"
    assert final_payload["model_name"] == "mineru"
    assert final_payload["success"] is True
    assert final_payload["vendor"] == "mineru"
    assert final_payload["source_type"] == "local_api"
    assert final_payload["latency_ms"] == 200
    assert final_payload["parsed"]["filename"] == "demo.png"
    assert final_payload["parsed"]["total_pages"] == 1
    assert final_payload["parsed"]["extraction_results"][0]["file_name"] == "demo.png"
    assert final_payload["parsed"]["extraction_results"][0]["md_res"] == ""
    blocks = final_payload["parsed"]["extraction_results"][0]["json_res"]
    assert blocks[0]["type"] == "chart"
    assert blocks[0]["sub_type"] == "flowchart"
    assert blocks[0]["content"] == "flowchart TD\nA-->B"
    assert blocks[0]["angle"] == 15
    assert blocks[1]["type"] == "image"
    assert blocks[1]["sub_type"] == "seal"
    assert blocks[1]["content"] == "某某公司印章"
    assert not (final_dir / "img-1_content_list_v2.json").exists()
    assert not (final_dir / "img-1_content_list.json").exists()
    assert (final_dir / "img-1_artifact.json").exists()
    assert summary["final_block_count"] == 2
