from __future__ import annotations

import json

from src.schema import (
    AdjudicationArtifact,
    CanonicalBlock,
    CanonicalDocument,
    CaptionStructured,
    ImageTask,
    ModelOutput,
    OcrRegion,
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
                content={
                    "img_path": "/tmp/demo.png",
                    "image_caption": ["某某公司印章"],
                },
                source="adjudicated",
                caption_structured=CaptionStructured(brief="某某公司印章"),
                ocr_regions=[
                    OcrRegion(role="seal", text="某某公司", confidence="high")
                ],
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


def test_writer_outputs_tmp_style_final_payload_and_removes_legacy_files(
    tmp_path,
) -> None:
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
                provenance={"source_block_type": "image", "source_angle": 15},
            ),
            CanonicalBlock(
                block_id="b2",
                page_idx=0,
                order_index=2,
                type="image",
                sub_type="seal",
                bbox=[100, 100, 200, 200],
                text="某某公司",
                content={
                    "img_path": "/tmp/demo.png",
                    "image_caption": ["某某公司印章"],
                },
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
        stage2_records=[
            {
                "mode": "flowchart_adjudication",
                "issue_id": "flow-1",
                "issue_type": "flowchart_graph_conflict",
                "prompt": "prompt body",
                "issue_payload": {"issue_id": "flow-1"},
                "success": True,
                "raw_text": '{"decision":"merge"}',
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 80,
                    "total_tokens": 200,
                },
                "patch_decision": {"decision": "merge"},
            }
        ],
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
    assert "ocr_regions" not in blocks[1]
    assert not (final_dir / "img-1_content_list_v2.json").exists()
    assert not (final_dir / "img-1_content_list.json").exists()
    assert (final_dir / "img-1_artifact.json").exists()
    stage2_payload = json.loads(
        (tmp_path / "judge_stage2" / "img-1.json").read_text(encoding="utf-8")
    )
    assert stage2_payload["record_count"] == 1
    assert stage2_payload["totals"]["total_tokens"] == 200
    assert summary["final_block_count"] == 2


def test_writer_uses_selected_qwen_metadata_for_final_payload(tmp_path) -> None:
    image_task = ImageTask(
        image_id="img-qwen-final",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    final_document = CanonicalDocument(
        document_id="img-qwen-final",
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
                bbox=[10, 10, 900, 900],
                text="流程图",
                content={"img_path": "/tmp/demo.png", "content": "flowchart TD\nA-->B\nB-->C"},
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
        raw_metadata={
            "selected_output_role": "qwen",
            "selected_model_name": "glm-4v-local",
            "selected_vendor": "glm",
            "selected_source_type": "local_service",
        },
    )
    artifact = AdjudicationArtifact(
        image_id="img-qwen-final",
        final_document=final_document,
    )
    mineru_output = ModelOutput(
        image_id="img-qwen-final",
        model_name="mineru",
        success=True,
        raw_text="",
        parsed={},
        latency_ms=120,
        vendor="mineru",
        source_type="local_api",
    )
    qwen_output = ModelOutput(
        image_id="img-qwen-final",
        model_name="glm-4v-local",
        success=True,
        raw_text="",
        parsed={},
        latency_ms=80,
        vendor="glm",
        source_type="local_service",
    )

    write_image_result(
        output_dir=tmp_path,
        image_task=image_task,
        mineru_output=mineru_output,
        qwen_output=qwen_output,
        mineru_document=final_document,
        qwen_document=final_document,
        mineru_label=None,
        qwen_label=None,
        artifact=artifact,
        stage2_records=None,
    )

    final_payload = json.loads(
        (tmp_path / "final" / "img-qwen-final.json").read_text(encoding="utf-8")
    )
    assert final_payload["model_name"] == "glm-4v-local"
    assert final_payload["vendor"] == "glm"
    assert final_payload["source_type"] == "local_service"
