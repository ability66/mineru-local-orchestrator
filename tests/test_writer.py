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
    ParsedLabel,
    StructuredLabel,
)
from src.writer import (
    build_content_list,
    build_content_list_v2,
    write_image_result,
    write_page_merged_markdown,
)


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
    expected_md = "\n\n".join(
        [
            "![Figure](<img src=/tmp/demo.png>)\n```mermaid\nflowchart TD\nA-->B\n```",
            "![Figure](<img src=/tmp/demo.png>)\n```seal\n某某公司\n```",
        ]
    )
    assert final_payload["parsed"]["extraction_results"][0]["md_res"] == expected_md
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
    assert (final_dir / "img-1.md").read_text(encoding="utf-8") == expected_md
    assert (final_dir / "img-1_artifact.json").exists()
    stage2_payload = json.loads(
        (tmp_path / "judge_stage2" / "img-1.json").read_text(encoding="utf-8")
    )
    assert stage2_payload["record_count"] == 1
    assert stage2_payload["totals"]["total_tokens"] == 200
    assert summary["final_block_count"] == 2


def test_writer_markdown_prefers_mineru_text_and_final_special_blocks(tmp_path) -> None:
    image_task = ImageTask(
        image_id="img-markdown-hybrid",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    mineru_document = CanonicalDocument(
        document_id="img-markdown-hybrid",
        source="mineru",
        backend="mineru",
        page_count=1,
        blocks=[
            CanonicalBlock(
                block_id="title-1",
                page_idx=0,
                order_index=1,
                type="title",
                bbox=[0, 0, 100, 50],
                text="Mineru 标题",
                text_level=2,
                content={"title_content": [{"type": "text", "content": "Mineru 标题"}]},
                source="mineru",
            ),
            CanonicalBlock(
                block_id="chart-1",
                page_idx=0,
                order_index=2,
                type="chart",
                bbox=[0, 60, 400, 300],
                text="old chart text",
                content={"img_path": "/tmp/chart.png", "content": "old chart text"},
                source="mineru",
                caption_structured=CaptionStructured(brief="旧图表"),
            ),
        ],
    )
    final_table = "| 指标 | 值 |\n| --- | --- |\n| 增长率 | 12% |"
    final_document = CanonicalDocument(
        document_id="img-markdown-hybrid",
        source="qwen",
        backend="qwen",
        page_count=1,
        blocks=[
            CanonicalBlock(
                block_id="title-1",
                page_idx=0,
                order_index=1,
                type="title",
                bbox=[0, 0, 100, 50],
                text="Qwen 标题",
                text_level=2,
                content={"title_content": [{"type": "text", "content": "Qwen 标题"}]},
                source="qwen",
            ),
            CanonicalBlock(
                block_id="chart-1",
                page_idx=0,
                order_index=2,
                type="table",
                bbox=[0, 60, 400, 300],
                text="Qwen 表格",
                content={
                    "img_path": "/tmp/chart.png",
                    "table_body": final_table,
                    "table_caption": ["Qwen 表格"],
                },
                source="qwen",
                structured_label=StructuredLabel(
                    kind="table",
                    content=final_table,
                    format="markdown",
                    source="model",
                ),
                caption_structured=CaptionStructured(brief="Qwen 表格"),
            ),
        ],
    )
    artifact = AdjudicationArtifact(
        image_id="img-markdown-hybrid",
        final_document=final_document,
    )

    write_image_result(
        output_dir=tmp_path,
        image_task=image_task,
        mineru_output=ModelOutput(
            image_id="img-markdown-hybrid",
            model_name="mineru",
            success=True,
            raw_text="",
            parsed={},
        ),
        qwen_output=ModelOutput(
            image_id="img-markdown-hybrid",
            model_name="qwen",
            success=True,
            raw_text="",
            parsed={},
        ),
        mineru_document=mineru_document,
        qwen_document=final_document,
        mineru_label=None,
        qwen_label=None,
        artifact=artifact,
        stage2_records=None,
    )

    final_payload = json.loads(
        (tmp_path / "final" / "img-markdown-hybrid.json").read_text(encoding="utf-8")
    )
    markdown_text = final_payload["parsed"]["extraction_results"][0]["md_res"]

    assert "## Mineru 标题" in markdown_text
    assert "Qwen 标题" not in markdown_text
    assert "```tablechart" in markdown_text
    assert final_table in markdown_text
    assert "old chart text" not in markdown_text
    assert (
        (tmp_path / "final" / "img-markdown-hybrid.md").read_text(encoding="utf-8")
        == markdown_text
    )


def test_writer_merges_page_crop_markdown_by_page_and_block_order(tmp_path) -> None:
    final_dir = tmp_path / "final"
    final_dir.mkdir(parents=True)
    (final_dir / "doc1_02_010_flowchart.md").write_text("third", encoding="utf-8")
    (final_dir / "doc1_02_002_table.md").write_text("first", encoding="utf-8")
    (final_dir / "doc1_02_003_text.md").write_text("second", encoding="utf-8")

    written_paths = write_page_merged_markdown(
        output_dir=tmp_path,
        image_tasks=[
            ImageTask(
                image_id="doc1_02_010_flowchart",
                image_path="data/doc1_02_010_flowchart.jpg",
                file_name="doc1_02_010_flowchart.jpg",
                file_ext=".jpg",
                page_output_id="doc1_02",
                merge_order="010",
                is_page_crop=True,
            ),
            ImageTask(
                image_id="doc1_02_002_table",
                image_path="data/doc1_02_002_table.jpg",
                file_name="doc1_02_002_table.jpg",
                file_ext=".jpg",
                page_output_id="doc1_02",
                merge_order="002",
                is_page_crop=True,
            ),
            ImageTask(
                image_id="doc1_02_003_text",
                image_path="data/doc1_02_003_text.jpg",
                file_name="doc1_02_003_text.jpg",
                file_ext=".jpg",
                page_output_id="doc1_02",
                merge_order="003",
                is_page_crop=True,
            ),
        ],
    )

    assert written_paths == [tmp_path / "final" / "doc1_02.md"]
    assert (tmp_path / "final" / "doc1_02.md").read_text(encoding="utf-8") == (
        "first\n\nsecond\n\nthird"
    )


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


def test_writer_uses_selected_auxiliary_metadata_for_final_payload(tmp_path) -> None:
    image_task = ImageTask(
        image_id="img-aux-final",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    final_document = CanonicalDocument(
        document_id="img-aux-final",
        source="paddle",
        backend="paddle",
        page_count=1,
        blocks=[
            CanonicalBlock(
                block_id="p1",
                page_idx=0,
                order_index=1,
                type="image",
                sub_type="seal",
                bbox=[10, 10, 900, 900],
                text="某某公司印章",
                content={"img_path": "/tmp/demo.png", "image_caption": ["某某公司印章"]},
                source="paddle",
                caption_structured=CaptionStructured(brief="某某公司印章"),
            )
        ],
        raw_metadata={
            "selected_output_role": "paddle",
            "selected_model_name": "paddle-local",
            "selected_vendor": "paddleocr",
            "selected_source_type": "local_service",
        },
    )
    artifact = AdjudicationArtifact(
        image_id="img-aux-final",
        final_document=final_document,
    )

    write_image_result(
        output_dir=tmp_path,
        image_task=image_task,
        mineru_output=ModelOutput(
            image_id="img-aux-final",
            model_name="mineru",
            success=True,
            raw_text="",
            parsed={},
            latency_ms=120,
            vendor="mineru",
            source_type="local_api",
        ),
        qwen_output=ModelOutput(
            image_id="img-aux-final",
            model_name="qwen-judge",
            success=True,
            raw_text="",
            parsed={},
            latency_ms=80,
            vendor="qwen",
            source_type="local_service",
        ),
        mineru_document=final_document,
        qwen_document=final_document,
        mineru_label=None,
        qwen_label=None,
        artifact=artifact,
        stage2_records=None,
    )

    final_payload = json.loads(
        (tmp_path / "final" / "img-aux-final.json").read_text(encoding="utf-8")
    )
    assert final_payload["model_name"] == "paddle-local"
    assert final_payload["vendor"] == "paddleocr"
    assert final_payload["source_type"] == "local_service"


def test_writer_persists_extra_stage1_outputs_for_glm_and_paddle(tmp_path) -> None:
    image_task = ImageTask(
        image_id="img-extra-stage1",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    base_document = CanonicalDocument(
        document_id="img-extra-stage1",
        source="mineru",
        backend="mineru",
        page_count=1,
        blocks=[],
    )
    artifact = AdjudicationArtifact(
        image_id="img-extra-stage1",
        final_document=base_document,
    )
    glm_document = CanonicalDocument(
        document_id="img-extra-stage1",
        source="glm",
        backend="glm",
        page_count=1,
        blocks=[],
    )
    paddle_document = CanonicalDocument(
        document_id="img-extra-stage1",
        source="paddle",
        backend="paddle",
        page_count=1,
        blocks=[],
    )

    write_image_result(
        output_dir=tmp_path,
        image_task=image_task,
        mineru_output=ModelOutput(
            image_id="img-extra-stage1",
            model_name="mineru",
            success=True,
            raw_text="",
            parsed={},
        ),
        qwen_output=None,
        mineru_document=base_document,
        qwen_document=base_document,
        mineru_label=None,
        qwen_label=None,
        artifact=artifact,
        stage2_records=None,
        extra_stage1_results={
            "glm": {
                "output": ModelOutput(
                    image_id="img-extra-stage1",
                    model_name="glm",
                    success=True,
                    raw_text='{"ok":true}',
                    parsed={"ok": True},
                ),
                "document": glm_document,
                "label": None,
            },
            "paddle": {
                "output": ModelOutput(
                    image_id="img-extra-stage1",
                    model_name="paddle",
                    success=True,
                    raw_text='{"ok":true}',
                    parsed={"ok": True},
                ),
                "document": paddle_document,
                "label": None,
            },
        },
    )

    assert (tmp_path / "raw" / "glm" / "img-extra-stage1.json").exists()
    assert (tmp_path / "raw" / "paddle" / "img-extra-stage1.json").exists()
    assert (tmp_path / "normalized" / "glm" / "img-extra-stage1.json").exists()
    assert (tmp_path / "normalized" / "paddle" / "img-extra-stage1.json").exists()


def test_writer_projects_single_block_normalized_view_for_non_structured_input(tmp_path) -> None:
    image_task = ImageTask(
        image_id="circle_Aug09869",
        image_path="data/stamp/circle_Aug09869.png",
        file_name="circle_Aug09869.png",
        file_ext=".png",
    )
    mineru_document = CanonicalDocument(
        document_id="circle_Aug09869",
        source="mineru",
        backend="mineru",
        page_count=1,
        blocks=[
            CanonicalBlock(
                block_id="b1",
                page_idx=0,
                order_index=1,
                type="image",
                sub_type="seal",
                bbox=[0, 0, 999, 999],
                text="上海日轲电子有限公司",
                content={
                    "img_path": "data/stamp/circle_Aug09869.png",
                    "image_caption": ["上海日轲电子有限公司"],
                },
                source="mineru",
                caption_structured=CaptionStructured(brief="上海日轲电子有限公司"),
                ocr_regions=[OcrRegion(role="seal", text="上海日轲电子有限公司")],
            ),
            CanonicalBlock(
                block_id="b2",
                page_idx=0,
                order_index=2,
                type="paragraph",
                bbox=[194, 330, 543, 384],
                text="4541982082",
                content={
                    "paragraph_content": [{"type": "text", "content": "4541982082"}]
                },
                source="mineru",
            ),
            CanonicalBlock(
                block_id="b3",
                page_idx=0,
                order_index=3,
                type="image",
                sub_type="natural_image",
                bbox=[361, 390, 641, 644],
                text="Red hammer and sickle symbol on white background (no text or numbers)",
                content={
                    "img_path": "data/stamp/circle_Aug09869.png",
                    "image_caption": [
                        "Red hammer and sickle symbol on white background (no text or numbers)"
                    ],
                },
                source="mineru",
            ),
        ],
    )
    mineru_label = ParsedLabel(
        image_type="seal",
        caption="上海日轲电子有限公司",
        caption_structured=CaptionStructured(
            brief="上海日轲电子有限公司",
            visual_type="seal",
        ),
        ocr_regions=[OcrRegion(role="seal", text="上海日轲电子有限公司")],
    )
    artifact = AdjudicationArtifact(
        image_id="circle_Aug09869",
        final_document=mineru_document,
    )

    write_image_result(
        output_dir=tmp_path,
        image_task=image_task,
        mineru_output=ModelOutput(
            image_id="circle_Aug09869",
            model_name="mineru",
            success=True,
            raw_text="",
            parsed={},
        ),
        qwen_output=None,
        mineru_document=mineru_document,
        qwen_document=mineru_document,
        mineru_label=mineru_label,
        qwen_label=mineru_label,
        artifact=artifact,
        stage2_records=None,
    )

    normalized_payload = json.loads(
        (tmp_path / "normalized" / "mineru" / "circle_Aug09869.json").read_text(
            encoding="utf-8"
        )
    )
    blocks = normalized_payload["document"]["blocks"]
    assert len(blocks) == 1
    assert blocks[0]["type"] == "image"
    assert blocks[0]["sub_type"] == "seal"
    assert blocks[0]["text"] == (
        "上海日轲电子有限公司\n\n4541982082\n\n"
        "Red hammer and sickle symbol on white background (no text or numbers)"
    )
    assert blocks[0]["content"]["image_caption"] == [
        "上海日轲电子有限公司\n\n4541982082\n\n"
        "Red hammer and sickle symbol on white background (no text or numbers)"
    ]
    assert normalized_payload["document"]["raw_metadata"]["normalized_view"] == "single_block_projection"
    assert normalized_payload["document"]["raw_metadata"]["source_block_count"] == 3
