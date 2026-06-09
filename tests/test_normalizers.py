from __future__ import annotations

import json

from src.pipeline.normalizers import (
    normalize_mineru_payload,
    normalize_paddle_payload,
    normalize_qwen_payload,
)
from src.pipeline.table_utils import is_table_like
from src.schema import ImageTask, ModelOutput


def test_normalize_mineru_payload_unwraps_nested_data_container() -> None:
    image_task = ImageTask(
        image_id="img-1",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    model_output = ModelOutput(
        image_id="img-1",
        model_name="mineru",
        success=True,
        raw_text="",
        parsed={
            "code": 0,
            "data": {
                "content_list_v2": [[
                    {
                        "type": "image",
                        "sub_type": "seal",
                        "bbox": [0, 0, 1000, 1000],
                        "content": {
                            "image_caption": ["某某公司印章"],
                            "img_path": "data/demo.png",
                        },
                    }
                ]]
            },
        },
    )

    _, document, label = normalize_mineru_payload(
        image_task=image_task,
        model_output=model_output,
    )

    assert not document.warnings
    assert len(document.blocks) == 1
    assert document.blocks[0].type == "image"
    assert document.blocks[0].sub_type == "seal"
    assert document.blocks[0].text == "某某公司印章"
    assert document.blocks[0].caption_structured.visual_type == "seal"
    assert label is not None
    assert label.image_type == "seal"
    assert label.caption_structured.visual_type == "seal"
    assert any(region.role == "seal" for region in label.ocr_regions)


def test_normalize_mineru_payload_normalizes_ocr_role_and_confidence() -> None:
    image_task = ImageTask(
        image_id="img-ocr",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    model_output = ModelOutput(
        image_id="img-ocr",
        model_name="mineru",
        success=True,
        raw_text="",
        parsed={
            "content_list_v2": [[
                {
                    "type": "image",
                    "sub_type": "seal",
                    "bbox": [0, 0, 1000, 1000],
                    "content": {"img_path": "data/demo.png"},
                    "ocr_regions": [
                        {"role": "stamp", "text": "某某公司", "confidence": 0.92}
                    ],
                }
            ]]
        },
    )

    _, document, label = normalize_mineru_payload(
        image_task=image_task,
        model_output=model_output,
    )

    region = document.blocks[0].ocr_regions[0]
    assert region.role == "seal"
    assert region.confidence == "high"
    assert label is not None
    assert label.image_type == "seal"
    assert label.caption_structured.visual_type == "seal"
    assert label.ocr_regions[0].role == "seal"


def test_normalize_mineru_payload_unwraps_extraction_result_json_res() -> None:
    image_task = ImageTask(
        image_id="img-extract",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    model_output = ModelOutput(
        image_id="img-extract",
        model_name="mineru",
        success=True,
        raw_text="",
        parsed={
            "success": True,
            "extraction_result": {
                "page": 0,
                "filename": "demo.png",
                "md_res": "# demo",
                "json_res": {
                    "content_list_v2": [[
                        {
                            "type": "table",
                            "bbox": [0, 0, 1000, 1000],
                            "content": {
                                "table_body": "|a|b|",
                                "table_caption": ["表格标题"],
                                "img_path": "data/demo.png",
                            },
                        }
                    ]]
                },
            },
        },
    )

    _, document, label = normalize_mineru_payload(
        image_task=image_task,
        model_output=model_output,
    )

    assert not document.warnings
    assert len(document.blocks) == 1
    assert document.blocks[0].type == "table"
    assert label is not None
    assert label.image_type == "table"


def test_normalize_mineru_payload_converts_one_based_pages_to_zero_based() -> None:
    image_task = ImageTask(
        image_id="img-one-based",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    model_output = ModelOutput(
        image_id="img-one-based",
        model_name="mineru",
        success=True,
        raw_text="",
        parsed={
            "parsed": {
                "filename": "demo.png",
                "total_pages": 1,
                "extraction_results": [
                    {
                        "page": 1,
                        "file_name": "demo.png",
                        "md_res": "page1",
                        "json_res": [
                            {
                                "type": "text",
                                "bbox": [0, 100, 1000, 200],
                                "angle": 0,
                                "content": "第一页正文",
                            }
                        ],
                    }
                ],
            }
        },
    )

    _, document, label = normalize_mineru_payload(
        image_task=image_task,
        model_output=model_output,
    )

    assert not document.warnings
    assert document.page_count == 1
    assert len(document.blocks) == 1
    assert document.blocks[0].page_idx == 0
    assert document.blocks[0].text == "第一页正文"
    assert label is not None
    assert label.caption == "第一页正文"


def test_normalize_mineru_payload_unwraps_tmp_shape_extraction_results_with_flat_json_res() -> None:
    image_task = ImageTask(
        image_id="img-extract-list",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    model_output = ModelOutput(
        image_id="img-extract-list",
        model_name="mineru",
        success=True,
        raw_text="",
        parsed={
            "parsed": {
                "filename": "demo.png",
                "total_pages": 2,
                "extraction_results": [
                    {
                        "page": 0,
                        "file_name": "demo.png",
                        "md_res": "page0",
                        "json_res": [
                            {
                                "type": "title",
                                "bbox": [0, 0, 1000, 100],
                                "angle": 0,
                                "content": "第一页标题",
                            }
                        ],
                    },
                    {
                        "page": 1,
                        "file_name": "demo.png",
                        "md_res": "page1",
                        "json_res": [
                            {
                                "type": "text",
                                "bbox": [0, 100, 1000, 200],
                                "angle": 0,
                                "content": "第二页正文",
                            },
                            {
                                "type": "image",
                                "bbox": [0, 200, 400, 500],
                                "angle": 90,
                                "content": "某某公司印章",
                                "sub_type": "seal",
                            },
                        ],
                    },
                ],
            }
        },
    )

    _, document, label = normalize_mineru_payload(
        image_task=image_task,
        model_output=model_output,
    )

    assert not document.warnings
    assert document.page_count == 2
    assert len(document.blocks) == 3
    assert document.blocks[0].type == "title"
    assert document.blocks[1].type == "paragraph"
    assert document.blocks[0].text == "第一页标题"
    assert document.blocks[0].content["title_content"][0]["content"] == "第一页标题"
    assert document.blocks[1].text == "第二页正文"
    assert document.blocks[1].content["paragraph_content"][0]["content"] == "第二页正文"
    assert document.blocks[2].type == "image"
    assert document.blocks[2].sub_type == "seal"
    assert document.blocks[2].text == "某某公司印章"
    assert document.blocks[2].content["image_caption"] == ["某某公司印章"]
    assert document.blocks[2].ocr_regions[0].role == "seal"
    assert document.blocks[2].ocr_regions[0].text == "某某公司印章"
    assert document.blocks[2].provenance["format"] == "json_res_flat"
    assert document.blocks[2].provenance["source_angle"] == 90.0
    assert label is not None
    assert label.caption == "第一页标题"


def test_normalize_paddle_payload_extracts_seal_blocks_from_parsing_res_list() -> None:
    image_task = ImageTask(
        image_id="circle_Aug09866",
        image_path="data/stamp/circle_Aug09866.png",
        file_name="circle_Aug09866.png",
        file_ext=".png",
    )
    model_output = ModelOutput(
        image_id="circle_Aug09866",
        model_name="paddle-local",
        success=True,
        raw_text="",
        parsed={
            "filename": "circle_Aug09866.png",
            "total_pages": 1,
            "extraction_results": [
                {
                    "page": 1,
                    "filename": "circle_Aug09866.png",
                    "md_res": "seal",
                    "json_res": {
                        "width": 640,
                        "height": 640,
                        "parsing_res_list": [
                            {
                                "block_label": "seal",
                                "block_content": "上海木田电器电机有限公司",
                                "block_bbox": [1, 0, 640, 636],
                                "block_id": 0,
                                "block_order": 1,
                            }
                        ],
                        "layout_det_res": {
                            "boxes": [
                                {
                                    "label": "seal",
                                    "score": 0.9626336097717285,
                                    "coordinate": [1, 0, 640, 636],
                                    "order": 1,
                                }
                            ]
                        },
                    },
                }
            ],
        },
    )

    _, document, label = normalize_paddle_payload(
        image_task=image_task,
        model_output=model_output,
    )

    assert not document.warnings
    assert document.page_count == 1
    assert len(document.blocks) == 1
    block = document.blocks[0]
    assert block.page_idx == 0
    assert block.type == "image"
    assert block.sub_type == "seal"
    assert block.text == "上海木田电器电机有限公司"
    assert block.content["image_caption"] == ["上海木田电器电机有限公司"]
    assert block.confidence is not None and block.confidence > 0.9
    assert block.caption_structured.visual_type == "seal"
    assert block.ocr_regions[0].role == "seal"
    assert block.ocr_regions[0].text == "上海木田电器电机有限公司"
    assert block.ocr_regions[0].confidence == "high"
    assert label is not None
    assert label.image_type == "seal"
    assert label.caption == "上海木田电器电机有限公司"
    assert label.caption_structured.visual_type == "seal"
    assert label.ocr_regions[0].role == "seal"


def test_normalize_qwen_payload_prefers_mineru_style_document_as_source_of_truth() -> None:
    image_task = ImageTask(
        image_id="img-qwen",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    payload = {
        "image_type": "chart",
        "caption": "图表标题",
        "structured_label": {
            "kind": "text",
            "content": "趋势说明",
            "format": "plain_text",
        },
        "content_list_v2": [[
            {
                "type": "table",
                "bbox": [0, 0, 1000, 1000],
                "content": {
                    "table_body": "|a|b|",
                    "table_caption": ["表格标题"],
                    "img_path": "data/demo.png",
                },
            }
        ]],
    }
    model_output = ModelOutput(
        image_id="img-qwen",
        model_name="qwen",
        success=True,
        raw_text=json.dumps(payload, ensure_ascii=False),
    )

    _, document, label = normalize_qwen_payload(
        image_task=image_task,
        model_output=model_output,
    )

    assert document.blocks[0].type == "table"
    assert label is not None
    assert label.image_type == "table"
    assert label.structured_label.kind == "table"


def test_normalize_qwen_payload_uses_top_level_summary_only_as_mineru_style_patch() -> None:
    image_task = ImageTask(
        image_id="img-flow",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    payload = {
        "image_type": "flowchart",
        "caption": "流程图标题",
        "structured_label": {
            "kind": "mermaid",
            "content": "flowchart TD\nA-->B",
            "format": "mermaid",
        },
        "flowchart_graph": {
            "node_order_rule": "top_to_bottom_left_to_right",
            "nodes": [
                {"node_id": "N001", "order_index": 1, "shape": "rectangle", "text": "A"},
                {"node_id": "N002", "order_index": 2, "shape": "rectangle", "text": "B"},
            ],
            "edges": [{"source": "N001", "target": "N002", "label": ""}],
        },
        "content_list_v2": [[
            {
                "type": "chart",
                "bbox": [0, 0, 1000, 1000],
                "content": {
                    "chart_caption": ["流程图标题"],
                    "img_path": "data/demo.png",
                },
            }
        ]],
    }
    model_output = ModelOutput(
        image_id="img-flow",
        model_name="qwen",
        success=True,
        raw_text=json.dumps(payload, ensure_ascii=False),
    )

    _, document, label = normalize_qwen_payload(
        image_task=image_task,
        model_output=model_output,
    )

    block = document.blocks[0]
    assert block.type == "chart"
    assert block.sub_type == "flowchart"
    assert block.content["content"] == "flowchart TD\nA-->B"
    assert block.flowchart_graph is not None
    assert label is not None
    assert label.image_type == "flowchart"


def test_normalize_qwen_payload_accepts_flat_string_content_blocks() -> None:
    image_task = ImageTask(
        image_id="img-qwen-flat",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    payload = {
        "content_list_v2": [[
            {
                "type": "title",
                "bbox": [0, 0, 1000, 100],
                "content": "示例标题",
            },
            {
                "type": "chart",
                "sub_type": "flowchart",
                "bbox": [0, 100, 1000, 1000],
                "content": "flowchart TD\nA-->B",
            },
        ]],
    }
    model_output = ModelOutput(
        image_id="img-qwen-flat",
        model_name="qwen",
        success=True,
        raw_text=json.dumps(payload, ensure_ascii=False),
    )

    _, document, label = normalize_qwen_payload(
        image_task=image_task,
        model_output=model_output,
    )

    assert document.blocks[0].type == "title"
    assert document.blocks[0].text == "示例标题"
    assert document.blocks[0].content["title_content"][0]["content"] == "示例标题"
    assert document.blocks[1].type == "chart"
    assert document.blocks[1].sub_type == "flowchart"
    assert document.blocks[1].content["content"] == "flowchart TD\nA-->B"
    assert document.blocks[1].structured_label.kind == "mermaid"
    assert label is not None
    assert label.image_type == "flowchart"


def test_normalize_qwen_payload_preserves_seal_as_formal_image_type() -> None:
    image_task = ImageTask(
        image_id="img-qwen-seal",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    payload = {
        "image_type": "seal",
        "caption": "某某公司印章",
        "caption_structured": {
            "brief": "某某公司印章",
            "visual_type": "印章",
            "main_subject": "某某公司印章",
        },
        "ocr_regions": [
            {"role": "stamp", "text": "某某公司", "confidence": "high"}
        ],
    }
    model_output = ModelOutput(
        image_id="img-qwen-seal",
        model_name="qwen",
        success=True,
        raw_text=json.dumps(payload, ensure_ascii=False),
    )

    _, document, label = normalize_qwen_payload(
        image_task=image_task,
        model_output=model_output,
    )

    assert document.blocks[0].type == "image"
    assert document.blocks[0].sub_type == "seal"
    assert document.blocks[0].caption_structured.visual_type == "seal"
    assert label is not None
    assert label.image_type == "seal"
    assert label.caption_structured.visual_type == "seal"


def test_normalize_qwen_payload_does_not_treat_plain_text_flowchart_content_as_mermaid() -> None:
    image_task = ImageTask(
        image_id="img-qwen-flow-text",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    payload = {
        "content_list_v2": [[
            {
                "type": "chart",
                "sub_type": "flowchart",
                "bbox": [0, 100, 1000, 1000],
                "content": "这是一个审批流程图，从申请到审批结束",
            },
        ]],
    }
    model_output = ModelOutput(
        image_id="img-qwen-flow-text",
        model_name="qwen",
        success=True,
        raw_text=json.dumps(payload, ensure_ascii=False),
    )

    _, document, label = normalize_qwen_payload(
        image_task=image_task,
        model_output=model_output,
    )

    block = document.blocks[0]
    assert block.sub_type == "flowchart"
    assert "content" not in block.content
    assert block.content["chart_caption"] == ["这是一个审批流程图，从申请到审批结束"]
    assert block.structured_label.kind == "text"
    assert label is not None
    assert label.image_type == "flowchart"


def test_normalize_mineru_payload_converts_chart_markdown_table_to_table_block() -> None:
    image_task = ImageTask(
        image_id="img-chart-markdown-table",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    markdown_table = "| 指标 | 值 |\n| --- | --- |\n| 增长率 | 12% |"
    model_output = ModelOutput(
        image_id="img-chart-markdown-table",
        model_name="mineru",
        success=True,
        raw_text="",
        parsed={
            "content_list_v2": [[
                {
                    "type": "chart",
                    "bbox": [0, 0, 1000, 1000],
                    "content": {
                        "content": markdown_table,
                        "chart_caption": ["图表标题"],
                        "img_path": "data/demo.png",
                    },
                }
            ]]
        },
    )

    _, document, label = normalize_mineru_payload(
        image_task=image_task,
        model_output=model_output,
    )

    block = document.blocks[0]
    assert block.type == "table"
    assert block.sub_type is None
    assert block.structured_label.kind == "table"
    assert block.structured_label.format == "markdown"
    assert block.content["table_body"] == markdown_table
    assert label is not None
    assert label.image_type == "table"


def test_table_detector_does_not_capture_flowchart_branch() -> None:
    image_task = ImageTask(
        image_id="img-flow-no-table",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    model_output = ModelOutput(
        image_id="img-flow-no-table",
        model_name="mineru",
        success=True,
        raw_text="",
        parsed={
            "content_list_v2": [[
                {
                    "type": "chart",
                    "sub_type": "flowchart",
                    "bbox": [0, 0, 1000, 1000],
                    "content": {
                        "content": "flowchart TD\nA-->B",
                        "img_path": "data/demo.png",
                    },
                }
            ]]
        },
    )

    _, document, label = normalize_mineru_payload(
        image_task=image_task,
        model_output=model_output,
    )

    assert document.blocks[0].sub_type == "flowchart"
    assert is_table_like(document) is False
    assert label is not None
    assert is_table_like(label) is False
