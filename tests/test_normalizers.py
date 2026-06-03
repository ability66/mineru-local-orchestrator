from __future__ import annotations

import json

from src.pipeline.normalizers import normalize_mineru_payload, normalize_qwen_payload
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
    assert label is not None
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


def test_normalize_mineru_payload_unwraps_multi_page_extraction_result_list() -> None:
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
            "extraction_result": [
                {
                    "page": 0,
                    "filename": "demo.png",
                    "md_res": "page0",
                    "json_res": [
                        {
                            "type": "title",
                            "bbox": [0, 0, 1000, 100],
                            "content": {
                                "title_content": [{"type": "text", "content": "第一页标题"}],
                                "level": 1,
                            },
                        }
                    ],
                },
                {
                    "page": 1,
                    "filename": "demo.png",
                    "md_res": "page1",
                    "json_res": [
                        {
                            "type": "paragraph",
                            "bbox": [0, 100, 1000, 200],
                            "content": {
                                "paragraph_content": [{"type": "text", "content": "第二页正文"}],
                            },
                        }
                    ],
                },
            ]
        },
    )

    _, document, label = normalize_mineru_payload(
        image_task=image_task,
        model_output=model_output,
    )

    assert not document.warnings
    assert document.page_count == 2
    assert len(document.blocks) == 2
    assert document.blocks[0].type == "title"
    assert document.blocks[1].type == "paragraph"
    assert label is not None
    assert label.caption == "第一页标题"


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
