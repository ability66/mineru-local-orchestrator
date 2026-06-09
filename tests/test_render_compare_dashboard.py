from __future__ import annotations

import json

from src.render_compare_dashboard import (
    _build_panel_from_final_payload,
    generate_compare_dashboard,
)


def test_generate_compare_dashboard_builds_dropdown_page(tmp_path) -> None:
    output_dir = tmp_path / "outputs"
    (output_dir / "normalized" / "mineru").mkdir(parents=True)
    (output_dir / "normalized" / "paddle").mkdir(parents=True)
    (output_dir / "normalized" / "glm").mkdir(parents=True)
    (output_dir / "normalized" / "qwen").mkdir(parents=True)
    (output_dir / "final").mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "demo.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    normalized_payload = {
        "document": {
            "blocks": [
                {
                    "type": "image",
                    "sub_type": "flowchart",
                    "content": {
                        "content": "flowchart TD\nA-->B",
                        "img_path": str(data_dir / "demo.png"),
                    },
                }
            ]
        },
        "derived_label": {
            "image_type": "flowchart",
            "caption": "flowchart TD\nA-->B",
            "structured_label": {
                "kind": "mermaid",
                "content": "flowchart TD\nA-->B",
                "format": "mermaid",
                "source": "model",
            },
        },
    }
    artifact_payload = {
        "final_document": normalized_payload["document"],
        "final_label": normalized_payload["derived_label"],
        "issues": [],
        "patch_decisions": [],
    }

    (output_dir / "normalized" / "mineru" / "demo.json").write_text(
        json.dumps(normalized_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "normalized" / "paddle" / "demo.json").write_text(
        json.dumps(normalized_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "normalized" / "glm" / "demo.json").write_text(
        json.dumps(normalized_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "normalized" / "qwen" / "demo.json").write_text(
        json.dumps(normalized_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "final" / "demo_artifact.json").write_text(
        json.dumps(artifact_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    html_path = generate_compare_dashboard(
        output_dir=output_dir,
        dashboard_dir=output_dir / "compare_dashboard",
    )

    assert html_path is not None
    html = html_path.read_text(encoding="utf-8")
    assert 'select id="type-select"' in html
    assert 'select id="image-select"' in html
    assert ">全部类型<" in html
    assert ">流程图<" in html
    assert ">demo<" in html
    assert 'data-record-type="flowchart"' in html
    assert 'event.key === "ArrowUp"' in html
    assert 'event.key === "ArrowDown"' in html
    assert 'event.key === "ArrowLeft"' in html
    assert 'event.key === "ArrowRight"' in html
    assert 'moveSelection(typeSelect' not in html
    assert 'moveSelection(select, -1)' in html
    assert 'moveSelection(select, 1)' in html
    assert "Original" in html
    assert "MinerU" in html
    assert "Paddle" in html
    assert "GLM" in html
    assert "Qwen" in html
    assert "Final" in html
    assert "normalized/paddle/demo.json" in html
    assert "normalized/glm/demo.json" in html


def test_generate_compare_dashboard_uses_final_payload_subtype_for_seal_records(tmp_path) -> None:
    output_dir = tmp_path / "outputs"
    (output_dir / "normalized" / "mineru").mkdir(parents=True)
    (output_dir / "normalized" / "qwen").mkdir(parents=True)
    (output_dir / "final").mkdir(parents=True)

    final_payload = {
        "parsed": {
            "filename": "seal-demo.png",
            "extraction_results": [
                {
                    "page": 0,
                    "json_res": [
                        {
                            "type": "image",
                            "sub_type": "seal",
                            "content": "上海木田电器电机有限公司",
                        }
                    ],
                }
            ],
        }
    }

    (output_dir / "final" / "seal-demo.json").write_text(
        json.dumps(final_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    html_path = generate_compare_dashboard(
        output_dir=output_dir,
        dashboard_dir=output_dir / "compare_dashboard",
    )

    assert html_path is not None
    html = html_path.read_text(encoding="utf-8")
    assert ">印章<" in html
    assert 'data-record-type="seal"' in html


def test_generate_compare_dashboard_does_not_render_seal_text_as_mermaid(tmp_path) -> None:
    output_dir = tmp_path / "outputs"
    (output_dir / "normalized" / "mineru").mkdir(parents=True)

    data_dir = tmp_path / "data" / "stamp"
    data_dir.mkdir(parents=True)
    (data_dir / "circle_Aug09869.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    normalized_payload = {
        "document": {
            "blocks": [
                {
                    "type": "image",
                    "sub_type": "seal",
                    "bbox": [0, 0, 999, 999],
                    "text": "上海日轲电子有限公司",
                    "content": {
                        "image_caption": ["上海日轲电子有限公司"],
                        "img_path": str(data_dir / "circle_Aug09869.png"),
                    },
                    "ocr_regions": [
                        {"role": "seal", "text": "上海日轲电子有限公司", "confidence": "medium"}
                    ],
                },
                {
                    "type": "paragraph",
                    "bbox": [194, 330, 543, 384],
                    "text": "4541982082",
                    "content": {
                        "paragraph_content": [{"type": "text", "content": "4541982082"}]
                    },
                },
                {
                    "type": "image",
                    "sub_type": "natural_image",
                    "bbox": [361, 390, 641, 644],
                    "text": "Red hammer and sickle symbol on white background (no text or numbers)",
                    "content": {
                        "image_caption": [
                            "Red hammer and sickle symbol on white background (no text or numbers)"
                        ],
                        "img_path": str(data_dir / "circle_Aug09869.png"),
                    },
                },
            ]
        },
        "derived_label": {
            "image_type": "seal",
            "caption": "上海日轲电子有限公司",
            "structured_label": {
                "kind": "none",
                "content": "",
                "format": "none",
                "source": "none",
            },
            "flowchart_graph": None,
            "ocr_regions": [
                {"role": "seal", "text": "上海日轲电子有限公司", "confidence": "medium"}
            ],
        },
    }

    (output_dir / "normalized" / "mineru" / "circle_Aug09869.json").write_text(
        json.dumps(normalized_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    html_path = generate_compare_dashboard(
        output_dir=output_dir,
        dashboard_dir=output_dir / "compare_dashboard",
    )

    assert html_path is not None
    html = html_path.read_text(encoding="utf-8")
    assert ">印章<" in html
    assert "4541982082" in html
    assert 'data-mermaid-b64="' not in html


def test_generate_compare_dashboard_shows_full_single_block_text_view(tmp_path) -> None:
    output_dir = tmp_path / "outputs"
    (output_dir / "normalized" / "mineru").mkdir(parents=True)

    data_dir = tmp_path / "data" / "stamp"
    data_dir.mkdir(parents=True)
    (data_dir / "circle_Aug09869.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    merged_text = (
        "上海日轲电子有限公司\n\n4541982082\n\n"
        "Red hammer and sickle symbol on white background (no text or numbers)"
    )
    normalized_payload = {
        "document": {
            "document_id": "circle_Aug09869",
            "source": "mineru",
            "backend": "mineru",
            "page_count": 1,
            "blocks": [
                {
                    "block_id": "b1",
                    "page_idx": 0,
                    "order_index": 1,
                    "type": "image",
                    "sub_type": "seal",
                    "bbox": [0, 0, 999, 999],
                    "text": merged_text,
                    "content": {
                        "image_caption": [merged_text],
                        "img_path": str(data_dir / "circle_Aug09869.png"),
                    },
                }
            ],
            "warnings": [],
            "raw_metadata": {
                "normalized_view": "single_block_projection",
                "source_block_count": 3,
            },
        },
        "derived_label": {
            "image_type": "seal",
            "caption": "上海日轲电子有限公司",
            "structured_label": {
                "kind": "none",
                "content": "",
                "format": "none",
                "source": "none",
            },
        },
    }

    (output_dir / "normalized" / "mineru" / "circle_Aug09869.json").write_text(
        json.dumps(normalized_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    html_path = generate_compare_dashboard(
        output_dir=output_dir,
        dashboard_dir=output_dir / "compare_dashboard",
    )

    assert html_path is not None
    html = html_path.read_text(encoding="utf-8")
    assert ">印章<" in html
    assert "上海日轲电子有限公司" in html
    assert "4541982082" in html
    assert "Red hammer and sickle symbol on white background (no text or numbers)" in html


def test_final_panel_prefers_final_label_semantics_over_final_document_blocks() -> None:
    artifact_payload = {
        "final_document": {
            "blocks": [
                {
                    "type": "image",
                    "sub_type": "flowchart",
                    "text": "flowchart TD\nA-->B",
                    "content": {"content": "flowchart TD\nA-->B"},
                    "structured_label": {
                        "kind": "mermaid",
                        "content": "flowchart TD\nA-->B",
                    },
                }
            ]
        },
        "final_label": {
            "image_type": "seal",
            "caption": "上海日轲电子有限公司",
            "structured_label": {
                "kind": "none",
                "content": "",
                "format": "none",
                "source": "none",
            },
        },
    }

    panel = _build_panel_from_final_payload(
        final_payload=None,
        artifact_payload=artifact_payload,
        snapshot_lookup={},
        title="Final",
        source_path="outputs/final/demo_artifact.json",
    )

    assert panel.image_type == "seal"
    assert panel.caption == "上海日轲电子有限公司"
    assert panel.render_kind == "text"
    assert panel.render_text == "flowchart TD\nA-->B"


def test_generate_compare_dashboard_prefers_final_label_type_for_record_type(tmp_path) -> None:
    output_dir = tmp_path / "outputs"
    (output_dir / "final").mkdir(parents=True)

    artifact_payload = {
        "final_document": {
            "blocks": [
                {
                    "type": "image",
                    "sub_type": "flowchart",
                    "text": "flowchart TD\nA-->B",
                    "content": {"content": "flowchart TD\nA-->B"},
                }
            ]
        },
        "final_label": {
            "image_type": "seal",
            "caption": "上海日轲电子有限公司",
            "structured_label": {
                "kind": "none",
                "content": "",
                "format": "none",
                "source": "none",
            },
        },
        "issues": [],
        "patch_decisions": [],
    }

    (output_dir / "final" / "demo_artifact.json").write_text(
        json.dumps(artifact_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    html_path = generate_compare_dashboard(
        output_dir=output_dir,
        dashboard_dir=output_dir / "compare_dashboard",
    )

    assert html_path is not None
    html = html_path.read_text(encoding="utf-8")
    assert ">印章<" in html
    assert 'data-record-type="seal"' in html


def test_generate_compare_dashboard_renders_table_markdown_as_html_table(
    tmp_path,
) -> None:
    output_dir = tmp_path / "outputs"
    (output_dir / "normalized" / "mineru").mkdir(parents=True)

    table_markdown = (
        "| 检查项 | 结果 |\n"
        "| --- | --- |\n"
        "| CA19-9 | 升高 |\n"
        "| CT | 可见病灶 |"
    )
    normalized_payload = {
        "document": {
            "blocks": [
                {
                    "type": "table",
                    "sub_type": "table",
                    "content": {"table_body": table_markdown},
                    "structured_label": {
                        "kind": "table",
                        "content": table_markdown,
                        "format": "markdown",
                        "source": "model",
                    },
                }
            ]
        },
        "derived_label": {
            "image_type": "table",
            "caption": "检验结果",
            "structured_label": {
                "kind": "table",
                "content": table_markdown,
                "format": "markdown",
                "source": "model",
            },
        },
    }

    (output_dir / "normalized" / "mineru" / "table-demo.json").write_text(
        json.dumps(normalized_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    html_path = generate_compare_dashboard(
        output_dir=output_dir,
        dashboard_dir=output_dir / "compare_dashboard",
    )

    assert html_path is not None
    html = html_path.read_text(encoding="utf-8")
    assert ">表格<" in html
    assert "Rendered Markdown" in html
    assert '<table class="markdown-table">' in html
    assert "<th>检查项</th>" in html
    assert "<th>结果</th>" in html
    assert "<td>CA19-9</td>" in html
    assert "<td>升高</td>" in html


def test_generate_compare_dashboard_renders_table_markdown_when_label_kind_is_text(
    tmp_path,
) -> None:
    output_dir = tmp_path / "outputs"
    (output_dir / "normalized" / "mineru").mkdir(parents=True)

    table_markdown = (
        "| 项目 | 数值 |\n"
        "| --- | --- |\n"
        "| A | 12 |\n"
        "| B | 20 |"
    )
    normalized_payload = {
        "document": {
            "blocks": [
                {
                    "type": "chart",
                    "sub_type": "table_like",
                    "text": table_markdown,
                    "structured_label": {
                        "kind": "text",
                        "content": table_markdown,
                        "format": "markdown",
                        "source": "model",
                    },
                }
            ]
        },
        "derived_label": {
            "image_type": "table",
            "caption": "统计表",
            "structured_label": {
                "kind": "text",
                "content": table_markdown,
                "format": "markdown",
                "source": "model",
            },
        },
    }

    (output_dir / "normalized" / "mineru" / "table-kind-text.json").write_text(
        json.dumps(normalized_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    html_path = generate_compare_dashboard(
        output_dir=output_dir,
        dashboard_dir=output_dir / "compare_dashboard",
    )

    assert html_path is not None
    html = html_path.read_text(encoding="utf-8")
    assert "Rendered Markdown" in html
    assert '<table class="markdown-table">' in html
    assert "<th>项目</th>" in html
    assert "<td>20</td>" in html


def test_generate_compare_dashboard_shows_qwen_adjudication_for_non_flowchart(
    tmp_path,
) -> None:
    output_dir = tmp_path / "outputs"
    (output_dir / "normalized" / "mineru").mkdir(parents=True)
    (output_dir / "normalized" / "qwen").mkdir(parents=True)
    (output_dir / "final").mkdir(parents=True)

    mineru_payload = {
        "document": {
            "blocks": [
                {
                    "type": "image",
                    "sub_type": "seal",
                    "text": "上海木田电器电机有限公司",
                }
            ]
        },
        "derived_label": {"image_type": "seal", "caption": "印章"},
    }
    qwen_payload = {
        "document": {
            "blocks": [
                {
                    "type": "image",
                    "sub_type": "seal",
                    "text": "QWEN原始文本不应显示",
                }
            ]
        },
        "derived_label": {"image_type": "seal", "caption": "Qwen 印章"},
    }
    artifact_payload = {
        "final_document": mineru_payload["document"],
        "final_label": {
            "image_type": "seal",
            "caption": "印章",
            "structured_label": {
                "kind": "text",
                "content": "上海木田电器电机有限公司",
                "format": "plain_text",
                "source": "model",
            },
        },
        "consensus": {
            "decision": "accepted",
            "reasons": ["seal issues resolved by second-stage adjudication"],
        },
        "reasons": ["seal issues resolved by second-stage adjudication"],
        "seal_selection": {
            "selected_candidate": "paddle",
            "reason": "paddle text is more complete",
            "confidence": "high",
        },
        "issues": [],
        "patch_decisions": [],
    }

    (output_dir / "normalized" / "mineru" / "seal-demo.json").write_text(
        json.dumps(mineru_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "normalized" / "qwen" / "seal-demo.json").write_text(
        json.dumps(qwen_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "final" / "seal-demo_artifact.json").write_text(
        json.dumps(artifact_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    html_path = generate_compare_dashboard(
        output_dir=output_dir,
        dashboard_dir=output_dir / "compare_dashboard",
    )

    assert html_path is not None
    html = html_path.read_text(encoding="utf-8")
    assert ">Qwen<" in html
    assert "final/seal-demo_artifact.json" in html
    assert "裁决结果：accepted" in html
    assert "裁决原因：" in html
    assert "seal issues resolved by second-stage adjudication" in html
    assert "印章候选：paddle" in html
    assert "选择原因：paddle text is more complete" in html
    assert "QWEN原始文本不应显示" not in html


def test_generate_compare_dashboard_hides_empty_qwen_adjudication_panel_for_non_flowchart(
    tmp_path,
) -> None:
    output_dir = tmp_path / "outputs"
    (output_dir / "normalized" / "mineru").mkdir(parents=True)
    (output_dir / "normalized" / "qwen").mkdir(parents=True)
    (output_dir / "final").mkdir(parents=True)

    normalized_payload = {
        "document": {
            "blocks": [
                {
                    "type": "table",
                    "sub_type": "table",
                    "content": {"table_body": "| A | B |\n| --- | --- |\n| 1 | 2 |"},
                }
            ]
        },
        "derived_label": {"image_type": "table", "caption": "表格"},
    }
    artifact_payload = {
        "final_document": normalized_payload["document"],
        "final_label": {
            "image_type": "table",
            "caption": "表格",
            "structured_label": {
                "kind": "table",
                "content": "| A | B |\n| --- | --- |\n| 1 | 2 |",
                "format": "markdown",
                "source": "model",
            },
        },
        "issues": [],
        "patch_decisions": [],
    }

    (output_dir / "normalized" / "mineru" / "table-hide-qwen.json").write_text(
        json.dumps(normalized_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "normalized" / "qwen" / "table-hide-qwen.json").write_text(
        json.dumps(normalized_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "final" / "table-hide-qwen_artifact.json").write_text(
        json.dumps(artifact_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    html_path = generate_compare_dashboard(
        output_dir=output_dir,
        dashboard_dir=output_dir / "compare_dashboard",
    )

    assert html_path is not None
    html = html_path.read_text(encoding="utf-8")
    assert ">表格<" in html
    assert ">Qwen<" not in html
