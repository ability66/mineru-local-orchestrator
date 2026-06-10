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


def test_generate_compare_dashboard_renders_table_markdown(
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
    assert '<option value="chart">图表</option>' in html
    assert '<option value="table">表格</option>' not in html
    assert 'data-record-type="chart"' in html
    assert "Rendered Markdown" in html
    assert '<table class="markdown-table">' in html
    assert "<th>检查项</th>" in html
    assert "<th>结果</th>" in html
    assert "<td>CA19-9</td>" in html
    assert "<td>升高</td>" in html


def test_generate_compare_dashboard_renders_markdown_table_with_latex(
    tmp_path,
) -> None:
    output_dir = tmp_path / "outputs"
    (output_dir / "normalized" / "mineru").mkdir(parents=True)

    table_markdown = (
        "| 指标 | 表达式 |\n"
        "| --- | --- |\n"
        "| 面积 | $x^2 + y^2$ |\n"
        "| 积分 | \\(\\int_0^1 x^2 dx\\) |"
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
            "caption": "Markdown 表格",
            "structured_label": {
                "kind": "table",
                "content": table_markdown,
                "format": "markdown",
                "source": "model",
            },
        },
    }

    (output_dir / "normalized" / "mineru" / "table-latex-demo.json").write_text(
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
    assert "<th>指标</th>" in html
    assert "$x^2 + y^2$" in html
    assert "tex-chtml.js" in html


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
            "image_type": "chart",
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


def test_generate_compare_dashboard_shows_qwen_chart_second_pass_table_for_non_flowchart(
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
                    "type": "table",
                    "sub_type": "table",
                    "content": {
                        "table_body": "| 指标 | 值 |\n| --- | --- |\n| 增长率 | 12% |",
                        "table_caption": ["MinerU 表格"],
                    },
                }
            ]
        },
        "derived_label": {"image_type": "table", "caption": "MinerU 表格"},
    }
    qwen_payload = {
        "document": {
            "blocks": [
                {
                    "type": "image",
                    "sub_type": "chart",
                    "text": "Qwen 原始识别文本不应显示",
                }
            ]
        },
        "derived_label": {"image_type": "chart", "caption": "Qwen 图表"},
    }
    artifact_payload = {
        "final_document": {
            "blocks": [
                {
                    "type": "table",
                    "sub_type": "table",
                    "content": {
                        "table_body": "| 指标 | 值 |\n| --- | --- |\n| 增长率 | 12% |\n| 同比 | 8% |",
                        "table_caption": ["Qwen 终裁表格"],
                    },
                }
            ],
            "raw_metadata": {
                "table_analysis": {
                    "candidate_roles": ["mineru", "qwen", "glm", "paddle"],
                    "reference_role": "glm",
                }
            },
        },
        "final_label": {
            "image_type": "table",
            "caption": "Qwen 终裁表格",
            "structured_label": {
                "kind": "table",
                "content": "| 指标 | 值 |\n| --- | --- |\n| 增长率 | 12% |\n| 同比 | 8% |",
                "format": "markdown",
                "source": "model",
            },
        },
        "issues": [
            {
                "issue_id": "table-m1",
                "candidate_payload": {
                    "review_mode": "chart_table_second_pass",
                    "branch_mode": "chart_table",
                },
            }
        ],
        "patch_decisions": [
            {
                "issue_id": "table-m1",
                "decision": "merge",
                "reason": "chart table second-pass adjudication",
                "patch": {
                    "type": "table",
                    "content": {
                        "table_body": "| 指标 | 值 |\n| --- | --- |\n| 增长率 | 12% |\n| 同比 | 8% |",
                        "table_caption": ["Qwen 终裁表格"],
                    },
                },
            }
        ],
    }

    (output_dir / "normalized" / "mineru" / "chart-qwen-table.json").write_text(
        json.dumps(mineru_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "normalized" / "qwen" / "chart-qwen-table.json").write_text(
        json.dumps(qwen_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "final" / "chart-qwen-table_artifact.json").write_text(
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
    assert "final/chart-qwen-table_artifact.json" in html
    assert "展示二阶段 Qwen 终裁表格" in html
    assert "候选来源：mineru, qwen, glm, paddle" in html
    assert "参考候选：glm" in html
    assert "展示裁决结果与原因" not in html
    assert "Rendered Markdown" in html
    assert "<th>指标</th>" in html
    assert "<td>8%</td>" in html
    assert "Qwen 终裁表格" in html
    assert "Qwen 原始识别文本不应显示" not in html


def test_generate_compare_dashboard_renders_multiple_chart_tables_from_artifact(
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
                    "block_id": "m1",
                    "type": "chart",
                    "sub_type": "chart",
                    "content": {"chart_caption": ["(a) 5-way 10-shot"]},
                },
                {
                    "block_id": "m2",
                    "type": "chart",
                    "sub_type": "chart",
                    "content": {"chart_caption": ["(b) 5-way full-shot"]},
                },
            ]
        },
        "derived_label": {"image_type": "chart", "caption": "MinerU 图表"},
    }
    qwen_payload = {
        "document": {
            "blocks": [
                {
                    "block_id": "q1",
                    "type": "table",
                    "sub_type": "table",
                    "content": {
                        "table_body": "| Sessions | Left |\n| --- | --- |\n| 1 | 62 |\n| 2 | 57 |",
                        "table_caption": ["(a) 5-way 10-shot"],
                    },
                },
                {
                    "block_id": "q2",
                    "type": "table",
                    "sub_type": "table",
                    "content": {
                        "table_body": "| Sessions | Right |\n| --- | --- |\n| 1 | 61 |\n| 2 | 59 |",
                        "table_caption": ["(b) 5-way full-shot"],
                    },
                },
            ]
        },
        "derived_label": {"image_type": "chart", "caption": "Qwen 图表"},
    }
    artifact_payload = {
        "final_document": {
            "blocks": [
                {
                    "block_id": "m1",
                    "type": "table",
                    "sub_type": "table",
                    "content": {
                        "table_body": "| Sessions | Left |\n| --- | --- |\n| 1 | 62 |\n| 2 | 57 |",
                        "table_caption": ["(a) 5-way 10-shot"],
                    },
                },
                {
                    "block_id": "m2",
                    "type": "table",
                    "sub_type": "table",
                    "content": {
                        "table_body": "| Sessions | Right |\n| --- | --- |\n| 1 | 61 |\n| 2 | 59 |",
                        "table_caption": ["(b) 5-way full-shot"],
                    },
                },
            ]
        },
        "final_label": {
            "image_type": "table",
            "caption": "(a) 5-way 10-shot",
            "structured_label": {
                "kind": "table",
                "content": "| Sessions | Left |\n| --- | --- |\n| 1 | 62 |\n| 2 | 57 |",
                "format": "markdown",
                "source": "model",
            },
        },
        "issues": [
            {
                "issue_id": "table-m1",
                "candidate_payload": {
                    "review_mode": "chart_table_second_pass",
                    "branch_mode": "chart_table",
                },
            },
            {
                "issue_id": "table-m2",
                "candidate_payload": {
                    "review_mode": "chart_table_second_pass",
                    "branch_mode": "chart_table",
                },
            },
        ],
        "patch_decisions": [
            {
                "issue_id": "table-m1",
                "decision": "merge",
                "reason": "left chart reconstructed",
                "patch": {
                    "type": "table",
                    "content": {
                        "table_body": "| Sessions | Left |\n| --- | --- |\n| 1 | 62 |\n| 2 | 57 |",
                        "table_caption": ["(a) 5-way 10-shot"],
                    },
                },
            },
            {
                "issue_id": "table-m2",
                "decision": "merge",
                "reason": "right chart reconstructed",
                "patch": {
                    "type": "table",
                    "content": {
                        "table_body": "| Sessions | Right |\n| --- | --- |\n| 1 | 61 |\n| 2 | 59 |",
                        "table_caption": ["(b) 5-way full-shot"],
                    },
                },
            },
        ],
    }

    (output_dir / "normalized" / "mineru" / "multi-chart.json").write_text(
        json.dumps(mineru_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "normalized" / "qwen" / "multi-chart.json").write_text(
        json.dumps(qwen_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "final" / "multi-chart_artifact.json").write_text(
        json.dumps(artifact_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    html_path = generate_compare_dashboard(
        output_dir=output_dir,
        dashboard_dir=output_dir / "compare_dashboard",
    )

    assert html_path is not None
    html = html_path.read_text(encoding="utf-8")
    assert "展示二阶段 Qwen 终裁表格" in html
    assert "(a) 5-way 10-shot" in html
    assert "(b) 5-way full-shot" in html
    assert "<td>62</td>" in html
    assert "<td>59</td>" in html
    assert "| Sessions | Right |" in html


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
    assert '<option value="chart">图表</option>' in html
    assert '<option value="table">表格</option>' not in html
    assert ">Qwen<" not in html


def test_generate_compare_dashboard_reorders_flowchart_panels_and_hides_empty_provider_cards(
    tmp_path,
) -> None:
    output_dir = tmp_path / "outputs"
    (output_dir / "normalized" / "mineru").mkdir(parents=True)
    (output_dir / "normalized" / "paddle").mkdir(parents=True)
    (output_dir / "normalized" / "glm").mkdir(parents=True)
    (output_dir / "normalized" / "qwen").mkdir(parents=True)
    (output_dir / "final").mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "flowchart.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    mineru_payload = {
        "document": {
            "blocks": [
                {
                    "type": "image",
                    "sub_type": "flowchart",
                    "content": {
                        "content": 'flowchart TD\nA["开始"] --> B["结束"]',
                        "img_path": str(data_dir / "flowchart.png"),
                    },
                }
            ]
        },
        "derived_label": {
            "image_type": "flowchart",
            "caption": "MinerU 流程图",
            "structured_label": {
                "kind": "mermaid",
                "content": 'flowchart TD\nA["开始"] --> B["结束"]',
                "format": "mermaid",
                "source": "model",
            },
        },
    }
    qwen_payload = {
        "document": {
            "blocks": [
                {
                    "type": "chart",
                    "sub_type": "flowchart",
                    "content": {
                        "content": 'flowchart TD\nStart["开始"] --> End["结束"]',
                        "img_path": str(data_dir / "flowchart.png"),
                    },
                }
            ]
        },
        "derived_label": {
            "image_type": "flowchart",
            "caption": "Qwen 流程图",
            "structured_label": {
                "kind": "mermaid",
                "content": 'flowchart TD\nStart["开始"] --> End["结束"]',
                "format": "mermaid",
                "source": "model",
            },
        },
    }
    artifact_payload = {
        "final_document": qwen_payload["document"],
        "final_label": {
            "image_type": "flowchart",
            "caption": "Final 流程图",
            "structured_label": {
                "kind": "mermaid",
                "content": 'flowchart TD\nStart["开始"] --> End["结束"]',
                "format": "mermaid",
                "source": "model",
            },
        },
        "consensus": {
            "decision": "use_qwen_fields",
            "reasons": ["Qwen 在冲突节点的结构更准确"],
        },
        "patch_decisions": [
            {
                "issue_id": "flowchart-1",
                "decision": "use_qwen_fields",
                "reason": "参考侧在冲突点上更合理",
            }
        ],
        "issues": [
            {
                "issue_id": "flowchart-1",
                "candidate_payload": {
                    "ocr_reference_sources": [
                        {
                            "reference_model_role": "paddle",
                            "reference_model_name": "paddle-local",
                            "ocr_reference_texts": ["审批通过"],
                        },
                        {
                            "reference_model_role": "glm",
                            "reference_model_name": "glm-local",
                            "ocr_reference_texts": ["人工复核"],
                        },
                    ]
                },
            }
        ],
    }

    (output_dir / "normalized" / "mineru" / "flowchart-demo.json").write_text(
        json.dumps(mineru_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "normalized" / "qwen" / "flowchart-demo.json").write_text(
        json.dumps(qwen_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "final" / "flowchart-demo_artifact.json").write_text(
        json.dumps(artifact_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    html_path = generate_compare_dashboard(
        output_dir=output_dir,
        dashboard_dir=output_dir / "compare_dashboard",
    )

    assert html_path is not None
    html = html_path.read_text(encoding="utf-8")
    assert ">Paddle<" not in html
    assert ">GLM<" not in html
    assert ">Judge Reason<" in html
    assert "裁决结果：use_qwen_fields" in html
    assert "Qwen 在冲突节点的结构更准确" in html
    assert "流程图文字参考：" in html
    assert "paddle-local: 审批通过" in html
    assert "glm-local: 人工复核" in html
    assert html.index(">Qwen<") < html.index(">MinerU<")
    assert html.index(">MinerU<") < html.index(">Final<")
    assert html.index(">Final<") < html.index(">Judge Reason<")


def test_generate_compare_dashboard_moves_metadata_below_rendered_content(
    tmp_path,
) -> None:
    output_dir = tmp_path / "outputs"
    (output_dir / "normalized" / "mineru").mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    image_path = data_dir / "table.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    table_markdown = (
        "| 指标 | 表达式 |\n"
        "| --- | --- |\n"
        "| 面积 | $x^2 + y^2$ |"
    )
    normalized_payload = {
        "document": {
            "blocks": [
                {
                    "type": "table",
                    "sub_type": "table",
                    "content": {
                        "table_body": table_markdown,
                        "img_path": str(image_path),
                    },
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
            "caption": "Markdown 表格",
            "structured_label": {
                "kind": "table",
                "content": table_markdown,
                "format": "markdown",
                "source": "model",
            },
        },
    }

    (output_dir / "normalized" / "mineru" / "metadata-demo.json").write_text(
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
    assert '<div class="card-meta">' in html
    assert html.index("Rendered Markdown") < html.index("Caption：Markdown 表格")
    assert html.index("Rendered Markdown") < html.index("文件：normalized/mineru/metadata-demo.json")
    assert html.index('<img src="data:image/png;base64,') < html.index(str(image_path))
