from __future__ import annotations

import json

from src.render_compare_dashboard import generate_compare_dashboard


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
            "image_type": "natural_image",
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
