from __future__ import annotations

import json

from src.render_compare_dashboard import generate_compare_dashboard


def test_generate_compare_dashboard_builds_dropdown_page(tmp_path) -> None:
    output_dir = tmp_path / "outputs"
    (output_dir / "normalized" / "mineru").mkdir(parents=True)
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
    assert "Qwen" in html
    assert "Final" in html


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
