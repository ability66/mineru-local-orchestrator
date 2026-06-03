from __future__ import annotations

import json

from src.render_mermaid_compare import collect_mermaid_snapshots, generate_compare_page


def test_collect_mermaid_snapshots_derives_mermaid_from_graph_payload(tmp_path) -> None:
    output_dir = tmp_path / "outputs"
    (output_dir / "normalized" / "mineru").mkdir(parents=True)
    (output_dir / "normalized" / "qwen").mkdir(parents=True)
    (output_dir / "final").mkdir(parents=True)

    mineru_payload = {
        "document": {
            "blocks": [
                {
                    "type": "chart",
                    "sub_type": "flowchart",
                    "flowchart_graph": {
                        "nodes": [
                            {"node_id": "N001", "order_index": 1, "shape": "rectangle", "text": "开始"},
                            {"node_id": "N002", "order_index": 2, "shape": "diamond", "text": "审批"},
                        ],
                        "edges": [{"source": "N001", "target": "N002", "label": ""}],
                    },
                }
            ]
        },
        "derived_label": None,
    }
    qwen_payload = {
        "document": {
            "blocks": [
                {
                    "type": "chart",
                    "sub_type": "flowchart",
                    "content": {"content": "flowchart TD\nA-->B"},
                    "structured_label": {
                        "kind": "mermaid",
                        "content": "flowchart TD\nA-->B",
                        "format": "mermaid",
                    },
                }
            ]
        },
        "derived_label": None,
    }
    final_payload = {
        "parsed": {
            "extraction_results": [
                {
                    "page": 0,
                    "json_res": [
                        {
                            "type": "chart",
                            "sub_type": "flowchart",
                            "content": "flowchart TD\nN001-->N002",
                        }
                    ],
                }
            ]
        }
    }
    artifact_payload = {
        "graph_fusion": {
            "mermaid": "flowchart TD\nN001-->N002",
            "fusion_status": "fused",
            "fusion_method": "visual_order",
        }
    }

    (output_dir / "normalized" / "mineru" / "demo.json").write_text(
        json.dumps(mineru_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "normalized" / "qwen" / "demo.json").write_text(
        json.dumps(qwen_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "final" / "demo.json").write_text(
        json.dumps(final_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "final" / "demo_artifact.json").write_text(
        json.dumps(artifact_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    snapshots = collect_mermaid_snapshots(image_id="demo", output_dir=output_dir)

    assert snapshots[0].status == "derived"
    assert snapshots[0].render_code.startswith("flowchart TD")
    assert snapshots[1].status == "valid"
    assert snapshots[1].render_code == "flowchart TD\nA-->B"
    assert snapshots[2].status == "valid"
    assert snapshots[3].status == "valid"


def test_generate_compare_page_is_offline_and_copies_local_asset(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "outputs"
    (output_dir / "normalized" / "mineru").mkdir(parents=True)
    (output_dir / "normalized" / "qwen").mkdir(parents=True)
    (output_dir / "final").mkdir(parents=True)
    (output_dir / "normalized" / "mineru" / "demo.json").write_text(
        json.dumps({"document": {"blocks": []}, "derived_label": None}, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "normalized" / "qwen" / "demo.json").write_text(
        json.dumps({"document": {"blocks": []}, "derived_label": None}, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "final" / "demo.json").write_text(
        json.dumps({"parsed": {"extraction_results": []}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "final" / "demo_artifact.json").write_text(
        json.dumps({"graph_fusion": {}}, ensure_ascii=False),
        encoding="utf-8",
    )

    vendor_path = tmp_path / "vendor" / "mermaid.min.js"
    vendor_path.parent.mkdir(parents=True)
    vendor_path.write_text("window.mermaid = { initialize(){}, render: async () => ({ svg: '<svg></svg>' }) };", encoding="utf-8")
    monkeypatch.setattr("src.render_mermaid_compare.MERMAID_VENDOR_PATH", vendor_path)

    html_path = generate_compare_page(
        image_id="demo",
        output_dir=output_dir,
        compare_dir=output_dir / "compare_mermaid",
    )

    html = html_path.read_text(encoding="utf-8")
    assert "assets/mermaid.min.js" in html
    assert "cdn" not in html.lower()
    assert (output_dir / "compare_mermaid" / "assets" / "mermaid.min.js").exists()
