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
    assert "Original Image" in html


def test_collect_mermaid_snapshots_supports_legacy_final_and_issue_candidate_payload(tmp_path) -> None:
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
    (output_dir / "final" / "demo_content_list_v2.json").write_text(
        json.dumps(
            [[{"type": "chart", "sub_type": "flowchart", "content": {"content": "flowchart TD\nA-->B"}}]],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output_dir / "final" / "demo_artifact.json").write_text(
        json.dumps(
            {
                "graph_fusion": {},
                "issues": [
                    {
                        "issue_type": "flowchart_graph_conflict",
                        "candidate_payload": {
                            "reference_mermaid": "flowchart TD\nN001-->N002",
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    snapshots = collect_mermaid_snapshots(image_id="demo", output_dir=output_dir)

    assert snapshots[2].status == "valid"
    assert snapshots[2].render_code == "flowchart TD\nN001-->N002"
    assert snapshots[3].status == "valid"
    assert snapshots[3].render_code == "flowchart TD\nA-->B"


def test_collect_mermaid_snapshots_supports_image_flowchart_blocks_and_qwen_issue_fallback(tmp_path) -> None:
    output_dir = tmp_path / "outputs"
    (output_dir / "normalized" / "mineru").mkdir(parents=True)
    (output_dir / "final").mkdir(parents=True)

    (output_dir / "normalized" / "mineru" / "figure1.json").write_text(
        json.dumps(
            {
                "document": {
                    "blocks": [
                        {
                            "type": "image",
                            "sub_type": "flowchart",
                            "text": "```mermaid\ngraph TD\nA-->B\n```",
                            "content": {
                                "image_caption": ["```mermaid\ngraph TD\nA-->B\n```"],
                                "img_path": "data/flowchart_crops/figure1.png",
                            },
                        }
                    ]
                },
                "derived_label": {
                    "image_type": "flowchart",
                    "caption": "```mermaid\ngraph TD\nA-->B\n```",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output_dir / "final" / "figure1.json").write_text(
        json.dumps(
            {
                "parsed": {
                    "filename": "figure1.png",
                    "extraction_results": [
                        {
                            "page": 0,
                            "json_res": [
                                {
                                    "type": "image",
                                    "sub_type": "flowchart",
                                    "content": "```mermaid\ngraph TD\nA-->B\n```",
                                }
                            ],
                        }
                    ],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output_dir / "final" / "figure1_artifact.json").write_text(
        json.dumps(
            {
                "graph_fusion": {},
                "issues": [
                    {
                        "issue_type": "flowchart_graph_conflict",
                        "qwen_block": {
                            "type": "chart",
                            "sub_type": "flowchart",
                            "content": {"content": "flowchart TD\nQ-->R"},
                        },
                        "candidate_payload": {
                            "reference_mermaid": "flowchart TD\nQ-->R",
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    snapshots = collect_mermaid_snapshots(image_id="figure1", output_dir=output_dir)

    assert snapshots[0].status == "valid"
    assert "A-->B" in snapshots[0].render_code
    assert snapshots[1].status == "valid"
    assert snapshots[1].render_code == "flowchart TD\nQ-->R"
    assert snapshots[2].status == "valid"
    assert snapshots[2].render_code == "flowchart TD\nQ-->R"
    assert snapshots[3].status == "valid"
    assert "A-->B" in snapshots[3].render_code


def test_collect_mermaid_snapshots_sanitizes_render_code_for_html_rendering(tmp_path) -> None:
    output_dir = tmp_path / "outputs"
    (output_dir / "normalized" / "mineru").mkdir(parents=True)
    (output_dir / "normalized" / "qwen").mkdir(parents=True)
    (output_dir / "final").mkdir(parents=True)

    (output_dir / "normalized" / "mineru" / "unsafe.json").write_text(
        json.dumps(
            {
                "document": {
                    "blocks": [
                        {
                            "type": "chart",
                            "sub_type": "flowchart",
                            "content": {
                                "content": (
                                    "flowchart TD\n"
                                    "Resectable[无转移，肿瘤可切除或<br/>临界可切除] --> "
                                    "LiverMRI[肝脏MRI<br/>或PET-CT]\n"
                                    "ChestAbdCT -->|发现囊性病变<br/>或CT无法确定| CT"
                                )
                            },
                        }
                    ]
                },
                "derived_label": None,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output_dir / "normalized" / "qwen" / "unsafe.json").write_text(
        json.dumps({"document": {"blocks": []}, "derived_label": None}, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "final" / "unsafe.json").write_text(
        json.dumps({"parsed": {"extraction_results": []}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "final" / "unsafe_artifact.json").write_text(
        json.dumps({"graph_fusion": {}}, ensure_ascii=False),
        encoding="utf-8",
    )

    snapshots = collect_mermaid_snapshots(image_id="unsafe", output_dir=output_dir)

    assert snapshots[0].status == "valid"
    assert 'LiverMRI["肝脏MRI<br/>或PET-CT"]' in snapshots[0].render_code
    assert "|发现囊性病变 / 或CT无法确定|" in snapshots[0].render_code
