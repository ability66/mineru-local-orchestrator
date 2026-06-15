from __future__ import annotations

from src.flowvqa_eval import build_flowvqa_eval_payload, build_mermaid_render_code
from src.pipeline.flowchart_utils import mermaid_from_flowchart_graph


def test_build_mermaid_render_code_removes_escaped_quotes() -> None:
    unsafe_mermaid = mermaid_from_flowchart_graph(
        {
            "nodes": [
                {
                    "node_id": "N001",
                    "order_index": 1,
                    "shape": "rounded",
                    "text": '["Start"]',
                },
                {
                    "node_id": "N002",
                    "order_index": 2,
                    "shape": "rectangle",
                    "text": '"/Access/"',
                },
            ],
            "edges": [{"source": "N001", "target": "N002", "label": ""}],
        }
    )

    assert '\\"' in unsafe_mermaid
    render_code = build_mermaid_render_code(unsafe_mermaid)

    assert render_code.startswith("flowchart TD")
    assert '\\"' not in render_code


def test_build_flowvqa_eval_payload_uses_render_safe_ground_truth() -> None:
    unsafe_mermaid = mermaid_from_flowchart_graph(
        {
            "nodes": [
                {
                    "node_id": "N001",
                    "order_index": 1,
                    "shape": "rounded",
                    "text": '["Start"]',
                },
                {
                    "node_id": "N002",
                    "order_index": 2,
                    "shape": "rectangle",
                    "text": "Access",
                },
            ],
            "edges": [{"source": "N001", "target": "N002", "label": ""}],
        }
    )

    payload = build_flowvqa_eval_payload(
        reference={
            "sample_id": "demo",
            "split": "test",
            "source_path": "Data/test_full.json",
            "question_count": 2,
            "ground_truth_mermaid": unsafe_mermaid,
        },
        predictions_by_source={
            "mineru_raw": "flowchart TD\nM[Start] --> N[Access]",
            "final": "flowchart TD\nA[Start] --> B[Access]",
        },
        source_meta_by_source={
            "mineru_raw": {
                "title": "MinerU Raw",
                "source_path": "raw/mineru/demo.json",
            },
            "final": {
                "title": "Ours",
                "source_path": "final/demo.json",
            },
        },
    )

    assert payload is not None
    assert payload["ground_truth_mermaid"] == unsafe_mermaid
    assert '\\"' not in payload["ground_truth_render_code"]
    assert payload["mermaid_by_source"]["mineru_raw"]["title"] == "MinerU Raw"
    assert payload["mermaid_by_source"]["mineru_raw"]["source_path"] == "raw/mineru/demo.json"
    assert payload["mermaid_by_source"]["final"]["title"] == "Ours"
