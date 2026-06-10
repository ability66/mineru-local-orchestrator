from __future__ import annotations

from eval_dataset.mermaid_td_f1.evaluator import (
    evaluate_mermaid_flowchart,
    normalize_virtual_nodes,
    parse_mermaid_flowchart,
)


def edge_set(graph) -> set[tuple[str, str, str, str]]:
    return {(edge.source, edge.target, edge.text, edge.kind) for edge in graph.edges}


def test_root_ignores_virtual_node() -> None:
    graph = parse_mermaid_flowchart(
        """flowchart TD
  V[ ] --> A[开始]
  A --> B[处理]
"""
    )

    assert graph.root == "A"
    assert graph.nodes["V"].is_virtual is True


def test_virtual_routing_nodes_are_contracted_without_penalty() -> None:
    gold = """flowchart TD
  A[处理A] --> C[处理C]
  A[处理A] --> D[处理D]
  B[处理B] --> C[处理C]
  B[处理B] --> D[处理D]
"""
    pred = """flowchart TD
  A[处理A] --> V[ ]
  B[处理B] --> V
  V --> C[处理C]
  V --> D[处理D]
"""

    normalized_pred = normalize_virtual_nodes(parse_mermaid_flowchart(pred))
    normalized_gold = normalize_virtual_nodes(parse_mermaid_flowchart(gold))

    assert "V" not in normalized_pred.nodes
    assert edge_set(normalized_pred) == edge_set(normalized_gold)

    result = evaluate_mermaid_flowchart(pred_mermaid=pred, gold_mermaid=gold)

    assert result["structure_f1"] > 0.99
    assert result["node_text_f1"] > 0.99
    assert result["edge_text_f1"] > 0.99
    assert result["binding_f1"] > 0.99
    assert result["final_td_f1"] > 0.99


def test_chained_virtual_nodes_are_contracted_iteratively() -> None:
    gold = """flowchart TD
  A[开始] --> B[处理]
"""
    pred = """flowchart TD
  A[开始] --> V1[ ]
  V1 --> V2[ ]
  V2 --> B[处理]
"""

    normalized_pred = normalize_virtual_nodes(parse_mermaid_flowchart(pred))

    assert "V1" not in normalized_pred.nodes
    assert "V2" not in normalized_pred.nodes
    assert edge_set(normalized_pred) == {("A", "B", "", "solid")}

    result = evaluate_mermaid_flowchart(pred_mermaid=pred, gold_mermaid=gold)

    assert result["final_td_f1"] > 0.99
