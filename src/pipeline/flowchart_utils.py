from __future__ import annotations

from typing import Any


def looks_like_mermaid(content: str) -> bool:
    text = str(content or "").strip()
    if not text:
        return False

    from src.graph_fusion import extract_weak_flowchart_graph_from_mermaid

    return extract_weak_flowchart_graph_from_mermaid(text) is not None


def build_flowchart_graph_payload(graph_fusion_result: Any) -> dict[str, Any] | None:
    if graph_fusion_result is None:
        return None

    nodes = getattr(graph_fusion_result, "nodes", None)
    edges = getattr(graph_fusion_result, "edges", None)
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return None

    normalized_nodes: list[dict[str, Any]] = []
    for node in nodes:
        fused_id = str(getattr(node, "fused_id", "") or "").strip()
        if not fused_id:
            continue
        bbox_hints = getattr(node, "bbox_hints", None)
        bbox_hint = bbox_hints[0] if isinstance(bbox_hints, list) and bbox_hints else None
        normalized_nodes.append(
            {
                "node_id": fused_id,
                "order_index": getattr(node, "order_index", None),
                "row_index": _first_vote(getattr(node, "row_index_votes", None)),
                "col_index": _first_vote(getattr(node, "col_index_votes", None)),
                "bbox_hint": bbox_hint if isinstance(bbox_hint, list) else None,
                "shape": str(getattr(node, "representative_shape", "") or "unknown").strip() or "unknown",
                "text": str(getattr(node, "representative_text", "") or "").strip(),
            }
        )

    normalized_edges: list[dict[str, Any]] = []
    for edge in edges:
        source = str(getattr(edge, "source", "") or "").strip()
        target = str(getattr(edge, "target", "") or "").strip()
        if not source or not target:
            continue
        normalized_edges.append(
            {
                "source": source,
                "target": target,
                "label": str(getattr(edge, "label", "") or "").strip(),
            }
        )

    if not normalized_nodes:
        return None

    return {
        "node_order_rule": "fused_visual_order",
        "nodes": normalized_nodes,
        "edges": normalized_edges,
        "graph_source": "fused_graph",
        "weak_candidate": False,
    }


def build_flowchart_candidate_patch(
    mermaid: str,
    graph_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    patch: dict[str, Any] = {
        "type": "chart",
        "sub_type": "flowchart",
    }
    if looks_like_mermaid(mermaid):
        patch["content"] = {"content": str(mermaid or "").strip()}
    if graph_payload is not None:
        patch["flowchart_graph"] = graph_payload
    return patch


def _first_vote(values: Any) -> int | None:
    if not isinstance(values, list) or not values:
        return None
    first = values[0]
    try:
        return int(first)
    except (TypeError, ValueError):
        return None
