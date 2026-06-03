from __future__ import annotations

from html import escape
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


def mermaid_from_flowchart_graph(graph_payload: dict[str, Any] | None) -> str:
    if not isinstance(graph_payload, dict):
        return ""

    raw_nodes = graph_payload.get("nodes")
    raw_edges = graph_payload.get("edges")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        return ""
    if not isinstance(raw_edges, list):
        raw_edges = []

    nodes: list[dict[str, Any]] = []
    for index, item in enumerate(raw_nodes, start=1):
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("node_id", "") or "").strip()
        if not node_id:
            continue
        order_index = _coerce_int(item.get("order_index"), default=index)
        nodes.append(
            {
                "node_id": node_id,
                "order_index": order_index if order_index is not None else index,
                "shape": str(item.get("shape", "") or "unknown").strip() or "unknown",
                "text": str(item.get("text", "") or "").strip() or node_id,
            }
        )

    if not nodes:
        return ""

    node_order_lookup = {item["node_id"]: int(item["order_index"]) for item in nodes}
    lines = ["flowchart TD"]
    for node in sorted(nodes, key=lambda item: (int(item["order_index"]), item["node_id"])):
        lines.append(
            _format_mermaid_node(
                node_id=str(node["node_id"]),
                text=str(node["text"]),
                shape=str(node["shape"]),
            )
        )

    edges: list[dict[str, str]] = []
    for item in raw_edges:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "") or "").strip()
        target = str(item.get("target", "") or "").strip()
        if not source or not target:
            continue
        edges.append(
            {
                "source": source,
                "target": target,
                "label": str(item.get("label", "") or "").strip(),
            }
        )

    for edge in sorted(
        edges,
        key=lambda item: (
            node_order_lookup.get(item["source"], 10**9),
            node_order_lookup.get(item["target"], 10**9),
            item["source"],
            item["target"],
        ),
    ):
        if edge["label"]:
            lines.append(
                f'{edge["source"]} -->|{_escape_mermaid_label(edge["label"])}| {edge["target"]}'
            )
        else:
            lines.append(f'{edge["source"]} --> {edge["target"]}')

    return "\n".join(lines)


def _first_vote(values: Any) -> int | None:
    if not isinstance(values, list) or not values:
        return None
    first = values[0]
    try:
        return int(first)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _format_mermaid_node(node_id: str, text: str, shape: str) -> str:
    escaped_text = _escape_mermaid_text(text)
    normalized_shape = str(shape or "").strip().lower()
    if normalized_shape == "diamond":
        return f'{node_id}{{"{escaped_text}"}}'
    if normalized_shape == "rounded":
        return f'{node_id}("{escaped_text}")'
    if normalized_shape == "ellipse":
        return f'{node_id}(("{escaped_text}"))'
    return f'{node_id}["{escaped_text}"]'


def _escape_mermaid_text(text: str) -> str:
    return escape(str(text or "").strip(), quote=False).replace('"', '\\"')


def _escape_mermaid_label(text: str) -> str:
    return _escape_mermaid_text(text).replace("|", "/")
