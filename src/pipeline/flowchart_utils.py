from __future__ import annotations

import re
import unicodedata
from html import escape, unescape
from typing import Any

_BR_TAG_RE = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)
_EDGE_LABEL_RE = re.compile(r"\|(?P<label>[^|\n]+)\|")
_FLOWCHART_HEADER_RE = re.compile(
    r"(?im)^\s*(flowchart|graph)\s+(TD|TB|BT|RL|LR)\b"
)
_FLOWCHART_EDGE_RE = re.compile(r"(-->|-\.->|==>)")
_GRAPH_SIGNATURE_DROP_RE = re.compile(r"[\s\-_.,，、;；:：!?？！\"'`“”‘’·•()\[\]{}<>/\\|]+")
_AUXILIARY_FLOWCHART_NODE_RE = re.compile(
    r"^(split|merge|junction|join|branch|router|connector|hub)(\d+)?$",
    re.IGNORECASE,
)


def normalize_mermaid_text(text: str) -> str:
    value = _strip_mermaid_code_fences(str(text or ""))
    value = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not value:
        return ""
    sanitized_lines = [_sanitize_mermaid_line(line) for line in value.splitlines()]
    return "\n".join(sanitized_lines).strip()


def looks_like_mermaid(content: str) -> bool:
    text = normalize_mermaid_text(content)
    if not text:
        return False
    if not _has_mermaid_flow_signal(text):
        return False

    from src.graph_fusion import extract_weak_flowchart_graph_from_mermaid

    return extract_weak_flowchart_graph_from_mermaid(text) is not None


def flowchart_graph_from_mermaid(content: str) -> dict[str, Any] | None:
    text = normalize_mermaid_text(content)
    if not text:
        return None

    from src.graph_fusion import extract_weak_flowchart_graph_from_mermaid

    return extract_weak_flowchart_graph_from_mermaid(text)


def _has_mermaid_flow_signal(text: str) -> bool:
    if _FLOWCHART_HEADER_RE.search(text):
        return True
    return _FLOWCHART_EDGE_RE.search(text) is not None


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
    normalized_mermaid = normalize_mermaid_text(mermaid)
    if looks_like_mermaid(normalized_mermaid):
        patch["content"] = {"content": normalized_mermaid}
    if graph_payload is not None:
        patch["flowchart_graph"] = graph_payload
    return patch


def build_flowchart_patch_from_mermaid(mermaid: str) -> dict[str, Any]:
    normalized_mermaid = normalize_mermaid_text(mermaid)
    return build_flowchart_candidate_patch(
        mermaid=normalized_mermaid,
        graph_payload=flowchart_graph_from_mermaid(normalized_mermaid),
    )


def diff_flowchart_graphs(
    current_graph: dict[str, Any] | None,
    reference_graph: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    current_nodes = _node_signatures(current_graph)
    reference_nodes = _node_signatures(reference_graph)
    current_edges = _edge_signatures(current_graph, current_nodes)
    reference_edges = _edge_signatures(reference_graph, reference_nodes)

    diffs: list[dict[str, Any]] = []

    for node_key, reference_node in reference_nodes.items():
        if node_key not in current_nodes:
            diffs.append(
                {
                    "diff_kind": "missing_node",
                    "node_key": node_key,
                    "reference_node": reference_node,
                }
            )

    for edge_key, reference_edge in reference_edges.items():
        if edge_key in current_edges:
            continue
        reverse_key = _reverse_edge_key(edge_key)
        if reverse_key in current_edges:
            diffs.append(
                {
                    "diff_kind": "edge_direction_conflict",
                    "edge_key": edge_key,
                    "reference_edge": reference_edge,
                    "current_edge": current_edges[reverse_key],
                }
            )
            continue
        labelless_key = _edge_key_without_label(edge_key)
        current_labelless = {
            _edge_key_without_label(item_key): item for item_key, item in current_edges.items()
        }
        if labelless_key in current_labelless:
            diffs.append(
                {
                    "diff_kind": "edge_label_conflict",
                    "edge_key": edge_key,
                    "reference_edge": reference_edge,
                    "current_edge": current_labelless[labelless_key],
                }
            )
            continue
        diffs.append(
            {
                "diff_kind": "missing_edge",
                "edge_key": edge_key,
                "reference_edge": reference_edge,
            }
        )

    return diffs


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


def _node_signatures(graph_payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    comparison_nodes, _comparison_edges = _comparison_graph_components(graph_payload)
    if not comparison_nodes:
        return {}

    nodes: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(comparison_nodes, start=1):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "") or "").strip()
        node_id = str(item.get("node_id", "") or "").strip()
        signature = _normalize_graph_text(text) or f"id:{node_id or index}"
        nodes[signature] = {
            "node_id": node_id or f"N{index:03d}",
            "order_index": _coerce_int(item.get("order_index"), default=index) or index,
            "shape": str(item.get("shape", "") or "unknown").strip() or "unknown",
            "text": text or node_id or f"N{index:03d}",
        }
    return nodes


def _edge_signatures(
    graph_payload: dict[str, Any] | None,
    node_lookup: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    _comparison_nodes, comparison_edges = _comparison_graph_components(graph_payload)
    if not comparison_edges:
        return {}

    id_to_key = {
        str(node.get("node_id", "") or "").strip(): node_key
        for node_key, node in node_lookup.items()
    }
    edges: dict[str, dict[str, Any]] = {}
    for item in comparison_edges:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source", "") or "").strip()
        target_id = str(item.get("target", "") or "").strip()
        if not source_id or not target_id:
            continue
        source_key = id_to_key.get(source_id, f"id:{source_id}")
        target_key = id_to_key.get(target_id, f"id:{target_id}")
        label = str(item.get("label", "") or "").strip()
        signature = f"{source_key}|{_normalize_graph_text(label)}|{target_key}"
        edges[signature] = {
            "source": source_key,
            "target": target_key,
            "label": label,
            "source_text": node_lookup.get(source_key, {}).get("text", source_id),
            "target_text": node_lookup.get(target_key, {}).get("text", target_id),
        }
    return edges


def _reverse_edge_key(edge_key: str) -> str:
    source, label, target = edge_key.split("|", 2)
    return f"{target}|{label}|{source}"


def _edge_key_without_label(edge_key: str) -> str:
    source, _label, target = edge_key.split("|", 2)
    return f"{source}||{target}"


def _normalize_graph_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = unescape(text)
    text = _BR_TAG_RE.sub("", text)
    text = unicodedata.normalize("NFKC", text).lower()
    text = (
        text.replace("（", "(")
        .replace("）", ")")
        .replace("，", ",")
        .replace("、", ",")
        .replace("：", ":")
        .replace("；", ";")
        .replace("？", "?")
        .replace("！", "!")
        .replace("／", "/")
        .replace("－", "-")
        .replace("—", "-")
        .replace("–", "-")
    )
    return _GRAPH_SIGNATURE_DROP_RE.sub("", text)


def _comparison_graph_components(
    graph_payload: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(graph_payload, dict):
        return [], []
    raw_nodes = graph_payload.get("nodes")
    raw_edges = graph_payload.get("edges")
    if not isinstance(raw_nodes, list):
        return [], []

    normalized_nodes = [
        item for item in raw_nodes if isinstance(item, dict)
    ]
    normalized_edges = [
        item for item in raw_edges if isinstance(item, dict)
    ] if isinstance(raw_edges, list) else []
    auxiliary_node_ids = {
        str(node.get("node_id", "") or "").strip()
        for node in normalized_nodes
        if _is_auxiliary_flowchart_node(node)
    }
    comparison_nodes = [
        node
        for node in normalized_nodes
        if str(node.get("node_id", "") or "").strip() not in auxiliary_node_ids
    ]
    comparison_edges = _collapse_auxiliary_flowchart_edges(
        normalized_edges,
        auxiliary_node_ids,
    )
    return comparison_nodes, comparison_edges


def _is_auxiliary_flowchart_node(node_payload: dict[str, Any]) -> bool:
    node_id = str(node_payload.get("node_id", "") or "").strip()
    text = str(node_payload.get("text", "") or "").strip()
    shape = str(node_payload.get("shape", "") or "unknown").strip().lower()
    if not node_id or not text or shape not in {"", "unknown"}:
        return False
    normalized_text = _normalize_graph_text(text)
    return bool(
        normalized_text
        and _AUXILIARY_FLOWCHART_NODE_RE.fullmatch(normalized_text)
    )


def _collapse_auxiliary_flowchart_edges(
    raw_edges: list[dict[str, Any]],
    auxiliary_node_ids: set[str],
) -> list[dict[str, Any]]:
    edges = [
        {
            "source": str(item.get("source", "") or "").strip(),
            "target": str(item.get("target", "") or "").strip(),
            "label": str(item.get("label", "") or "").strip(),
        }
        for item in raw_edges
        if str(item.get("source", "") or "").strip()
        and str(item.get("target", "") or "").strip()
    ]
    if not auxiliary_node_ids:
        return _deduplicate_flowchart_edges(edges)

    collapsed_edges = list(edges)
    for auxiliary_node_id in sorted(auxiliary_node_ids):
        incoming_edges = [
            edge for edge in collapsed_edges if edge["target"] == auxiliary_node_id
        ]
        outgoing_edges = [
            edge for edge in collapsed_edges if edge["source"] == auxiliary_node_id
        ]
        collapsed_edges = [
            edge
            for edge in collapsed_edges
            if edge["source"] != auxiliary_node_id and edge["target"] != auxiliary_node_id
        ]
        if not incoming_edges or not outgoing_edges:
            continue
        for incoming_edge in incoming_edges:
            for outgoing_edge in outgoing_edges:
                source = incoming_edge["source"]
                target = outgoing_edge["target"]
                if (
                    not source
                    or not target
                    or source == target
                    or source in auxiliary_node_ids
                    or target in auxiliary_node_ids
                ):
                    continue
                collapsed_edges.append(
                    {
                        "source": source,
                        "target": target,
                        "label": _merge_collapsed_edge_labels(
                            incoming_edge["label"],
                            outgoing_edge["label"],
                        ),
                    }
                )
    return _deduplicate_flowchart_edges(
        [
            edge
            for edge in collapsed_edges
            if edge["source"] not in auxiliary_node_ids
            and edge["target"] not in auxiliary_node_ids
        ]
    )


def _merge_collapsed_edge_labels(left: str, right: str) -> str:
    left_label = str(left or "").strip()
    right_label = str(right or "").strip()
    if left_label and right_label:
        if _normalize_graph_text(left_label) == _normalize_graph_text(right_label):
            return right_label
        return f"{left_label} / {right_label}"
    return right_label or left_label


def _deduplicate_flowchart_edges(
    edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    ordered: list[dict[str, Any]] = []
    for edge in edges:
        source = str(edge.get("source", "") or "").strip()
        target = str(edge.get("target", "") or "").strip()
        label = str(edge.get("label", "") or "").strip()
        if not source or not target:
            continue
        key = "|".join(
            (
                _normalize_graph_text(source),
                _normalize_graph_text(label),
                _normalize_graph_text(target),
            )
        )
        if key in seen:
            continue
        seen.add(key)
        ordered.append(
            {
                "source": source,
                "target": target,
                "label": label,
            }
        )
    return ordered


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


def _strip_mermaid_code_fences(text: str) -> str:
    value = str(text or "").strip()
    if not value.startswith("```"):
        return value
    lines = value.splitlines()
    if lines:
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _sanitize_mermaid_line(line: str) -> str:
    if not line.strip():
        return ""
    sanitized = _sanitize_mermaid_edge_labels(line)
    return _sanitize_mermaid_node_labels(sanitized)


def _sanitize_mermaid_edge_labels(line: str) -> str:
    if "-->" not in line and "==>" not in line and "-.->" not in line:
        return line

    def _replace_label(match: re.Match[str]) -> str:
        label = _normalize_mermaid_edge_label(match.group("label"))
        return f"|{label}|"

    return _EDGE_LABEL_RE.sub(_replace_label, line)


def _sanitize_mermaid_node_labels(line: str) -> str:
    chars: list[str] = []
    position = 0
    while position < len(line):
        char = line[position]
        if not _is_mermaid_node_id_char(char):
            chars.append(char)
            position += 1
            continue
        if position > 0 and _is_mermaid_node_id_char(line[position - 1]):
            chars.append(char)
            position += 1
            continue

        node_end = position
        while node_end < len(line) and _is_mermaid_node_id_char(line[node_end]):
            node_end += 1
        node_id = line[position:node_end]
        open_token, close_token = _detect_mermaid_node_shape(line, node_end)
        if open_token is None or close_token is None:
            chars.append(node_id)
            position = node_end
            continue

        label_start = node_end + len(open_token)
        label_end = line.find(close_token, label_start)
        if label_end < 0:
            chars.append(node_id)
            position = node_end
            continue

        raw_label = line[label_start:label_end]
        sanitized_label = _sanitize_mermaid_node_label(raw_label)
        chars.append(node_id)
        chars.append(open_token)
        chars.append(sanitized_label)
        chars.append(close_token)
        position = label_end + len(close_token)
    return "".join(chars)


def _detect_mermaid_node_shape(
    line: str,
    position: int,
) -> tuple[str | None, str | None]:
    if line.startswith("((", position):
        return "((", "))"
    if line.startswith("[", position):
        return "[", "]"
    if line.startswith("{", position):
        return "{", "}"
    if line.startswith("(", position):
        return "(", ")"
    return None, None


def _sanitize_mermaid_node_label(label: str) -> str:
    text = str(label or "").strip()
    if not text:
        return text
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = text[1:-1].strip()
    text = _normalize_mermaid_node_text(text)
    return f'"{text}"'


def _normalize_mermaid_node_text(text: str) -> str:
    normalized = str(text or "").replace("\n", " ").strip()
    normalized = _BR_TAG_RE.sub("<br/>", normalized)
    return normalized.replace('"', '\\"')


def _normalize_mermaid_edge_label(label: str) -> str:
    normalized = str(label or "").replace("\n", " ").strip()
    normalized = _BR_TAG_RE.sub(" / ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.replace("|", "/")


def _is_mermaid_node_id_char(char: str) -> bool:
    return char.isalnum() or char in "_:-"
