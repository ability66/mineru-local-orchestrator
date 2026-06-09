from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
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
_NODE_MATCH_THRESHOLD = 0.65
_EDGE_MATCH_THRESHOLD = 0.70
_FLOWCHART_OBJECTIVE_WEIGHTS = {
    "node": 0.45,
    "edge": 0.40,
    "path": 0.10,
    "subgraph": 0.05,
}
_GRAPH_STRUCTURE_WEIGHTS = {
    "node_f1": 0.35,
    "edge_f1": 0.45,
    "path": 0.10,
    "subgraph": 0.05,
    "collapse_penalty": 0.025,
    "split_penalty": 0.025,
}
_MERMAID_SCORE_WEIGHTS = {
    "canonical_text": 0.10,
    "ast": 0.15,
    "graph_structure": 0.45,
    "diagram_semantic": 0.20,
    "render_or_visual": 0.10,
}


@dataclass
class _ComparableNode:
    node_id: str
    label: str
    normalized_label: str
    shape: str
    parent: str
    normalized_parent: str
    order_index: int
    order_ratio: float
    in_degree: int
    out_degree: int
    depth_from_start: int | None
    depth_to_end: int | None
    is_source: bool
    is_sink: bool
    incoming_context: Counter[str]
    outgoing_context: Counter[str]


@dataclass
class _ComparableEdge:
    edge_id: str
    source: str
    target: str
    label: str
    normalized_label: str
    edge_type: str


@dataclass
class _ComparableGraph:
    nodes: dict[str, _ComparableNode]
    edges: list[_ComparableEdge]
    edge_lookup: dict[tuple[str, str], list[_ComparableEdge]]
    reachability: dict[str, set[str]]
    label_groups: dict[str, list[str]]
    subgraph_groups: Counter[str]


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
    return score_flowchart_graph_similarity(
        current_graph=current_graph,
        reference_graph=reference_graph,
    )["diffs"]


def score_mermaid_similarity(
    current_mermaid: str,
    reference_mermaid: str,
) -> dict[str, Any]:
    normalized_current = normalize_mermaid_text(current_mermaid)
    normalized_reference = normalize_mermaid_text(reference_mermaid)
    canonical_text_sim = _normalized_text_similarity(
        normalized_current,
        normalized_reference,
    )

    current_graph = flowchart_graph_from_mermaid(normalized_current)
    reference_graph = flowchart_graph_from_mermaid(normalized_reference)
    syntax_valid = int(
        bool(current_graph)
        and bool(reference_graph)
        and looks_like_mermaid(normalized_current)
        and looks_like_mermaid(normalized_reference)
    )
    if syntax_valid == 0:
        return {
            "syntax_valid": 0,
            "canonical_text_sim": round(canonical_text_sim, 4),
            "ast_sim": 0.0,
            "graph_structure_sim": 0.0,
            "diagram_semantic_sim": 0.0,
            "render_or_visual_sim": 0.0,
            "mermaid_score": round(min(0.20, canonical_text_sim * 0.20), 4),
        }

    graph_result = score_flowchart_graph_similarity(
        current_graph=current_graph,
        reference_graph=reference_graph,
    )
    ast_sim = _flowchart_ast_similarity(current_graph, reference_graph)
    diagram_semantic_sim = _diagram_semantic_similarity(graph_result)
    render_or_visual_sim = _render_or_visual_similarity(
        normalized_current,
        normalized_reference,
        current_graph,
        reference_graph,
    )
    mermaid_score = _clamp_score(
        _MERMAID_SCORE_WEIGHTS["canonical_text"] * canonical_text_sim
        + _MERMAID_SCORE_WEIGHTS["ast"] * ast_sim
        + _MERMAID_SCORE_WEIGHTS["graph_structure"]
        * float(graph_result.get("graph_structure_sim", 0.0))
        + _MERMAID_SCORE_WEIGHTS["diagram_semantic"] * diagram_semantic_sim
        + _MERMAID_SCORE_WEIGHTS["render_or_visual"] * render_or_visual_sim
    )
    return {
        "syntax_valid": 1,
        "canonical_text_sim": round(canonical_text_sim, 4),
        "ast_sim": round(ast_sim, 4),
        "graph_structure_sim": round(
            float(graph_result.get("graph_structure_sim", 0.0)),
            4,
        ),
        "diagram_semantic_sim": round(diagram_semantic_sim, 4),
        "render_or_visual_sim": round(render_or_visual_sim, 4),
        "mermaid_score": round(mermaid_score, 4),
        "graph_metrics": graph_result,
    }


def score_flowchart_graph_similarity(
    current_graph: dict[str, Any] | None,
    reference_graph: dict[str, Any] | None,
) -> dict[str, Any]:
    reference = _build_comparable_graph(reference_graph)
    current = _build_comparable_graph(current_graph)

    if not reference.nodes and not current.nodes:
        return {
            "node_precision": 1.0,
            "node_recall": 1.0,
            "node_f1": 1.0,
            "edge_precision": 1.0,
            "edge_recall": 1.0,
            "edge_f1": 1.0,
            "node_agreement": 1.0,
            "edge_agreement": 1.0,
            "path_agreement": 1.0,
            "subgraph_agreement": 1.0,
            "duplicate_collapse_penalty": 0.0,
            "node_split_penalty": 0.0,
            "graph_structure_sim": 1.0,
            "matched_node_count": 0,
            "matched_edge_count": 0,
            "node_match_threshold": _NODE_MATCH_THRESHOLD,
            "edge_match_threshold": _EDGE_MATCH_THRESHOLD,
            "diffs": [],
            "mapping": {},
        }
    if not reference.nodes or not current.nodes:
        missing_diffs = []
        for node in reference.nodes.values():
            missing_diffs.append(
                {
                    "diff_kind": "missing_node",
                    "node_key": _node_key(reference, node),
                    "reference_node": _node_payload(node),
                }
            )
        for edge in reference.edges:
            missing_diffs.append(
                {
                    "diff_kind": "missing_edge",
                    "edge_key": _edge_key(reference, edge),
                    "reference_edge": _edge_payload(reference, edge),
                }
            )
        return {
            "node_precision": 0.0,
            "node_recall": 0.0,
            "node_f1": 0.0,
            "edge_precision": 0.0,
            "edge_recall": 0.0,
            "edge_f1": 0.0,
            "node_agreement": 0.0,
            "edge_agreement": 0.0,
            "path_agreement": 0.0,
            "subgraph_agreement": 0.0,
            "duplicate_collapse_penalty": 1.0 if reference.nodes else 0.0,
            "node_split_penalty": 0.0,
            "graph_structure_sim": 0.0,
            "matched_node_count": 0,
            "matched_edge_count": 0,
            "node_match_threshold": _NODE_MATCH_THRESHOLD,
            "edge_match_threshold": _EDGE_MATCH_THRESHOLD,
            "diffs": missing_diffs,
            "mapping": {},
        }

    node_sim_lookup = _node_similarity_lookup(reference, current)
    mapping = _initial_node_mapping(reference, current, node_sim_lookup)
    mapping = _refine_node_mapping(reference, current, mapping, node_sim_lookup)
    return _evaluate_flowchart_mapping(reference, current, mapping, node_sim_lookup)


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


def _build_comparable_graph(graph_payload: dict[str, Any] | None) -> _ComparableGraph:
    comparison_nodes, comparison_edges = _comparison_graph_components(graph_payload)
    if not comparison_nodes:
        return _ComparableGraph(
            nodes={},
            edges=[],
            edge_lookup={},
            reachability={},
            label_groups={},
            subgraph_groups=Counter(),
        )

    nodes: dict[str, dict[str, Any]] = {}
    node_ids: list[str] = []
    for index, item in enumerate(comparison_nodes, start=1):
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("node_id", "") or "").strip() or f"N{index:03d}"
        label = str(item.get("text", "") or "").strip() or node_id
        nodes[node_id] = {
            "node_id": node_id,
            "label": label,
            "normalized_label": _normalize_graph_text(label),
            "shape": str(item.get("shape", "") or "unknown").strip().lower() or "unknown",
            "parent": str(item.get("parent", "") or "").strip(),
            "normalized_parent": _normalize_graph_text(item.get("parent", "")),
            "order_index": _coerce_int(item.get("order_index"), default=index) or index,
            "incoming_context": Counter(),
            "outgoing_context": Counter(),
            "in_degree": 0,
            "out_degree": 0,
        }
        node_ids.append(node_id)

    edges: list[_ComparableEdge] = []
    edge_lookup: dict[tuple[str, str], list[_ComparableEdge]] = defaultdict(list)
    outgoing_neighbors: dict[str, set[str]] = defaultdict(set)
    incoming_neighbors: dict[str, set[str]] = defaultdict(set)
    for index, item in enumerate(comparison_edges, start=1):
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "") or "").strip()
        target = str(item.get("target", "") or "").strip()
        if source not in nodes or target not in nodes:
            continue
        label = str(item.get("label", "") or "").strip()
        edge_type = str(item.get("edge_type", "") or "-->").strip() or "-->"
        edge = _ComparableEdge(
            edge_id=f"E{index:03d}",
            source=source,
            target=target,
            label=label,
            normalized_label=_normalize_graph_text(label),
            edge_type=edge_type,
        )
        edges.append(edge)
        edge_lookup[(source, target)].append(edge)
        nodes[source]["out_degree"] += 1
        nodes[target]["in_degree"] += 1
        outgoing_neighbors[source].add(target)
        incoming_neighbors[target].add(source)

    for edge in edges:
        source_label = str(nodes[edge.source]["normalized_label"] or f"id:{edge.source}")
        target_label = str(nodes[edge.target]["normalized_label"] or f"id:{edge.target}")
        nodes[edge.target]["incoming_context"].update(
            [f"{source_label}|{edge.normalized_label}|{edge.edge_type}"]
        )
        nodes[edge.source]["outgoing_context"].update(
            [f"{target_label}|{edge.normalized_label}|{edge.edge_type}"]
        )

    depth_from_start = _directed_distance_lookup(
        node_ids=node_ids,
        neighbors=outgoing_neighbors,
        seeds=[
            node_id
            for node_id, payload in nodes.items()
            if int(payload["in_degree"]) == 0
        ],
    )
    depth_to_end = _directed_distance_lookup(
        node_ids=node_ids,
        neighbors=incoming_neighbors,
        seeds=[
            node_id
            for node_id, payload in nodes.items()
            if int(payload["out_degree"]) == 0
        ],
    )
    reachability = _reachability_lookup(node_ids=node_ids, neighbors=outgoing_neighbors)
    label_groups: dict[str, list[str]] = defaultdict(list)
    subgraph_groups: Counter[str] = Counter()
    comparable_nodes: dict[str, _ComparableNode] = {}
    order_denominator = max(len(node_ids) - 1, 1)
    for order_position, node_id in enumerate(
        sorted(node_ids, key=lambda value: int(nodes[value]["order_index"])),
        start=0,
    ):
        payload = nodes[node_id]
        normalized_label = str(payload["normalized_label"] or f"id:{node_id}")
        label_groups[normalized_label].append(node_id)
        normalized_parent = str(payload["normalized_parent"])
        if normalized_parent:
            subgraph_groups[normalized_parent] += 1
        comparable_nodes[node_id] = _ComparableNode(
            node_id=node_id,
            label=str(payload["label"]),
            normalized_label=normalized_label,
            shape=str(payload["shape"]),
            parent=str(payload["parent"]),
            normalized_parent=normalized_parent,
            order_index=int(payload["order_index"]),
            order_ratio=order_position / order_denominator if node_ids else 0.0,
            in_degree=int(payload["in_degree"]),
            out_degree=int(payload["out_degree"]),
            depth_from_start=depth_from_start.get(node_id),
            depth_to_end=depth_to_end.get(node_id),
            is_source=int(payload["in_degree"]) == 0,
            is_sink=int(payload["out_degree"]) == 0,
            incoming_context=Counter(payload["incoming_context"]),
            outgoing_context=Counter(payload["outgoing_context"]),
        )

    return _ComparableGraph(
        nodes=comparable_nodes,
        edges=edges,
        edge_lookup=dict(edge_lookup),
        reachability=reachability,
        label_groups={key: list(value) for key, value in label_groups.items()},
        subgraph_groups=subgraph_groups,
    )


def _node_similarity_lookup(
    reference: _ComparableGraph,
    current: _ComparableGraph,
) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    for reference_node in reference.nodes.values():
        for current_node in current.nodes.values():
            scores[(reference_node.node_id, current_node.node_id)] = _node_similarity(
                reference_node,
                current_node,
            )
    return scores


def _initial_node_mapping(
    reference: _ComparableGraph,
    current: _ComparableGraph,
    node_sim_lookup: dict[tuple[str, str], float],
) -> dict[str, str | None]:
    reference_ids = list(reference.nodes)
    current_ids = list(current.nodes)
    if not reference_ids or not current_ids:
        return {node_id: None for node_id in reference_ids}

    size = len(reference_ids) + len(current_ids)
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    for row_index, reference_id in enumerate(reference_ids):
        for column_index, current_id in enumerate(current_ids):
            matrix[row_index][column_index] = float(
                node_sim_lookup.get((reference_id, current_id), 0.0)
            )

    assignment = _hungarian_max_assignment(matrix)
    mapping: dict[str, str | None] = {node_id: None for node_id in reference_ids}
    for row_index, column_index in enumerate(assignment[: len(reference_ids)]):
        if 0 <= column_index < len(current_ids):
            mapping[reference_ids[row_index]] = current_ids[column_index]
    return mapping


def _refine_node_mapping(
    reference: _ComparableGraph,
    current: _ComparableGraph,
    mapping: dict[str, str | None],
    node_sim_lookup: dict[tuple[str, str], float],
) -> dict[str, str | None]:
    best_mapping = dict(mapping)
    best_result = _evaluate_flowchart_mapping(
        reference,
        current,
        best_mapping,
        node_sim_lookup,
    )
    best_score = float(best_result.get("objective_score", 0.0))

    for _attempt in range(2):
        improved = False
        matched_reference_ids = [
            node_id for node_id, current_id in best_mapping.items() if current_id is not None
        ]
        for left_index, left_reference_id in enumerate(matched_reference_ids):
            for right_reference_id in matched_reference_ids[left_index + 1 :]:
                candidate_mapping = dict(best_mapping)
                candidate_mapping[left_reference_id], candidate_mapping[right_reference_id] = (
                    candidate_mapping[right_reference_id],
                    candidate_mapping[left_reference_id],
                )
                candidate_result = _evaluate_flowchart_mapping(
                    reference,
                    current,
                    candidate_mapping,
                    node_sim_lookup,
                )
                candidate_score = float(candidate_result.get("objective_score", 0.0))
                if candidate_score > best_score + 1e-6:
                    best_mapping = candidate_mapping
                    best_result = candidate_result
                    best_score = candidate_score
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue

        current_ids = list(current.nodes)
        for reference_id in list(best_mapping):
            used_current_ids = {
                node_id for node_id in best_mapping.values() if node_id is not None
            }
            current_choice = best_mapping.get(reference_id)
            candidate_current_ids = [None] + [
                node_id for node_id in current_ids if node_id not in used_current_ids
            ]
            if current_choice is not None:
                candidate_current_ids.append(current_choice)
            for candidate_current_id in candidate_current_ids:
                if candidate_current_id == current_choice:
                    continue
                candidate_mapping = dict(best_mapping)
                candidate_mapping[reference_id] = candidate_current_id
                candidate_result = _evaluate_flowchart_mapping(
                    reference,
                    current,
                    candidate_mapping,
                    node_sim_lookup,
                )
                candidate_score = float(candidate_result.get("objective_score", 0.0))
                if candidate_score > best_score + 1e-6:
                    best_mapping = candidate_mapping
                    best_result = candidate_result
                    best_score = candidate_score
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best_mapping


def _evaluate_flowchart_mapping(
    reference: _ComparableGraph,
    current: _ComparableGraph,
    mapping: dict[str, str | None],
    node_sim_lookup: dict[tuple[str, str], float],
) -> dict[str, Any]:
    matched_pairs: dict[str, str] = {}
    node_similarities: dict[str, float] = {}
    for reference_id, current_id in mapping.items():
        if current_id is None:
            continue
        similarity = float(node_sim_lookup.get((reference_id, current_id), 0.0))
        if similarity < _NODE_MATCH_THRESHOLD:
            continue
        matched_pairs[reference_id] = current_id
        node_similarities[reference_id] = similarity

    matched_reference_count = len(matched_pairs)
    matched_current_count = len({node_id for node_id in matched_pairs.values()})
    node_precision = matched_current_count / max(len(current.nodes), 1)
    node_recall = matched_reference_count / max(len(reference.nodes), 1)
    node_f1 = _f1_score(node_precision, node_recall)
    node_agreement = (
        sum(node_similarities.values()) / len(node_similarities)
        if node_similarities
        else 0.0
    )

    subgraph_scores = [
        _subgraph_similarity(reference.nodes[reference_id], current.nodes[current_id])
        for reference_id, current_id in matched_pairs.items()
    ]
    subgraph_agreement = (
        sum(subgraph_scores) / len(subgraph_scores) if subgraph_scores else 0.0
    )
    path_agreement = _path_agreement(reference, current, matched_pairs)
    edge_metrics = _evaluate_edge_alignment(reference, current, matched_pairs)
    duplicate_collapse_penalty = _duplicate_collapse_penalty(reference, matched_pairs)
    node_split_penalty = _node_split_penalty(
        reference,
        current,
        matched_pairs,
        node_sim_lookup,
    )
    graph_structure_sim = _clamp_score(
        _GRAPH_STRUCTURE_WEIGHTS["node_f1"] * node_f1
        + _GRAPH_STRUCTURE_WEIGHTS["edge_f1"] * edge_metrics["edge_f1"]
        + _GRAPH_STRUCTURE_WEIGHTS["path"] * path_agreement
        + _GRAPH_STRUCTURE_WEIGHTS["subgraph"] * subgraph_agreement
        - _GRAPH_STRUCTURE_WEIGHTS["collapse_penalty"] * duplicate_collapse_penalty
        - _GRAPH_STRUCTURE_WEIGHTS["split_penalty"] * node_split_penalty
    )
    objective_score = _clamp_score(
        _FLOWCHART_OBJECTIVE_WEIGHTS["node"] * node_agreement
        + _FLOWCHART_OBJECTIVE_WEIGHTS["edge"] * edge_metrics["edge_f1"]
        + _FLOWCHART_OBJECTIVE_WEIGHTS["path"] * path_agreement
        + _FLOWCHART_OBJECTIVE_WEIGHTS["subgraph"] * subgraph_agreement
    )

    return {
        "node_precision": round(node_precision, 4),
        "node_recall": round(node_recall, 4),
        "node_f1": round(node_f1, 4),
        "edge_precision": round(edge_metrics["edge_precision"], 4),
        "edge_recall": round(edge_metrics["edge_recall"], 4),
        "edge_f1": round(edge_metrics["edge_f1"], 4),
        "node_agreement": round(node_agreement, 4),
        "edge_agreement": round(edge_metrics["edge_f1"], 4),
        "path_agreement": round(path_agreement, 4),
        "subgraph_agreement": round(subgraph_agreement, 4),
        "duplicate_collapse_penalty": round(duplicate_collapse_penalty, 4),
        "node_split_penalty": round(node_split_penalty, 4),
        "graph_structure_sim": round(graph_structure_sim, 4),
        "matched_node_count": matched_reference_count,
        "matched_edge_count": edge_metrics["matched_edge_count"],
        "node_match_threshold": _NODE_MATCH_THRESHOLD,
        "edge_match_threshold": _EDGE_MATCH_THRESHOLD,
        "diffs": _missing_reference_nodes(reference, matched_pairs)
        + edge_metrics["diffs"],
        "mapping": dict(matched_pairs),
        "objective_score": round(objective_score, 4),
    }


def _evaluate_edge_alignment(
    reference: _ComparableGraph,
    current: _ComparableGraph,
    matched_pairs: dict[str, str],
) -> dict[str, Any]:
    used_current_edge_ids: set[str] = set()
    matched_reference_edge_count = 0
    diffs: list[dict[str, Any]] = []

    for reference_edge in reference.edges:
        mapped_source = matched_pairs.get(reference_edge.source)
        mapped_target = matched_pairs.get(reference_edge.target)
        if mapped_source is None or mapped_target is None:
            diffs.append(
                {
                    "diff_kind": "missing_edge",
                    "edge_key": _edge_key(reference, reference_edge),
                    "reference_edge": _edge_payload(reference, reference_edge),
                }
            )
            continue

        same_direction_candidates = [
            edge
            for edge in current.edge_lookup.get((mapped_source, mapped_target), [])
            if edge.edge_id not in used_current_edge_ids
        ]
        reverse_candidates = current.edge_lookup.get((mapped_target, mapped_source), [])
        if same_direction_candidates:
            best_edge = max(
                same_direction_candidates,
                key=lambda edge: _edge_similarity(reference_edge, edge),
            )
            label_similarity = _edge_label_similarity(reference_edge, best_edge)
            type_similarity = _edge_type_similarity(reference_edge, best_edge)
            edge_similarity = _edge_similarity(reference_edge, best_edge)
            if (
                edge_similarity >= _EDGE_MATCH_THRESHOLD
                and (label_similarity >= 0.4 or not reference_edge.normalized_label)
                and type_similarity >= 0.5
            ):
                matched_reference_edge_count += 1
                used_current_edge_ids.add(best_edge.edge_id)
                continue
            diffs.append(
                {
                    "diff_kind": "edge_label_conflict",
                    "edge_key": _edge_key(reference, reference_edge),
                    "reference_edge": _edge_payload(reference, reference_edge),
                    "current_edge": _edge_payload(current, best_edge),
                }
            )
            continue
        if reverse_candidates:
            diffs.append(
                {
                    "diff_kind": "edge_direction_conflict",
                    "edge_key": _edge_key(reference, reference_edge),
                    "reference_edge": _edge_payload(reference, reference_edge),
                    "current_edge": _edge_payload(
                        current,
                        max(reverse_candidates, key=lambda edge: _edge_similarity(reference_edge, edge)),
                    ),
                }
            )
            continue
        diffs.append(
            {
                "diff_kind": "missing_edge",
                "edge_key": _edge_key(reference, reference_edge),
                "reference_edge": _edge_payload(reference, reference_edge),
            }
        )

    matched_current_edge_count = len(used_current_edge_ids)
    edge_precision = matched_current_edge_count / max(len(current.edges), 1)
    edge_recall = matched_reference_edge_count / max(len(reference.edges), 1)
    return {
        "edge_precision": edge_precision,
        "edge_recall": edge_recall,
        "edge_f1": _f1_score(edge_precision, edge_recall),
        "matched_edge_count": matched_reference_edge_count,
        "diffs": diffs,
    }


def _missing_reference_nodes(
    reference: _ComparableGraph,
    matched_pairs: dict[str, str],
) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    for reference_node in reference.nodes.values():
        if reference_node.node_id in matched_pairs:
            continue
        diffs.append(
            {
                "diff_kind": "missing_node",
                "node_key": _node_key(reference, reference_node),
                "reference_node": _node_payload(reference_node),
            }
        )
    return diffs


def _node_similarity(
    reference_node: _ComparableNode,
    current_node: _ComparableNode,
) -> float:
    label_score = _normalized_text_similarity(
        reference_node.label,
        current_node.label,
    )
    shape_score = _shape_similarity(reference_node.shape, current_node.shape)
    subgraph_score = _subgraph_similarity(reference_node, current_node)
    incoming_score = _multiset_similarity(
        reference_node.incoming_context,
        current_node.incoming_context,
    )
    outgoing_score = _multiset_similarity(
        reference_node.outgoing_context,
        current_node.outgoing_context,
    )
    degree_score = _degree_similarity(reference_node, current_node)
    position_score = _position_similarity(reference_node, current_node)
    return _clamp_score(
        0.35 * label_score
        + 0.10 * shape_score
        + 0.10 * subgraph_score
        + 0.15 * incoming_score
        + 0.15 * outgoing_score
        + 0.10 * degree_score
        + 0.05 * position_score
    )


def _shape_similarity(left: str, right: str) -> float:
    normalized_left = str(left or "unknown").strip().lower() or "unknown"
    normalized_right = str(right or "unknown").strip().lower() or "unknown"
    if normalized_left == normalized_right:
        return 1.0
    if "unknown" in {normalized_left, normalized_right}:
        return 0.5
    return 0.0


def _subgraph_similarity(
    reference_node: _ComparableNode,
    current_node: _ComparableNode,
) -> float:
    if not reference_node.normalized_parent and not current_node.normalized_parent:
        return 1.0
    if not reference_node.normalized_parent or not current_node.normalized_parent:
        return 0.0
    if reference_node.normalized_parent == current_node.normalized_parent:
        return 1.0
    return _normalized_text_similarity(reference_node.parent, current_node.parent)


def _degree_similarity(
    reference_node: _ComparableNode,
    current_node: _ComparableNode,
) -> float:
    maximum = max(
        reference_node.in_degree + reference_node.out_degree,
        current_node.in_degree + current_node.out_degree,
        1,
    )
    difference = abs(reference_node.in_degree - current_node.in_degree) + abs(
        reference_node.out_degree - current_node.out_degree
    )
    return _clamp_score(1.0 - (difference / (2 * maximum)))


def _position_similarity(
    reference_node: _ComparableNode,
    current_node: _ComparableNode,
) -> float:
    role_score = (
        int(reference_node.is_source == current_node.is_source)
        + int(reference_node.is_sink == current_node.is_sink)
    ) / 2
    depth_start_score = _optional_int_similarity(
        reference_node.depth_from_start,
        current_node.depth_from_start,
    )
    depth_end_score = _optional_int_similarity(
        reference_node.depth_to_end,
        current_node.depth_to_end,
    )
    order_score = _clamp_score(
        1.0 - abs(reference_node.order_ratio - current_node.order_ratio)
    )
    return _clamp_score(
        0.35 * role_score
        + 0.25 * depth_start_score
        + 0.20 * depth_end_score
        + 0.20 * order_score
    )


def _path_agreement(
    reference: _ComparableGraph,
    current: _ComparableGraph,
    matched_pairs: dict[str, str],
) -> float:
    matched_reference_ids = list(matched_pairs)
    if not matched_reference_ids:
        return 0.0
    if len(matched_reference_ids) == 1:
        return 1.0

    comparisons = 0
    matched = 0
    for source_id in matched_reference_ids:
        for target_id in matched_reference_ids:
            if source_id == target_id:
                continue
            reference_reachable = target_id in reference.reachability.get(source_id, set())
            current_source_id = matched_pairs[source_id]
            current_target_id = matched_pairs[target_id]
            current_reachable = current_target_id in current.reachability.get(
                current_source_id,
                set(),
            )
            comparisons += 1
            if reference_reachable == current_reachable:
                matched += 1
    if comparisons == 0:
        return 1.0
    return matched / comparisons


def _duplicate_collapse_penalty(
    reference: _ComparableGraph,
    matched_pairs: dict[str, str],
) -> float:
    duplicate_reference_total = sum(
        len(node_ids)
        for node_ids in reference.label_groups.values()
        if len(node_ids) > 1
    )
    if duplicate_reference_total == 0:
        return 0.0

    collapsed_count = 0
    for node_ids in reference.label_groups.values():
        if len(node_ids) <= 1:
            continue
        matched_prediction_ids = {
            matched_pairs[node_id]
            for node_id in node_ids
            if node_id in matched_pairs
        }
        collapsed_count += max(0, len(node_ids) - len(matched_prediction_ids))
    return collapsed_count / duplicate_reference_total


def _node_split_penalty(
    reference: _ComparableGraph,
    current: _ComparableGraph,
    matched_pairs: dict[str, str],
    node_sim_lookup: dict[tuple[str, str], float],
) -> float:
    if not matched_pairs:
        return 0.0
    unmatched_current_ids = {
        node_id
        for node_id in current.nodes
        if node_id not in set(matched_pairs.values())
    }
    split_count = 0
    for current_id in unmatched_current_ids:
        best_score = 0.0
        for reference_id in matched_pairs:
            best_score = max(
                best_score,
                float(node_sim_lookup.get((reference_id, current_id), 0.0)),
            )
        if best_score >= 0.75:
            split_count += 1
    return split_count / max(len(matched_pairs), 1)


def _edge_similarity(
    reference_edge: _ComparableEdge,
    current_edge: _ComparableEdge,
) -> float:
    endpoint_similarity = 1.0
    label_similarity = _edge_label_similarity(reference_edge, current_edge)
    type_similarity = _edge_type_similarity(reference_edge, current_edge)
    direction_similarity = 1.0
    return _clamp_score(
        0.45 * endpoint_similarity
        + 0.25 * label_similarity
        + 0.20 * type_similarity
        + 0.10 * direction_similarity
    )


def _edge_label_similarity(
    reference_edge: _ComparableEdge,
    current_edge: _ComparableEdge,
) -> float:
    if not reference_edge.normalized_label and not current_edge.normalized_label:
        return 1.0
    if not reference_edge.normalized_label or not current_edge.normalized_label:
        return 0.0
    return _normalized_text_similarity(reference_edge.label, current_edge.label)


def _edge_type_similarity(
    reference_edge: _ComparableEdge,
    current_edge: _ComparableEdge,
) -> float:
    return 1.0 if reference_edge.edge_type == current_edge.edge_type else 0.0


def _flowchart_ast_similarity(
    current_graph: dict[str, Any] | None,
    reference_graph: dict[str, Any] | None,
) -> float:
    current = _build_comparable_graph(current_graph)
    reference = _build_comparable_graph(reference_graph)
    if not current.nodes and not reference.nodes:
        return 1.0
    node_count_score = _ratio_similarity(len(current.nodes), len(reference.nodes))
    edge_count_score = _ratio_similarity(len(current.edges), len(reference.edges))
    subgraph_count_score = _ratio_similarity(
        sum(reference.subgraph_groups.values()),
        sum(current.subgraph_groups.values()),
    )
    shape_distribution_score = _counter_similarity(
        Counter(node.shape for node in current.nodes.values()),
        Counter(node.shape for node in reference.nodes.values()),
    )
    return _clamp_score(
        0.35 * node_count_score
        + 0.30 * edge_count_score
        + 0.15 * subgraph_count_score
        + 0.20 * shape_distribution_score
    )


def _diagram_semantic_similarity(graph_metrics: dict[str, Any]) -> float:
    path_agreement = float(graph_metrics.get("path_agreement", 0.0))
    node_f1 = float(graph_metrics.get("node_f1", 0.0))
    edge_f1 = float(graph_metrics.get("edge_f1", 0.0))
    branch_agreement = _clamp_score(0.5 * edge_f1 + 0.5 * path_agreement)
    terminal_agreement = _clamp_score(0.5 * node_f1 + 0.5 * path_agreement)
    return _clamp_score(
        0.55 * path_agreement + 0.25 * branch_agreement + 0.20 * terminal_agreement
    )


def _render_or_visual_similarity(
    current_mermaid: str,
    reference_mermaid: str,
    current_graph: dict[str, Any] | None,
    reference_graph: dict[str, Any] | None,
) -> float:
    current_direction = _mermaid_direction(current_mermaid)
    reference_direction = _mermaid_direction(reference_mermaid)
    direction_score = 0.5
    if current_direction and reference_direction:
        direction_score = 1.0 if current_direction == reference_direction else 0.6

    current = _build_comparable_graph(current_graph)
    reference = _build_comparable_graph(reference_graph)
    size_score = _ratio_similarity(len(current.nodes), len(reference.nodes))
    return _clamp_score(0.6 * direction_score + 0.4 * size_score)


def _node_key(graph: _ComparableGraph, node: _ComparableNode) -> str:
    node_ids = graph.label_groups.get(node.normalized_label, [])
    if node.normalized_label and len(node_ids) == 1:
        return node.normalized_label
    if node.normalized_label:
        return f"{node.normalized_label}:{node.node_id}"
    return f"id:{node.node_id}"


def _edge_key(graph: _ComparableGraph, edge: _ComparableEdge) -> str:
    source_node = graph.nodes[edge.source]
    target_node = graph.nodes[edge.target]
    return "|".join(
        (
            _node_key(graph, source_node),
            edge.normalized_label,
            _node_key(graph, target_node),
        )
    )


def _node_payload(node: _ComparableNode) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "order_index": node.order_index,
        "shape": node.shape,
        "text": node.label,
        "parent": node.parent,
    }


def _edge_payload(
    graph: _ComparableGraph,
    edge: _ComparableEdge,
) -> dict[str, Any]:
    source_node = graph.nodes[edge.source]
    target_node = graph.nodes[edge.target]
    return {
        "source": _node_key(graph, source_node),
        "target": _node_key(graph, target_node),
        "label": edge.label,
        "edge_type": edge.edge_type,
        "source_text": source_node.label,
        "target_text": target_node.label,
    }


def _directed_distance_lookup(
    node_ids: list[str],
    neighbors: dict[str, set[str]],
    seeds: list[str],
) -> dict[str, int | None]:
    if not node_ids:
        return {}
    if not seeds:
        return {
            node_id: index + 1 for index, node_id in enumerate(sorted(node_ids))
        }

    distances: dict[str, int | None] = {node_id: None for node_id in node_ids}
    queue: deque[tuple[str, int]] = deque()
    for seed in sorted(set(seeds)):
        distances[seed] = 0
        queue.append((seed, 0))
    while queue:
        node_id, depth = queue.popleft()
        for neighbor in sorted(neighbors.get(node_id, set())):
            next_depth = depth + 1
            previous_depth = distances.get(neighbor)
            if previous_depth is None or next_depth < previous_depth:
                distances[neighbor] = next_depth
                queue.append((neighbor, next_depth))

    fallback_depth = max((depth for depth in distances.values() if depth is not None), default=0)
    for node_id in node_ids:
        if distances[node_id] is None:
            fallback_depth += 1
            distances[node_id] = fallback_depth
    return distances


def _reachability_lookup(
    node_ids: list[str],
    neighbors: dict[str, set[str]],
) -> dict[str, set[str]]:
    reachability: dict[str, set[str]] = {}
    for node_id in node_ids:
        visited: set[str] = set()
        queue: deque[str] = deque(sorted(neighbors.get(node_id, set())))
        while queue:
            current_node_id = queue.popleft()
            if current_node_id in visited:
                continue
            visited.add(current_node_id)
            for neighbor in sorted(neighbors.get(current_node_id, set())):
                if neighbor not in visited:
                    queue.append(neighbor)
        reachability[node_id] = visited
    return reachability


def _hungarian_max_assignment(matrix: list[list[float]]) -> list[int]:
    if not matrix:
        return []
    row_count = len(matrix)
    column_count = len(matrix[0])
    maximum = max((max(row) for row in matrix if row), default=0.0)
    cost = [
        [maximum - float(value) for value in row]
        for row in matrix
    ]
    u = [0.0] * (row_count + 1)
    v = [0.0] * (column_count + 1)
    p = [0] * (column_count + 1)
    way = [0] * (column_count + 1)

    for row_index in range(1, row_count + 1):
        p[0] = row_index
        column0 = 0
        minimum = [float("inf")] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = float("inf")
            column1 = 0
            for column_index in range(1, column_count + 1):
                if used[column_index]:
                    continue
                current_cost = cost[row0 - 1][column_index - 1] - u[row0] - v[column_index]
                if current_cost < minimum[column_index]:
                    minimum[column_index] = current_cost
                    way[column_index] = column0
                if minimum[column_index] < delta:
                    delta = minimum[column_index]
                    column1 = column_index
            for column_index in range(column_count + 1):
                if used[column_index]:
                    u[p[column_index]] += delta
                    v[column_index] -= delta
                else:
                    minimum[column_index] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break

    assignment = [-1] * row_count
    for column_index in range(1, column_count + 1):
        if p[column_index] != 0:
            assignment[p[column_index] - 1] = column_index - 1
    return assignment


def _multiset_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    overlap = sum((left & right).values())
    total = sum(left.values()) + sum(right.values())
    if total <= 0:
        return 0.0
    return (2 * overlap) / total


def _counter_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    overlap = sum((left & right).values())
    total = max(sum(left.values()), sum(right.values()), 1)
    return overlap / total


def _ratio_similarity(left: int, right: int) -> float:
    maximum = max(left, right, 1)
    return _clamp_score(1.0 - (abs(left - right) / maximum))


def _optional_int_similarity(left: int | None, right: int | None) -> float:
    if left is None and right is None:
        return 1.0
    if left is None or right is None:
        return 0.0
    return _ratio_similarity(left, right)


def _normalized_text_similarity(left: str, right: str) -> float:
    normalized_left = _normalize_graph_text(left)
    normalized_right = _normalize_graph_text(right)
    if not normalized_left and not normalized_right:
        return 1.0
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left == normalized_right:
        return 1.0
    left_bigrams = _char_bigrams(normalized_left)
    right_bigrams = _char_bigrams(normalized_right)
    union = left_bigrams | right_bigrams
    if not union:
        return 0.0
    return len(left_bigrams & right_bigrams) / len(union)


def _char_bigrams(text: str) -> set[str]:
    if len(text) < 2:
        return {text} if text else set()
    return {text[index : index + 2] for index in range(len(text) - 1)}


def _mermaid_direction(mermaid_text: str) -> str:
    match = _FLOWCHART_HEADER_RE.search(mermaid_text)
    if match is None:
        return ""
    return str(match.group(2) or "").strip().upper()


def _f1_score(precision: float, recall: float) -> float:
    if precision <= 0 or recall <= 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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
    class_names = [
        str(item).strip().lower()
        for item in node_payload.get("class_names", [])
        if str(item).strip()
    ] if isinstance(node_payload.get("class_names"), list) else []
    if bool(node_payload.get("hidden")) or "hidden" in class_names:
        return True
    if not node_id:
        return False
    if not text.strip():
        return True
    if shape not in {"", "unknown"}:
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
