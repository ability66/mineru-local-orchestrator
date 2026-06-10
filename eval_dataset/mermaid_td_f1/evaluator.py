from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from html import unescape
import re
from typing import Any

MATCH_THRESHOLD = 0.25
DIRECTIVE_PREFIXES = (
    "flowchart",
    "graph",
    "subgraph",
    "end",
    "style",
    "classdef",
    "class",
    "linkstyle",
    "click",
)
NODE_WEIGHTS = {
    "start": 0.5,
    "end": 0.5,
    "process": 1.0,
    "decision": 1.5,
    "subroutine": 1.2,
    "unknown": 1.0,
}
POS = {"是", "yes", "true", "通过", "成功", "pass", "approved", "success"}
NEG = {
    "否",
    "no",
    "false",
    "不通过",
    "失败",
    "fail",
    "failed",
    "reject",
    "rejected",
    "驳回",
}
ELSE = {"否则", "else", "other", "default"}
BR_RE = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)
ID_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")


@dataclass
class Node:
    id: str
    text: str
    type: str
    order: int
    is_virtual: bool = False


@dataclass
class Edge:
    source: str
    target: str
    text: str
    kind: str
    order: int


@dataclass
class Graph:
    nodes: dict[str, Node]
    edges: list[Edge]
    adjacency: dict[str, list[Edge]]
    root: str | None


@dataclass
class ParsedNode:
    node_id: str
    text: str
    node_type: str
    end: int
    explicit_label: bool
    is_virtual: bool = False


@dataclass
class Reachability:
    traversal_nodes: set[str]
    semantic_nodes: set[str]
    edges: set[int]


def strip_mermaid_fence(text: str) -> str:
    stripped = str(text or "").strip()
    if not stripped:
        return ""
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    if stripped.lower().startswith("mermaid\n"):
        stripped = stripped.split("\n", 1)[1].strip()
    return stripped


def normalize_text(text: str) -> str:
    value = strip_mermaid_fence(text).lower()
    value = unescape(value)
    value = value.replace("\\n", " ").replace("\\t", " ")
    value = value.replace('\\"', '"').replace("\\'", "'")
    value = BR_RE.sub(" ", value)
    value = value.replace("nbsp", " ")
    chars: list[str] = []
    for char in value:
        if char.isascii() and char.isalnum():
            chars.append(char)
            continue
        if _is_cjk(char):
            chars.append(char)
            continue
        chars.append(" ")
    return " ".join("".join(chars).split())


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    tokens: list[str] = []
    buffer: list[str] = []
    for char in normalized:
        if char == " ":
            if buffer:
                tokens.append("".join(buffer))
                buffer = []
            continue
        if char.isascii() and char.isalnum():
            buffer.append(char)
            continue
        if _is_cjk(char):
            if buffer:
                tokens.append("".join(buffer))
                buffer = []
            tokens.append(char)
            continue
        if buffer:
            tokens.append("".join(buffer))
            buffer = []
    if buffer:
        tokens.append("".join(buffer))
    return tokens


def token_f1(a: str, b: str) -> float:
    a_tokens = tokenize(a)
    b_tokens = tokenize(b)
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    a_counts = Counter(a_tokens)
    b_counts = Counter(b_tokens)
    overlap = sum(min(count, b_counts[token]) for token, count in a_counts.items())
    precision = overlap / len(a_tokens)
    recall = overlap / len(b_tokens)
    return safe_f1(precision, recall)


def evaluate_flowchart(
    ground_truth_mermaid: str,
    predicted_mermaid: str,
) -> dict[str, float]:
    result = evaluate_mermaid_flowchart(
        pred_mermaid=predicted_mermaid,
        gold_mermaid=ground_truth_mermaid,
    )
    return {
        "structure_f1": result["structure_f1"],
        "semantic_f1": result["semantic_f1"],
    }


def evaluate_mermaid_flowchart(pred_mermaid: str, gold_mermaid: str) -> dict[str, Any]:
    debug: dict[str, Any] = {
        "pred_root": None,
        "gold_root": None,
        "matched_nodes": [],
        "matched_edges": [],
        "errors": [],
    }
    try:
        pred_graph = normalize_virtual_nodes(parse_mermaid_flowchart(pred_mermaid))
    except Exception as exc:
        pred_graph = None
        debug["errors"].append(f"pred_parse_error: {exc}")
    try:
        gold_graph = normalize_virtual_nodes(parse_mermaid_flowchart(gold_mermaid))
    except Exception as exc:
        gold_graph = None
        debug["errors"].append(f"gold_parse_error: {exc}")

    parse_valid = bool(pred_graph and gold_graph)
    if not parse_valid:
        return _empty_result(parse_valid=False, debug=debug, penalty=0.0)

    assert pred_graph is not None
    assert gold_graph is not None
    debug["pred_root"] = pred_graph.root
    debug["gold_root"] = gold_graph.root

    if pred_graph.root is None:
        debug["errors"].append("pred_root_missing")
    if gold_graph.root is None:
        debug["errors"].append("gold_root_missing")
    if pred_graph.root is None or gold_graph.root is None:
        return _empty_result(parse_valid=False, debug=debug, penalty=0.0)

    pred_reachability = compute_reachability(pred_graph)
    gold_reachability = compute_reachability(gold_graph)

    counted_node_pairs: set[tuple[str, str]] = set()
    counted_edge_pairs: set[tuple[int, int]] = set()
    visited_recursion_pairs: set[tuple[str, str]] = set()
    matched_node_pairs: list[tuple[str, str]] = []
    matched_edge_pairs: list[tuple[Edge, Edge]] = []

    def align_pair(pred_id: str, gold_id: str, path_pairs: set[tuple[str, str]]) -> None:
        pair = (pred_id, gold_id)
        if pair in path_pairs:
            return
        if pred_id not in pred_graph.nodes or gold_id not in gold_graph.nodes:
            return
        if pred_id not in pred_reachability.traversal_nodes:
            return
        if gold_id not in gold_reachability.traversal_nodes:
            return

        pred_node = pred_graph.nodes[pred_id]
        gold_node = gold_graph.nodes[gold_id]
        if (
            not pred_node.is_virtual
            and not gold_node.is_virtual
            and pair not in counted_node_pairs
        ):
            counted_node_pairs.add(pair)
            matched_node_pairs.append(pair)

        if pair in visited_recursion_pairs:
            return
        visited_recursion_pairs.add(pair)

        next_path_pairs = set(path_pairs)
        next_path_pairs.add(pair)
        pred_out_edges = [
            edge
            for edge in pred_graph.adjacency.get(pred_id, [])
            if edge.order in pred_reachability.edges
        ]
        gold_out_edges = [
            edge
            for edge in gold_graph.adjacency.get(gold_id, [])
            if edge.order in gold_reachability.edges
        ]
        matched_pairs = match_outgoing_edges(
            pred_out_edges=pred_out_edges,
            gold_out_edges=gold_out_edges,
            pred_graph=pred_graph,
            gold_graph=gold_graph,
        )
        for pred_edge, gold_edge in matched_pairs:
            edge_pair = (pred_edge.order, gold_edge.order)
            if edge_pair not in counted_edge_pairs:
                counted_edge_pairs.add(edge_pair)
                matched_edge_pairs.append((pred_edge, gold_edge))
            align_pair(pred_edge.target, gold_edge.target, next_path_pairs)

    align_pair(pred_graph.root, gold_graph.root, set())

    structure_precision_num = 0.0
    structure_recall_num = 0.0
    for pred_id, gold_id in matched_node_pairs:
        pred_node = pred_graph.nodes[pred_id]
        gold_node = gold_graph.nodes[gold_id]
        type_sim = 1.0 if pred_node.type == gold_node.type else 0.0
        structure_precision_num += node_weight(pred_node) * type_sim
        structure_recall_num += node_weight(gold_node) * type_sim
    for pred_edge, gold_edge in matched_edge_pairs:
        structure_precision_num += edge_weight(pred_edge, pred_graph)
        structure_recall_num += edge_weight(gold_edge, gold_graph)

    pred_struct_units = sum(
        node_weight(pred_graph.nodes[node_id])
        for node_id in pred_reachability.semantic_nodes
    ) + sum(
        edge_weight(edge, pred_graph)
        for edge in pred_graph.edges
        if edge.order in pred_reachability.edges
    )
    gold_struct_units = sum(
        node_weight(gold_graph.nodes[node_id])
        for node_id in gold_reachability.semantic_nodes
    ) + sum(
        edge_weight(edge, gold_graph)
        for edge in gold_graph.edges
        if edge.order in gold_reachability.edges
    )

    structure_precision = safe_div(structure_precision_num, pred_struct_units)
    structure_recall = safe_div(structure_recall_num, gold_struct_units)
    structure_f1 = safe_f1(structure_precision, structure_recall)

    node_text_precision_num = 0.0
    node_text_recall_num = 0.0
    for pred_id, gold_id in matched_node_pairs:
        pred_node = pred_graph.nodes[pred_id]
        gold_node = gold_graph.nodes[gold_id]
        sim = sim_node(pred_node.text, gold_node.text)
        node_text_precision_num += node_weight(pred_node) * sim
        node_text_recall_num += node_weight(gold_node) * sim
    node_text_precision_den = sum(
        node_weight(pred_graph.nodes[node_id])
        for node_id in pred_reachability.semantic_nodes
    )
    node_text_recall_den = sum(
        node_weight(gold_graph.nodes[node_id])
        for node_id in gold_reachability.semantic_nodes
    )
    node_text_precision = safe_div(node_text_precision_num, node_text_precision_den)
    node_text_recall = safe_div(node_text_recall_num, node_text_recall_den)
    node_text_f1 = safe_f1(node_text_precision, node_text_recall)

    edge_text_precision_num = 0.0
    edge_text_recall_num = 0.0
    binding_precision_num = 0.0
    binding_recall_num = 0.0
    for pred_edge, gold_edge in matched_edge_pairs:
        edge_sim = sim_edge(pred_edge.text, gold_edge.text)
        bind_sim = sim_bind(
            pred_edge=pred_edge,
            gold_edge=gold_edge,
            pred_graph=pred_graph,
            gold_graph=gold_graph,
        )
        edge_text_precision_num += edge_weight(pred_edge, pred_graph) * edge_sim
        edge_text_recall_num += edge_weight(gold_edge, gold_graph) * edge_sim
        binding_precision_num += edge_weight(pred_edge, pred_graph) * bind_sim
        binding_recall_num += edge_weight(gold_edge, gold_graph) * bind_sim

    edge_text_precision_den = sum(
        edge_weight(edge, pred_graph)
        for edge in pred_graph.edges
        if edge.order in pred_reachability.edges
    )
    edge_text_recall_den = sum(
        edge_weight(edge, gold_graph)
        for edge in gold_graph.edges
        if edge.order in gold_reachability.edges
    )
    edge_text_precision = safe_div(edge_text_precision_num, edge_text_precision_den)
    edge_text_recall = safe_div(edge_text_recall_num, edge_text_recall_den)
    edge_text_f1 = safe_f1(edge_text_precision, edge_text_recall)

    binding_precision = safe_div(binding_precision_num, edge_text_precision_den)
    binding_recall = safe_div(binding_recall_num, edge_text_recall_den)
    binding_f1 = safe_f1(binding_precision, binding_recall)

    semantic_f1 = clamp01(
        0.45 * node_text_f1 + 0.25 * edge_text_f1 + 0.30 * binding_f1
    )
    penalty = compute_penalty(
        pred_graph=pred_graph,
        gold_graph=gold_graph,
        matched_node_pairs=matched_node_pairs,
        debug=debug,
    )
    final_td_f1 = clamp01((0.45 * structure_f1 + 0.55 * semantic_f1) * penalty)

    debug["matched_nodes"] = [
        {
            "pred_id": pred_id,
            "gold_id": gold_id,
            "pred_text": pred_graph.nodes[pred_id].text,
            "gold_text": gold_graph.nodes[gold_id].text,
        }
        for pred_id, gold_id in matched_node_pairs
    ]
    debug["matched_edges"] = [
        {
            "pred": edge_to_dict(pred_edge),
            "gold": edge_to_dict(gold_edge),
        }
        for pred_edge, gold_edge in matched_edge_pairs
    ]

    return {
        "parse_valid": True,
        "structure_f1": structure_f1,
        "node_text_f1": node_text_f1,
        "edge_text_f1": edge_text_f1,
        "binding_f1": binding_f1,
        "semantic_f1": semantic_f1,
        "penalty": penalty,
        "final_td_f1": final_td_f1,
        "debug": debug,
    }


def parse_mermaid_flowchart(mermaid: str) -> Graph:
    source = strip_mermaid_fence(mermaid)
    if not source.strip():
        raise ValueError("empty_mermaid")

    nodes: dict[str, Node] = {}
    edges: list[Edge] = []
    node_order = 0
    edge_order = 0

    for raw_line in source.splitlines():
        line = strip_inline_comment(raw_line)
        if not line.strip():
            continue
        for statement in split_statements(line):
            stripped = statement.strip()
            if not stripped:
                continue
            if is_directive_line(stripped):
                continue
            parsed_edges = parse_edge_chain(stripped)
            if parsed_edges:
                for parsed_node in parsed_edges["nodes"]:
                    if parsed_node.node_id not in nodes:
                        node_order += 1
                    upsert_node(nodes, parsed_node, node_order)
                for source_id, target_id, text, kind in parsed_edges["edges"]:
                    edge_order += 1
                    edges.append(
                        Edge(
                            source=source_id,
                            target=target_id,
                            text=text,
                            kind=kind,
                            order=edge_order,
                        )
                    )
                continue

            standalone_node = parse_node_at(stripped, 0)
            if standalone_node is not None and standalone_node.end == len(stripped):
                if standalone_node.node_id not in nodes:
                    node_order += 1
                upsert_node(nodes, standalone_node, node_order)

    if not nodes:
        raise ValueError("no_nodes_parsed")
    return rebuild_graph(nodes=nodes, edges=edges)


def normalize_virtual_nodes(graph: Graph) -> Graph:
    nodes = {node_id: clone_node(node) for node_id, node in graph.nodes.items()}
    edges = [clone_edge(edge) for edge in graph.edges]
    changed = True
    while changed:
        changed = False
        adjacency = build_adjacency(edges)
        incoming = build_incoming(edges)
        for node in sorted(nodes.values(), key=lambda item: item.order):
            if not is_contractible_virtual_node(
                node=node,
                incoming_edges=incoming.get(node.id, []),
                outgoing_edges=adjacency.get(node.id, []),
            ):
                continue
            edges = contract_virtual_node(
                node_id=node.id,
                nodes=nodes,
                edges=edges,
                incoming_edges=incoming.get(node.id, []),
                outgoing_edges=adjacency.get(node.id, []),
            )
            nodes.pop(node.id, None)
            changed = True
            break
    return rebuild_graph(nodes=nodes, edges=edges)


def compute_reachability(graph: Graph) -> Reachability:
    if graph.root is None or graph.root not in graph.nodes:
        return Reachability(traversal_nodes=set(), semantic_nodes=set(), edges=set())
    traversal_nodes: set[str] = set()
    semantic_nodes: set[str] = set()
    reachable_edges: set[int] = set()
    queue: deque[str] = deque([graph.root])
    while queue:
        node_id = queue.popleft()
        if node_id in traversal_nodes:
            continue
        traversal_nodes.add(node_id)
        node = graph.nodes[node_id]
        if not node.is_virtual:
            semantic_nodes.add(node_id)
        for edge in graph.adjacency.get(node_id, []):
            reachable_edges.add(edge.order)
            if edge.target not in traversal_nodes:
                queue.append(edge.target)
    return Reachability(
        traversal_nodes=traversal_nodes,
        semantic_nodes=semantic_nodes,
        edges=reachable_edges,
    )


def build_adjacency(edges: list[Edge]) -> dict[str, list[Edge]]:
    adjacency: dict[str, list[Edge]] = defaultdict(list)
    for edge in edges:
        adjacency[edge.source].append(edge)
    return dict(adjacency)


def build_incoming(edges: list[Edge]) -> dict[str, list[Edge]]:
    incoming: dict[str, list[Edge]] = defaultdict(list)
    for edge in edges:
        incoming[edge.target].append(edge)
    return dict(incoming)


def clone_node(node: Node) -> Node:
    return Node(
        id=node.id,
        text=node.text,
        type=node.type,
        order=node.order,
        is_virtual=node.is_virtual,
    )


def clone_edge(edge: Edge) -> Edge:
    return Edge(
        source=edge.source,
        target=edge.target,
        text=edge.text,
        kind=edge.kind,
        order=edge.order,
    )


def rebuild_graph(nodes: dict[str, Node], edges: list[Edge]) -> Graph:
    graph_nodes = {node_id: clone_node(node) for node_id, node in nodes.items()}
    rebuilt_edges: list[Edge] = []
    adjacency: dict[str, list[Edge]] = defaultdict(list)
    filtered_edges = [
        edge
        for edge in edges
        if edge.source in graph_nodes and edge.target in graph_nodes
    ]
    filtered_edges.sort(
        key=lambda edge: (
            edge.order,
            graph_nodes[edge.source].order,
            graph_nodes[edge.target].order,
            edge.source,
            edge.target,
            edge.text,
            edge.kind,
        )
    )
    for index, edge in enumerate(filtered_edges, start=1):
        rebuilt = Edge(
            source=edge.source,
            target=edge.target,
            text=edge.text,
            kind=edge.kind,
            order=index,
        )
        rebuilt_edges.append(rebuilt)
        adjacency[rebuilt.source].append(rebuilt)
    return Graph(
        nodes=graph_nodes,
        edges=rebuilt_edges,
        adjacency=dict(adjacency),
        root=select_root(graph_nodes),
    )


def select_root(nodes: dict[str, Node]) -> str | None:
    candidates = [node for node in nodes.values() if not node.is_virtual]
    if not candidates:
        return None
    return min(candidates, key=lambda node: node.order).id


def is_contractible_virtual_node(
    node: Node,
    incoming_edges: list[Edge],
    outgoing_edges: list[Edge],
) -> bool:
    if not node.is_virtual:
        return False
    if node.type not in {"process", "unknown"}:
        return False
    incoming_has_text = any(edge.text.strip() for edge in incoming_edges)
    outgoing_has_text = any(edge.text.strip() for edge in outgoing_edges)
    return not (incoming_has_text and outgoing_has_text)


def contract_virtual_node(
    node_id: str,
    nodes: dict[str, Node],
    edges: list[Edge],
    incoming_edges: list[Edge],
    outgoing_edges: list[Edge],
) -> list[Edge]:
    preserved_edges = [
        clone_edge(edge)
        for edge in edges
        if edge.source != node_id and edge.target != node_id
    ]
    existing_keys = {
        (edge.source, edge.target, edge.text, edge.kind)
        for edge in preserved_edges
    }
    new_edges: list[Edge] = []
    if incoming_edges and outgoing_edges:
        for incoming_edge in incoming_edges:
            for outgoing_edge in outgoing_edges:
                merged_text = merge_virtual_edge_text(
                    incoming_edge.text,
                    outgoing_edge.text,
                )
                if merged_text is None:
                    continue
                merged_kind = merge_edge_kind(
                    incoming_edge.kind,
                    outgoing_edge.kind,
                )
                key = (
                    incoming_edge.source,
                    outgoing_edge.target,
                    merged_text,
                    merged_kind,
                )
                if key in existing_keys:
                    continue
                existing_keys.add(key)
                new_edges.append(
                    Edge(
                        source=incoming_edge.source,
                        target=outgoing_edge.target,
                        text=merged_text,
                        kind=merged_kind,
                        order=min(incoming_edge.order, outgoing_edge.order),
                    )
                )
    return preserved_edges + new_edges


def merge_virtual_edge_text(incoming_text: str, outgoing_text: str) -> str | None:
    left = incoming_text.strip()
    right = outgoing_text.strip()
    if left and right:
        return None
    return left or right


def merge_edge_kind(incoming_kind: str, outgoing_kind: str) -> str:
    if incoming_kind == outgoing_kind:
        return incoming_kind
    if "thick" in {incoming_kind, outgoing_kind}:
        return "thick"
    if "dotted" in {incoming_kind, outgoing_kind}:
        return "dotted"
    return "solid"


def node_weight(node: Node) -> float:
    return NODE_WEIGHTS.get(node.type, NODE_WEIGHTS["unknown"])


def edge_weight(edge: Edge, graph: Graph) -> float:
    source_node = graph.nodes.get(
        edge.source,
        Node(edge.source, edge.source, "unknown", 0, False),
    )
    if source_node.type == "decision":
        return 1.5
    if edge.text.strip():
        return 0.8
    return 0.5


def sim_node(pred_text: str, gold_text: str) -> float:
    return token_f1(pred_text, gold_text)


def sim_edge(pred_text: str, gold_text: str) -> float:
    pred_polarity = classify_polarity(pred_text)
    gold_polarity = classify_polarity(gold_text)
    if {pred_polarity, gold_polarity} == {"pos", "neg"}:
        return 0.0
    if pred_polarity is not None and pred_polarity == gold_polarity:
        return 1.0
    return token_f1(pred_text, gold_text)


def sim_bind(pred_edge: Edge, gold_edge: Edge, pred_graph: Graph, gold_graph: Graph) -> float:
    pred_polarity = classify_polarity(pred_edge.text)
    gold_polarity = classify_polarity(gold_edge.text)
    if {pred_polarity, gold_polarity} == {"pos", "neg"}:
        return 0.0
    pred_target = pred_graph.nodes[pred_edge.target]
    gold_target = gold_graph.nodes[gold_edge.target]
    return clamp01(
        0.5 * sim_edge(pred_edge.text, gold_edge.text)
        + 0.5 * sim_node(pred_target.text, gold_target.text)
    )


def match_outgoing_edges(
    pred_out_edges: list[Edge],
    gold_out_edges: list[Edge],
    pred_graph: Graph,
    gold_graph: Graph,
) -> list[tuple[Edge, Edge]]:
    if not pred_out_edges or not gold_out_edges:
        return []
    scores = [
        [
            edge_candidate_score(
                pred_edge=pred_edge,
                gold_edge=gold_edge,
                pred_graph=pred_graph,
                gold_graph=gold_graph,
            )
            for gold_edge in gold_out_edges
        ]
        for pred_edge in pred_out_edges
    ]
    if len(pred_out_edges) <= 6 and len(gold_out_edges) <= 6:
        return exact_edge_matching(
            pred_out_edges=pred_out_edges,
            gold_out_edges=gold_out_edges,
            scores=scores,
        )
    return greedy_edge_matching(
        pred_out_edges=pred_out_edges,
        gold_out_edges=gold_out_edges,
        scores=scores,
    )


def edge_candidate_score(pred_edge: Edge, gold_edge: Edge, pred_graph: Graph, gold_graph: Graph) -> float:
    pred_target = pred_graph.nodes[pred_edge.target]
    gold_target = gold_graph.nodes[gold_edge.target]
    target_type_sim = 1.0 if pred_target.type == gold_target.type else 0.0
    target_text_sim = sim_node(pred_target.text, gold_target.text)
    edge_text_sim = sim_edge(pred_edge.text, gold_edge.text)
    pred_degree = len(pred_graph.adjacency.get(pred_target.id, []))
    gold_degree = len(gold_graph.adjacency.get(gold_target.id, []))
    degree_sim = 1.0 - abs(pred_degree - gold_degree) / max(pred_degree, gold_degree, 1)
    return clamp01(
        0.35 * target_type_sim
        + 0.30 * target_text_sim
        + 0.20 * edge_text_sim
        + 0.15 * degree_sim
    )


def exact_edge_matching(
    pred_out_edges: list[Edge],
    gold_out_edges: list[Edge],
    scores: list[list[float]],
) -> list[tuple[Edge, Edge]]:
    pred_count = len(pred_out_edges)
    gold_count = len(gold_out_edges)

    @lru_cache(maxsize=None)
    def dp(pred_index: int, used_mask: int) -> tuple[float, tuple[tuple[int, int], ...]]:
        if pred_index >= pred_count:
            return 0.0, ()
        best_score, best_pairs = dp(pred_index + 1, used_mask)
        for gold_index in range(gold_count):
            if used_mask & (1 << gold_index):
                continue
            score = scores[pred_index][gold_index]
            if score < MATCH_THRESHOLD:
                continue
            tail_score, tail_pairs = dp(pred_index + 1, used_mask | (1 << gold_index))
            total_score = score + tail_score
            if total_score > best_score:
                best_score = total_score
                best_pairs = ((pred_index, gold_index),) + tail_pairs
        return best_score, best_pairs

    _, best_pairs = dp(0, 0)
    return [
        (pred_out_edges[pred_index], gold_out_edges[gold_index])
        for pred_index, gold_index in best_pairs
    ]


def greedy_edge_matching(
    pred_out_edges: list[Edge],
    gold_out_edges: list[Edge],
    scores: list[list[float]],
) -> list[tuple[Edge, Edge]]:
    candidates: list[tuple[float, int, int]] = []
    for pred_index, row in enumerate(scores):
        for gold_index, score in enumerate(row):
            if score >= MATCH_THRESHOLD:
                candidates.append((score, pred_index, gold_index))
    candidates.sort(reverse=True)
    used_pred: set[int] = set()
    used_gold: set[int] = set()
    matches: list[tuple[Edge, Edge]] = []
    for _, pred_index, gold_index in candidates:
        if pred_index in used_pred or gold_index in used_gold:
            continue
        used_pred.add(pred_index)
        used_gold.add(gold_index)
        matches.append((pred_out_edges[pred_index], gold_out_edges[gold_index]))
    return matches


def compute_penalty(
    pred_graph: Graph,
    gold_graph: Graph,
    matched_node_pairs: list[tuple[str, str]],
    debug: dict[str, Any],
) -> float:
    penalty = 1.0
    if pred_graph.root is None or gold_graph.root is None:
        return 0.0
    pred_root = pred_graph.nodes[pred_graph.root]
    gold_root = gold_graph.nodes[gold_graph.root]
    if pred_root.type != gold_root.type:
        penalty *= 0.7
    if detect_polarity_reversal(pred_graph, gold_graph, matched_node_pairs):
        penalty *= 0.5
        if "polarity_reversal" not in debug["errors"]:
            debug["errors"].append("polarity_reversal")
    return clamp01(penalty)


def detect_polarity_reversal(
    pred_graph: Graph,
    gold_graph: Graph,
    matched_node_pairs: list[tuple[str, str]],
) -> bool:
    for pred_id, gold_id in matched_node_pairs:
        pred_node = pred_graph.nodes[pred_id]
        gold_node = gold_graph.nodes[gold_id]
        if pred_node.type != "decision" or gold_node.type != "decision":
            continue
        pred_edges = polarity_edge_map(pred_graph.adjacency.get(pred_id, []))
        gold_edges = polarity_edge_map(gold_graph.adjacency.get(gold_id, []))
        if "pos" not in pred_edges or "neg" not in pred_edges:
            continue
        if "pos" not in gold_edges or "neg" not in gold_edges:
            continue
        pred_pos_target = pred_graph.nodes[pred_edges["pos"].target]
        pred_neg_target = pred_graph.nodes[pred_edges["neg"].target]
        gold_pos_target = gold_graph.nodes[gold_edges["pos"].target]
        gold_neg_target = gold_graph.nodes[gold_edges["neg"].target]
        if sim_node(pred_pos_target.text, gold_neg_target.text) > sim_node(
            pred_pos_target.text,
            gold_pos_target.text,
        ) and sim_node(pred_neg_target.text, gold_pos_target.text) > sim_node(
            pred_neg_target.text,
            gold_neg_target.text,
        ):
            return True
    return False


def polarity_edge_map(edges: list[Edge]) -> dict[str, Edge]:
    mapping: dict[str, Edge] = {}
    for edge in edges:
        polarity = classify_polarity(edge.text)
        if polarity in {"pos", "neg", "else"} and polarity not in mapping:
            mapping[polarity] = edge
    return mapping


def classify_polarity(text: str) -> str | None:
    normalized = "".join(tokenize(text))
    if not normalized:
        return None
    if normalized in NORMALIZED_POS:
        return "pos"
    if normalized in NORMALIZED_NEG:
        return "neg"
    if normalized in NORMALIZED_ELSE:
        return "else"
    return None


def parse_edge_chain(statement: str) -> dict[str, Any] | None:
    first_node = parse_node_at(statement, 0)
    if first_node is None:
        return None
    parsed_nodes = [first_node]
    parsed_edges: list[tuple[str, str, str, str]] = []
    cursor = first_node.end
    current_node = first_node
    while True:
        connector = parse_connector(statement, cursor)
        if connector is None:
            break
        cursor = connector["end"]
        next_node = parse_node_at(statement, cursor)
        if next_node is None:
            break
        parsed_nodes.append(next_node)
        parsed_edges.append(
            (
                current_node.node_id,
                next_node.node_id,
                connector["text"],
                connector["kind"],
            )
        )
        current_node = next_node
        cursor = next_node.end
    if not parsed_edges:
        return None
    return {"nodes": parsed_nodes, "edges": parsed_edges}


def parse_connector(statement: str, start: int) -> dict[str, Any] | None:
    cursor = skip_spaces(statement, start)
    remainder = statement[cursor:]
    direct_connectors = {
        "-->": "solid",
        "-.->": "dotted",
        "==>": "thick",
    }
    for token, kind in direct_connectors.items():
        if remainder.startswith(token):
            cursor += len(token)
            cursor = skip_spaces(statement, cursor)
            label = ""
            if cursor < len(statement) and statement[cursor] == "|":
                end = statement.find("|", cursor + 1)
                if end != -1:
                    label = statement[cursor + 1 : end].strip()
                    cursor = end + 1
            return {"kind": kind, "text": label, "end": cursor}
    if remainder.startswith("--"):
        end_index = remainder.find("-->", 2)
        if end_index != -1:
            label = remainder[2:end_index].strip()
            return {
                "kind": "solid",
                "text": label,
                "end": cursor + end_index + 3,
            }
    if remainder.startswith("-."):
        end_index = remainder.find(".->", 2)
        if end_index != -1:
            label = remainder[2:end_index].strip()
            return {
                "kind": "dotted",
                "text": label,
                "end": cursor + end_index + 3,
            }
    return None


def parse_node_at(statement: str, start: int) -> ParsedNode | None:
    cursor = skip_spaces(statement, start)
    match = ID_RE.match(statement, cursor)
    if match is None:
        return None
    node_id = match.group(0)
    cursor = match.end()
    cursor = skip_spaces(statement, cursor)

    text = node_id
    node_type = "process"
    explicit_label = False
    if statement.startswith("[[", cursor):
        end = statement.find("]]", cursor + 2)
        if end == -1:
            return None
        text = clean_node_text(statement[cursor + 2 : end])
        node_type = "subroutine"
        cursor = end + 2
        explicit_label = True
    elif statement.startswith("((", cursor):
        end = statement.find("))", cursor + 2)
        if end == -1:
            return None
        text = clean_node_text(statement[cursor + 2 : end])
        node_type = infer_round_node_type(text)
        cursor = end + 2
        explicit_label = True
    elif statement.startswith("([", cursor):
        end = statement.find("])", cursor + 2)
        if end == -1:
            return None
        text = clean_node_text(statement[cursor + 2 : end])
        node_type = infer_round_node_type(text)
        cursor = end + 2
        explicit_label = True
    elif cursor < len(statement) and statement[cursor] == "(":
        end = statement.find(")", cursor + 1)
        if end == -1:
            return None
        text = clean_node_text(statement[cursor + 1 : end])
        node_type = infer_round_node_type(text)
        cursor = end + 1
        explicit_label = True
    elif cursor < len(statement) and statement[cursor] == "{":
        end = statement.find("}", cursor + 1)
        if end == -1:
            return None
        text = clean_node_text(statement[cursor + 1 : end])
        node_type = "decision"
        cursor = end + 1
        explicit_label = True
    elif cursor < len(statement) and statement[cursor] == "[":
        end = statement.find("]", cursor + 1)
        if end == -1:
            return None
        text = clean_node_text(statement[cursor + 1 : end])
        node_type = infer_bracket_node_type(text)
        cursor = end + 1
        explicit_label = True
    else:
        node_type = infer_plain_node_type(text)

    cursor = skip_spaces(statement, cursor)
    if statement.startswith(":::", cursor):
        cursor += 3
        class_match = re.match(r"[A-Za-z0-9_-]+", statement[cursor:])
        if class_match:
            cursor += class_match.end()
    is_virtual = explicit_label and is_virtual_label(text, node_type)
    return ParsedNode(
        node_id=node_id,
        text=text,
        node_type=node_type,
        end=cursor,
        explicit_label=explicit_label,
        is_virtual=is_virtual,
    )


def is_virtual_label(text: str, node_type: str) -> bool:
    if node_type not in {"process", "unknown"}:
        return False
    return clean_node_text(text) == ""


def clean_node_text(text: str) -> str:
    value = str(text or "").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value.strip()


def infer_plain_node_type(text: str) -> str:
    normalized = normalize_text(text)
    joined = normalized.replace(" ", "")
    if any(keyword in joined for keyword in ("开始", "start", "begin")):
        return "start"
    if any(keyword in joined for keyword in ("结束", "end", "finish", "done")):
        return "end"
    return "process"


def infer_round_node_type(text: str) -> str:
    return infer_plain_node_type(text)


def infer_bracket_node_type(text: str) -> str:
    return infer_plain_node_type(text)


def split_statements(line: str) -> list[str]:
    statements: list[str] = []
    buffer: list[str] = []
    quote: str | None = None
    nesting = 0
    for char in line:
        if quote is not None:
            buffer.append(char)
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            buffer.append(char)
            continue
        if char in "[{(":
            nesting += 1
            buffer.append(char)
            continue
        if char in "]})":
            nesting = max(0, nesting - 1)
            buffer.append(char)
            continue
        if char == ";" and nesting == 0:
            statements.append("".join(buffer))
            buffer = []
            continue
        buffer.append(char)
    statements.append("".join(buffer))
    return statements


def strip_inline_comment(line: str) -> str:
    quote: str | None = None
    nesting = 0
    index = 0
    while index < len(line):
        char = line[index]
        if quote is not None:
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        if char in "[{(":
            nesting += 1
            index += 1
            continue
        if char in "]})":
            nesting = max(0, nesting - 1)
            index += 1
            continue
        if char == "%" and index + 1 < len(line) and line[index + 1] == "%" and nesting == 0:
            return line[:index]
        index += 1
    return line


def is_directive_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    return any(
        lowered == prefix or lowered.startswith(f"{prefix} ")
        for prefix in DIRECTIVE_PREFIXES
    )


def upsert_node(nodes: dict[str, Node], parsed_node: ParsedNode, order: int) -> None:
    existing = nodes.get(parsed_node.node_id)
    if existing is None:
        nodes[parsed_node.node_id] = Node(
            id=parsed_node.node_id,
            text=parsed_node.text,
            type=parsed_node.node_type,
            order=order,
            is_virtual=parsed_node.is_virtual,
        )
        return

    if parsed_node.explicit_label:
        existing.text = parsed_node.text
        existing.is_virtual = parsed_node.is_virtual
    elif should_replace_text(existing.text, parsed_node.text, existing.id):
        existing.text = parsed_node.text

    if type_priority(parsed_node.node_type) >= type_priority(existing.type):
        existing.type = parsed_node.node_type


def should_replace_text(existing_text: str, new_text: str, node_id: str) -> bool:
    if not new_text.strip():
        return False
    if existing_text.strip() in {"", node_id} and new_text.strip() != node_id:
        return True
    if len(new_text.strip()) > len(existing_text.strip()) and existing_text.strip() == node_id:
        return True
    return False


def type_priority(node_type: str) -> int:
    priorities = {
        "unknown": 0,
        "process": 1,
        "start": 2,
        "end": 2,
        "subroutine": 3,
        "decision": 4,
    }
    return priorities.get(node_type, 0)


def skip_spaces(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return clamp01(numerator / denominator)


def safe_f1(precision: float, recall: float) -> float:
    if precision <= 0 or recall <= 0:
        return 0.0
    return clamp01((2 * precision * recall) / (precision + recall))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return 0x4E00 <= code <= 0x9FFF


def edge_to_dict(edge: Edge) -> dict[str, Any]:
    return {
        "source": edge.source,
        "target": edge.target,
        "text": edge.text,
        "kind": edge.kind,
        "order": edge.order,
    }


def _empty_result(parse_valid: bool, debug: dict[str, Any], penalty: float) -> dict[str, Any]:
    return {
        "parse_valid": parse_valid,
        "structure_f1": 0.0,
        "node_text_f1": 0.0,
        "edge_text_f1": 0.0,
        "binding_f1": 0.0,
        "semantic_f1": 0.0,
        "penalty": penalty,
        "final_td_f1": 0.0,
        "debug": debug,
    }


NORMALIZED_POS = {"".join(tokenize(value)) for value in POS}
NORMALIZED_NEG = {"".join(tokenize(value)) for value in NEG}
NORMALIZED_ELSE = {"".join(tokenize(value)) for value in ELSE}


__all__ = [
    "Node",
    "Edge",
    "Graph",
    "normalize_text",
    "tokenize",
    "token_f1",
    "parse_mermaid_flowchart",
    "normalize_virtual_nodes",
    "evaluate_mermaid_flowchart",
    "evaluate_flowchart",
]
