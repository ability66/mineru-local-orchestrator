from __future__ import annotations

from src.pipeline.flowchart_utils import (
    diff_flowchart_graphs,
    flowchart_graph_from_mermaid,
    looks_like_mermaid,
    normalize_mermaid_text,
    score_flowchart_graph_similarity,
    score_mermaid_similarity,
)


def test_normalize_mermaid_text_sanitizes_html_breaks_in_node_and_edge_labels() -> None:
    raw_mermaid = """```mermaid
flowchart TD
    Resectable[无转移，肿瘤可切除或<br/>临界可切除] --> LiverMRI[肝脏MRI<br/>或PET-CT]
    ChestAbdCT -->|发现囊性病变<br/>或CT无法确定| CT
```"""

    normalized = normalize_mermaid_text(raw_mermaid)

    assert (
        'Resectable["无转移，肿瘤可切除或<br/>临界可切除"] --> '
        'LiverMRI["肝脏MRI<br/>或PET-CT"]'
    ) in normalized
    assert "ChestAbdCT -->|发现囊性病变 / 或CT无法确定| CT" in normalized
    assert looks_like_mermaid(raw_mermaid)
    assert flowchart_graph_from_mermaid(raw_mermaid) is not None


def test_looks_like_mermaid_rejects_plain_text_and_single_tokens() -> None:
    assert not looks_like_mermaid("4541982082")
    assert not looks_like_mermaid("Red hammer and sickle symbol on white background (no text or numbers)")
    assert not looks_like_mermaid("A")
    assert not looks_like_mermaid("N001")
    assert looks_like_mermaid("A-->B")


def test_flowchart_graph_from_mermaid_preserves_visible_text_for_quoted_nodes() -> None:
    graph = flowchart_graph_from_mermaid(
        'flowchart TD\nRisk["家族史和/或高危因素"] -->|有| Genetic["遗传咨询"]\n'
        'Genetic --> CT_Q["是否可以进行增强CT?"]'
    )

    assert graph is not None
    node_texts = {node["text"] for node in graph["nodes"]}
    assert "遗传咨询" in node_texts
    assert "是否可以进行增强CT?" in node_texts


def test_diff_flowchart_graphs_ignores_ct_node_formatting_only_differences() -> None:
    current = """flowchart TD
Question["是否可以进行增强CT?"] -->|是| E["CT（多相对比增强，包括动脉晚期和门静脉期）"]"""
    reference = """flowchart TD
Decision1{"是否可以进行增强CT?"} -->|是| CT["CT<br/>多相对比增强,包括动脉晚期和门静脉期"]"""

    diffs = diff_flowchart_graphs(
        flowchart_graph_from_mermaid(current),
        flowchart_graph_from_mermaid(reference),
    )

    assert diffs == []


def test_diff_flowchart_graphs_collapses_split_placeholder_nodes() -> None:
    current = """flowchart TD
Start["A"] --> Other["Other"]
Start --> Local["Local"]"""
    reference = """flowchart TD
StartAlias["A"] --> Split
Split --> OtherAlias["Other"]
Split --> LocalAlias["Local"]"""

    diffs = diff_flowchart_graphs(
        flowchart_graph_from_mermaid(current),
        flowchart_graph_from_mermaid(reference),
    )

    assert diffs == []


def test_diff_flowchart_graphs_ignores_hidden_helper_nodes() -> None:
    current = """flowchart TD
CT["CT"] --> Other["其他发现或鉴别诊断"]
CT --> Resectable["无转移，肿瘤可切除或临界可切除"]"""
    reference = """flowchart TD
CTAlias["CT"] --> H[]:::hidden
H --> OtherAlias["其他发现或鉴别诊断"]
H --> ResectableAlias["无转移，肿瘤可切除或临界可切除"]
classDef hidden display:none;"""

    diffs = diff_flowchart_graphs(
        flowchart_graph_from_mermaid(current),
        flowchart_graph_from_mermaid(reference),
    )

    assert diffs == []


def test_diff_flowchart_graphs_detects_duplicate_label_collapse_by_context() -> None:
    current = """flowchart TD
Start1["发起"] --> Review["审批"]
Start2["复核"] --> Review
Review --> Pass["通过"]
Review --> Reject["拒绝"]"""
    reference = """flowchart TD
Start1["发起"] --> Review1["审批"]
Review1 --> Pass["通过"]
Start2["复核"] --> Review2["审批"]
Review2 --> Reject["拒绝"]"""

    diffs = diff_flowchart_graphs(
        flowchart_graph_from_mermaid(current),
        flowchart_graph_from_mermaid(reference),
    )
    metrics = score_flowchart_graph_similarity(
        flowchart_graph_from_mermaid(current),
        flowchart_graph_from_mermaid(reference),
    )

    assert any(diff["diff_kind"] == "missing_node" for diff in diffs)
    assert metrics["duplicate_collapse_penalty"] > 0
    assert metrics["graph_structure_sim"] < 1.0


def test_score_flowchart_graph_similarity_penalizes_duplicate_label_splitting() -> None:
    current = """flowchart TD
Start["开始"] --> Review1["审批"]
Start --> Review2["审批"]
Review1 --> Pass["通过"]
Review2 --> Pass"""
    reference = """flowchart TD
Start["开始"] --> Review["审批"]
Review --> Pass["通过"]"""

    metrics = score_flowchart_graph_similarity(
        flowchart_graph_from_mermaid(current),
        flowchart_graph_from_mermaid(reference),
    )

    assert metrics["node_split_penalty"] > 0
    assert metrics["graph_structure_sim"] < 1.0


def test_score_mermaid_similarity_prefers_structure_over_label_identity() -> None:
    current = """flowchart TD
Start1["发起"] --> Review["审批"]
Start2["复核"] --> Review
Review --> Pass["通过"]
Review --> Reject["拒绝"]"""
    reference = """flowchart TD
Start1["发起"] --> Review1["审批"]
Review1 --> Pass["通过"]
Start2["复核"] --> Review2["审批"]
Review2 --> Reject["拒绝"]"""

    metrics = score_mermaid_similarity(current, reference)

    assert metrics["syntax_valid"] == 1
    assert metrics["graph_structure_sim"] < 1.0
    assert metrics["mermaid_score"] < 1.0
