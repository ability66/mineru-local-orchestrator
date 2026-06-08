from __future__ import annotations

from src.pipeline.flowchart_utils import (
    flowchart_graph_from_mermaid,
    looks_like_mermaid,
    normalize_mermaid_text,
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
