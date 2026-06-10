from __future__ import annotations

from difflib import SequenceMatcher
from itertools import combinations
from typing import Any

from src.pipeline.table_utils import (
    MarkdownTableCell,
    extract_numeric_values_and_units,
    normalize_cell_text,
    normalize_latex_formula,
    parse_markdown_table,
)


def evaluate_markdown_table(table_a: str, table_b: str) -> dict[str, Any]:
    ir_a = parse_markdown_table(table_a)
    ir_b = parse_markdown_table(table_b)
    parse_valid = bool(ir_a.parse_valid and ir_b.parse_valid)
    has_table = bool(ir_a.has_table and ir_b.has_table)

    diagnostics: dict[str, Any] = {
        "table_a": {
            "row_count": ir_a.row_count,
            "col_count": ir_a.col_count,
            "parse_valid": ir_a.parse_valid,
            "has_table": ir_a.has_table,
            "diagnostics": ir_a.diagnostics,
        },
        "table_b": {
            "row_count": ir_b.row_count,
            "col_count": ir_b.col_count,
            "parse_valid": ir_b.parse_valid,
            "has_table": ir_b.has_table,
            "diagnostics": ir_b.diagnostics,
        },
        "matched_cells": [],
        "missing_cells": [],
        "extra_cells": [],
        "wrong_headers": [],
        "wrong_numeric": [],
        "wrong_formula": [],
    }
    if not parse_valid or not has_table:
        return {
            "parse_valid": parse_valid,
            "has_table": has_table,
            "grid_structure_sim": 0.0,
            "cell_content_sim": 0.0,
            "header_semantic_sim": 0.0,
            "numeric_fidelity": 0.0,
            "formula_sim": 0.0,
            "omission_rate": 1.0 if ir_a.has_table else 0.0,
            "hallucination_rate": 1.0 if ir_b.has_table else 0.0,
            "table_score": 0.0,
            "diagnostics": diagnostics,
        }

    cell_map_a = {(cell.row_index, cell.col_index): cell for cell in ir_a.cells}
    cell_map_b = {(cell.row_index, cell.col_index): cell for cell in ir_b.cells}
    matched_keys = sorted(set(cell_map_a) & set(cell_map_b))
    missing_keys = sorted(set(cell_map_a) - set(cell_map_b))
    extra_keys = sorted(set(cell_map_b) - set(cell_map_a))

    for key in missing_keys:
        diagnostics["missing_cells"].append(_cell_summary(cell_map_a[key]))
    for key in extra_keys:
        diagnostics["extra_cells"].append(_cell_summary(cell_map_b[key]))
    for key in matched_keys:
        diagnostics["matched_cells"].append(
            {
                "anchor": {"row": key[0], "col": key[1]},
                "a": _cell_summary(cell_map_a[key]),
                "b": _cell_summary(cell_map_b[key]),
            }
        )

    grid_structure_sim = _grid_structure_similarity(ir_a=ir_a, ir_b=ir_b)
    cell_content_sim = _cell_content_similarity(
        cell_map_a=cell_map_a,
        cell_map_b=cell_map_b,
        matched_keys=matched_keys,
    )
    header_semantic_sim = _header_semantic_similarity(
        cell_map_a=cell_map_a,
        cell_map_b=cell_map_b,
        diagnostics=diagnostics,
    )
    numeric_fidelity = _numeric_fidelity(
        cell_map_a=cell_map_a,
        cell_map_b=cell_map_b,
        matched_keys=matched_keys,
        diagnostics=diagnostics,
    )
    formula_sim = _formula_similarity(
        cell_map_a=cell_map_a,
        cell_map_b=cell_map_b,
        matched_keys=matched_keys,
        diagnostics=diagnostics,
    )
    omission_rate = round(len(missing_keys) / max(len(cell_map_a), 1), 4)
    hallucination_rate = round(len(extra_keys) / max(len(cell_map_b), 1), 4)

    table_score = (
        0.40 * grid_structure_sim
        + 0.30 * cell_content_sim
        + 0.10 * header_semantic_sim
        + 0.10 * numeric_fidelity
        + 0.10 * formula_sim
    )

    return {
        "parse_valid": parse_valid,
        "has_table": has_table,
        "grid_structure_sim": round(grid_structure_sim, 4),
        "cell_content_sim": round(cell_content_sim, 4),
        "header_semantic_sim": round(header_semantic_sim, 4),
        "numeric_fidelity": round(numeric_fidelity, 4),
        "formula_sim": round(formula_sim, 4),
        "omission_rate": omission_rate,
        "hallucination_rate": hallucination_rate,
        "table_score": round(table_score, 4),
        "diagnostics": diagnostics,
    }


def analyze_table_candidate_consensus(
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    normalized_candidates = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict)
        and str(candidate.get("table_format", "") or "").strip().lower() == "markdown"
        and str(candidate.get("table_text", "") or "").strip()
    ]
    if len(normalized_candidates) < 2:
        return None

    pairwise: list[dict[str, Any]] = []
    pair_scores: dict[tuple[str, str], float] = {}
    parse_failures = 0
    severe_conflicts: list[str] = []
    for left, right in combinations(normalized_candidates, 2):
        left_role = str(left.get("role", "") or "").strip().lower()
        right_role = str(right.get("role", "") or "").strip().lower()
        metrics = evaluate_markdown_table(
            str(left.get("table_text", "") or ""),
            str(right.get("table_text", "") or ""),
        )
        score = float(metrics.get("table_score", 0.0))
        pair_scores[(left_role, right_role)] = score
        pair_scores[(right_role, left_role)] = score
        pairwise.append(
            {
                "left": left_role,
                "right": right_role,
                "score": score,
                "metrics": metrics,
            }
        )
        if not metrics["parse_valid"] or not metrics["has_table"]:
            parse_failures += 1
        if (
            metrics["grid_structure_sim"] < 0.55
            or metrics["omission_rate"] > 0.35
            or metrics["hallucination_rate"] > 0.35
            or metrics["formula_sim"] < 0.55
        ):
            severe_conflicts.append(f"{left_role}-{right_role}")

    if parse_failures:
        return {
            "fallback": True,
            "reason": "table_parse_failure",
            "pairwise": pairwise,
            "candidate_roles": [
                str(candidate.get("role", "") or "").strip().lower()
                for candidate in normalized_candidates
            ],
        }

    role_to_candidate = {
        str(candidate.get("role", "") or "").strip().lower(): candidate
        for candidate in normalized_candidates
    }
    role_scores: dict[str, list[float]] = {role: [] for role in role_to_candidate}
    for item in pairwise:
        role_scores[item["left"]].append(float(item["score"]))
        role_scores[item["right"]].append(float(item["score"]))

    high_threshold = 0.88
    low_threshold = 0.68
    all_high = bool(pairwise) and min(item["score"] for item in pairwise) >= high_threshold
    stable_consensus = False
    consensus_kind = "none"
    consensus_cluster: list[str] = []
    reference_role: str | None = None
    requires_qwen = False
    review_reasons: list[str] = []

    if all_high and not severe_conflicts:
        stable_consensus = True
        consensus_kind = "all"
        consensus_cluster = sorted(role_to_candidate)
        non_mineru_roles = [role for role in consensus_cluster if role != "mineru"]
        reference_role = _best_reference_role(non_mineru_roles, role_scores) or (
            non_mineru_roles[0] if non_mineru_roles else None
        )
    else:
        strongest_pair = max(pairwise, key=lambda item: item["score"], default=None)
        if strongest_pair is not None:
            pair_roles = {strongest_pair["left"], strongest_pair["right"]}
            other_scores = [
                item["score"]
                for item in pairwise
                if pair_roles != {item["left"], item["right"]}
            ]
            if (
                "mineru" in pair_roles
                and strongest_pair["score"] >= high_threshold
                and (not other_scores or max(other_scores) <= low_threshold)
                and f"{strongest_pair['left']}-{strongest_pair['right']}" not in severe_conflicts
                and f"{strongest_pair['right']}-{strongest_pair['left']}" not in severe_conflicts
            ):
                stable_consensus = True
                consensus_kind = "pair"
                consensus_cluster = sorted(pair_roles)
                reference_role = next(
                    (role for role in consensus_cluster if role != "mineru"),
                    None,
                )

    if not stable_consensus:
        requires_qwen = True
        if severe_conflicts:
            review_reasons.append("severe_structure_or_formula_conflict")
        if not pairwise:
            review_reasons.append("pairwise_similarity_matrix_empty")
        else:
            review_reasons.append("no_stable_table_consensus")
        reference_role = _best_reference_role(
            [role for role in role_to_candidate if role != "mineru"],
            role_scores,
        )

    matrix = {
        role: {
            other_role: (
                1.0
                if role == other_role
                else round(pair_scores.get((role, other_role), 0.0), 4)
            )
            for other_role in role_to_candidate
        }
        for role in role_to_candidate
    }

    return {
        "fallback": False,
        "stable_consensus": stable_consensus,
        "consensus_kind": consensus_kind,
        "consensus_cluster": consensus_cluster,
        "reference_role": reference_role,
        "requires_qwen": requires_qwen,
        "review_reasons": review_reasons,
        "pairwise": pairwise,
        "matrix": matrix,
        "candidate_roles": sorted(role_to_candidate),
        "severe_conflicts": severe_conflicts,
    }


def _best_reference_role(roles: list[str], role_scores: dict[str, list[float]]) -> str | None:
    if not roles:
        return None
    scored_roles = sorted(
        roles,
        key=lambda role: (
            -(
                sum(role_scores.get(role, []))
                / max(len(role_scores.get(role, [])), 1)
            ),
            0 if role == "qwen" else 1,
            role,
        ),
    )
    return scored_roles[0]


def _grid_structure_similarity(ir_a: Any, ir_b: Any) -> float:
    row_sim = _ratio_similarity(ir_a.row_count, ir_b.row_count)
    col_sim = _ratio_similarity(ir_a.col_count, ir_b.col_count)
    positions_a = {(cell.row_index, cell.col_index) for cell in ir_a.cells}
    positions_b = {(cell.row_index, cell.col_index) for cell in ir_b.cells}
    position_sim = _jaccard(positions_a, positions_b)
    return _clamp(0.3 * row_sim + 0.3 * col_sim + 0.4 * position_sim)


def _cell_content_similarity(
    cell_map_a: dict[tuple[int, int], MarkdownTableCell],
    cell_map_b: dict[tuple[int, int], MarkdownTableCell],
    matched_keys: list[tuple[int, int]],
) -> float:
    if not matched_keys:
        return 0.0
    scores = [
        _text_similarity(
            cell_map_a[key].normalized_text,
            cell_map_b[key].normalized_text,
        )
        for key in matched_keys
    ]
    coverage = len(matched_keys) / max(max(len(cell_map_a), len(cell_map_b)), 1)
    return _clamp((sum(scores) / len(scores)) * coverage)


def _header_semantic_similarity(
    cell_map_a: dict[tuple[int, int], MarkdownTableCell],
    cell_map_b: dict[tuple[int, int], MarkdownTableCell],
    diagnostics: dict[str, Any],
) -> float:
    header_keys = sorted(
        {
            key
            for key, cell in cell_map_a.items()
            if cell.is_header
        }
        | {
            key
            for key, cell in cell_map_b.items()
            if cell.is_header
        }
    )
    if not header_keys:
        return 1.0

    scores: list[float] = []
    for key in header_keys:
        cell_a = cell_map_a.get(key)
        cell_b = cell_map_b.get(key)
        if cell_a is None or cell_b is None:
            diagnostics["wrong_headers"].append(
                {
                    "anchor": {"row": key[0], "col": key[1]},
                    "a": _cell_summary(cell_a),
                    "b": _cell_summary(cell_b),
                }
            )
            scores.append(0.0)
            continue
        score = _text_similarity(cell_a.text, cell_b.text)
        scores.append(score)
        if score < 1.0:
            diagnostics["wrong_headers"].append(
                {
                    "anchor": {"row": key[0], "col": key[1]},
                    "a": _cell_summary(cell_a),
                    "b": _cell_summary(cell_b),
                }
            )

    header_set_a = {
        (cell.col_index, normalize_cell_text(cell.text))
        for cell in cell_map_a.values()
        if cell.is_header
    }
    header_set_b = {
        (cell.col_index, normalize_cell_text(cell.text))
        for cell in cell_map_b.values()
        if cell.is_header
    }
    return _clamp(0.7 * (sum(scores) / len(scores)) + 0.3 * _jaccard(header_set_a, header_set_b))


def _numeric_fidelity(
    cell_map_a: dict[tuple[int, int], MarkdownTableCell],
    cell_map_b: dict[tuple[int, int], MarkdownTableCell],
    matched_keys: list[tuple[int, int]],
    diagnostics: dict[str, Any],
) -> float:
    cell_scores: list[float] = []
    relevant_pairs = 0
    for key in matched_keys:
        values_a = extract_numeric_values_and_units(cell_map_a[key].text)
        values_b = extract_numeric_values_and_units(cell_map_b[key].text)
        if not values_a and not values_b:
            continue
        relevant_pairs += 1
        score = _numeric_list_similarity(values_a, values_b)
        cell_scores.append(score)
        if score < 1.0:
            diagnostics["wrong_numeric"].append(
                {
                    "anchor": {"row": key[0], "col": key[1]},
                    "a": values_a,
                    "b": values_b,
                }
            )
    if relevant_pairs == 0:
        return 1.0
    return _clamp(sum(cell_scores) / len(cell_scores))


def _formula_similarity(
    cell_map_a: dict[tuple[int, int], MarkdownTableCell],
    cell_map_b: dict[tuple[int, int], MarkdownTableCell],
    matched_keys: list[tuple[int, int]],
    diagnostics: dict[str, Any],
) -> float:
    cell_scores: list[float] = []
    relevant_pairs = 0
    for key in matched_keys:
        formulas_a = [
            normalize_latex_formula(item)
            for item in cell_map_a[key].formulas
            if normalize_latex_formula(item)
        ]
        formulas_b = [
            normalize_latex_formula(item)
            for item in cell_map_b[key].formulas
            if normalize_latex_formula(item)
        ]
        if not formulas_a and not formulas_b:
            continue
        relevant_pairs += 1
        score = _list_similarity(formulas_a, formulas_b)
        cell_scores.append(score)
        if score < 1.0:
            diagnostics["wrong_formula"].append(
                {
                    "anchor": {"row": key[0], "col": key[1]},
                    "a": formulas_a,
                    "b": formulas_b,
                }
            )
    if relevant_pairs == 0:
        return 1.0
    return _clamp(sum(cell_scores) / len(cell_scores))


def _numeric_list_similarity(
    values_a: list[tuple[float, str | None]],
    values_b: list[tuple[float, str | None]],
) -> float:
    if not values_a and not values_b:
        return 1.0
    if not values_a or not values_b:
        return 0.0

    pair_scores: list[float] = []
    for left, right in zip(values_a, values_b):
        number_match = 1.0 if abs(left[0] - right[0]) <= 1e-9 else 0.0
        unit_match = 1.0 if (left[1] or "") == (right[1] or "") else 0.0
        pair_scores.append(0.7 * number_match + 0.3 * unit_match)
    count_penalty = min(len(values_a), len(values_b)) / max(len(values_a), len(values_b))
    return _clamp((sum(pair_scores) / len(pair_scores)) * count_penalty)


def _list_similarity(values_a: list[str], values_b: list[str]) -> float:
    if not values_a and not values_b:
        return 1.0
    if not values_a or not values_b:
        return 0.0
    if values_a == values_b:
        return 1.0
    return _jaccard(set(values_a), set(values_b))


def _text_similarity(left: str, right: str) -> float:
    normalized_left = normalize_cell_text(left)
    normalized_right = normalize_cell_text(right)
    if not normalized_left and not normalized_right:
        return 1.0
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left == normalized_right:
        return 1.0
    left_tokens = set(normalized_left.split())
    right_tokens = set(normalized_right.split())
    token_score = _jaccard(left_tokens, right_tokens)
    sequence_score = SequenceMatcher(None, normalized_left, normalized_right).ratio()
    return _clamp(max(sequence_score, (0.6 * token_score) + (0.4 * sequence_score)))


def _cell_summary(cell: MarkdownTableCell | None) -> dict[str, Any] | None:
    if cell is None:
        return None
    return {
        "row_index": cell.row_index,
        "col_index": cell.col_index,
        "text": cell.text,
        "normalized_text": cell.normalized_text,
        "is_header": cell.is_header,
        "formulas": list(cell.formulas),
    }


def _jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _ratio_similarity(left: int, right: int) -> float:
    if left == 0 and right == 0:
        return 1.0
    if left == 0 or right == 0:
        return 0.0
    return min(left, right) / max(left, right)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
