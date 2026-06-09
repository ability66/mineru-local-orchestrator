from __future__ import annotations

from src.pipeline.table_evaluator import evaluate_html_table


BASE_HTML_TABLE = (
    "<table><thead><tr><th>指标</th><th>数值</th><th>公式</th></tr></thead>"
    "<tbody><tr><td>增长率</td><td>12%</td><td>$\\dfrac{a}{b}$</td></tr></tbody></table>"
)


def test_evaluate_html_table_identical_score_is_near_one() -> None:
    metrics = evaluate_html_table(BASE_HTML_TABLE, BASE_HTML_TABLE)

    assert metrics["html_table_score"] == 1.0
    assert metrics["grid_structure_sim"] == 1.0


def test_evaluate_html_table_ignores_whitespace_and_implicit_tbody_noise() -> None:
    variant = (
        "<table>\n"
        "  <tbody><tr><th>指标</th><th>数值</th><th>公式</th></tr>\n"
        "  <tr><td>增长率</td><td>12%</td><td>$\\dfrac{a}{b}$</td></tr></tbody>\n"
        "</table>"
    )

    metrics = evaluate_html_table(BASE_HTML_TABLE, variant)

    assert metrics["html_table_score"] > 0.95
    assert metrics["grid_structure_sim"] > 0.95


def test_evaluate_html_table_penalizes_wrong_rowspan_or_colspan() -> None:
    left = (
        "<table><tr><th rowspan='2'>地区</th><th>Q1</th></tr>"
        "<tr><td>10</td></tr></table>"
    )
    right = (
        "<table><tr><th>地区</th><th>Q1</th></tr>"
        "<tr><td>华东</td><td>10</td></tr></table>"
    )

    metrics = evaluate_html_table(left, right)

    assert metrics["grid_structure_sim"] < 0.9
    assert metrics["diagnostics"]["wrong_spans"]


def test_evaluate_html_table_penalizes_header_tag_confusion() -> None:
    right = BASE_HTML_TABLE.replace("<th>指标</th>", "<td>指标</td>").replace(
        "<th>数值</th>",
        "<td>数值</td>",
    ).replace("<th>公式</th>", "<td>公式</td>")

    metrics = evaluate_html_table(BASE_HTML_TABLE, right)

    assert metrics["header_semantic_sim"] < 1.0
    assert metrics["diagnostics"]["wrong_headers"]


def test_evaluate_html_table_penalizes_wrong_numeric_value() -> None:
    right = BASE_HTML_TABLE.replace("12%", "18%")

    metrics = evaluate_html_table(BASE_HTML_TABLE, right)

    assert metrics["numeric_fidelity"] < 1.0
    assert metrics["diagnostics"]["wrong_numeric"]


def test_evaluate_html_table_penalizes_missing_unit() -> None:
    right = BASE_HTML_TABLE.replace("12%", "12")

    metrics = evaluate_html_table(BASE_HTML_TABLE, right)

    assert metrics["numeric_fidelity"] < 1.0
    assert metrics["diagnostics"]["wrong_numeric"]


def test_evaluate_html_table_normalizes_equivalent_latex_forms() -> None:
    right = BASE_HTML_TABLE.replace(r"\dfrac{a}{b}", r"\frac{a}{b}")

    metrics = evaluate_html_table(BASE_HTML_TABLE, right)

    assert metrics["formula_sim"] > 0.95


def test_evaluate_html_table_penalizes_wrong_formula_content() -> None:
    right = BASE_HTML_TABLE.replace(r"\dfrac{a}{b}", r"x^3")

    metrics = evaluate_html_table(BASE_HTML_TABLE, right)

    assert metrics["formula_sim"] < 0.6
    assert metrics["diagnostics"]["wrong_formula"]
