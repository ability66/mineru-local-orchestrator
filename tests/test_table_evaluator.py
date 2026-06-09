from __future__ import annotations

from src.pipeline.table_evaluator import evaluate_markdown_table


BASE_MARKDOWN_TABLE = (
    "| 指标 | 数值 | 公式 |\n"
    "| --- | --- | --- |\n"
    "| 增长率 | 12% | $\\dfrac{a}{b}$ |"
)


def test_evaluate_markdown_table_identical_score_is_one() -> None:
    metrics = evaluate_markdown_table(BASE_MARKDOWN_TABLE, BASE_MARKDOWN_TABLE)

    assert metrics["table_score"] == 1.0
    assert metrics["grid_structure_sim"] == 1.0


def test_evaluate_markdown_table_ignores_whitespace_noise() -> None:
    variant = (
        "| 指标 | 数值 | 公式 |\n"
        "| --- | --- | --- |\n"
        "|  增长率  | 12% |   $\\dfrac{a}{b}$   |"
    )

    metrics = evaluate_markdown_table(BASE_MARKDOWN_TABLE, variant)

    assert metrics["table_score"] > 0.95
    assert metrics["grid_structure_sim"] == 1.0


def test_evaluate_markdown_table_penalizes_structure_difference() -> None:
    right = (
        "| 地区 | Q1 |\n"
        "| --- | --- |\n"
        "| 华东 | 10 |"
    )

    metrics = evaluate_markdown_table(BASE_MARKDOWN_TABLE, right)

    assert metrics["grid_structure_sim"] < 0.9
    assert metrics["diagnostics"]["wrong_headers"]


def test_evaluate_markdown_table_penalizes_wrong_numeric_value() -> None:
    right = BASE_MARKDOWN_TABLE.replace("12%", "18%")

    metrics = evaluate_markdown_table(BASE_MARKDOWN_TABLE, right)

    assert metrics["numeric_fidelity"] < 1.0
    assert metrics["diagnostics"]["wrong_numeric"]


def test_evaluate_markdown_table_penalizes_missing_unit() -> None:
    right = BASE_MARKDOWN_TABLE.replace("12%", "12")

    metrics = evaluate_markdown_table(BASE_MARKDOWN_TABLE, right)

    assert metrics["numeric_fidelity"] < 1.0
    assert metrics["diagnostics"]["wrong_numeric"]


def test_evaluate_markdown_table_normalizes_equivalent_latex_forms() -> None:
    right = BASE_MARKDOWN_TABLE.replace(r"\dfrac{a}{b}", r"\frac{a}{b}")

    metrics = evaluate_markdown_table(BASE_MARKDOWN_TABLE, right)

    assert metrics["formula_sim"] > 0.95


def test_evaluate_markdown_table_penalizes_wrong_formula_content() -> None:
    right = BASE_MARKDOWN_TABLE.replace(r"\dfrac{a}{b}", r"x^3")

    metrics = evaluate_markdown_table(BASE_MARKDOWN_TABLE, right)

    assert metrics["formula_sim"] < 0.6
    assert metrics["diagnostics"]["wrong_formula"]
