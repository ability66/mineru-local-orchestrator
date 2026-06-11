from __future__ import annotations

import json
from pathlib import Path

from eval_dataset.chart_td_f1.evaluator import (
    evaluate_chart_table,
    evaluate_from_record,
    extract_chart_tables_from_record,
)


BASE_TABLE = """| year | revenue | profit |
| --- | ---: | ---: |
| 2023 | 100 | 20 |
| 2024 | 120 | 30 |"""


def test_identical_markdown_table_scores_one() -> None:
    result = evaluate_chart_table(BASE_TABLE, BASE_TABLE, tolerance="strict")

    assert result["parse_success"] == 1.0
    assert result["triple_precision"] == 1.0
    assert result["triple_recall"] == 1.0
    assert result["triple_f1"] == 1.0
    assert result["triple_iou"] == 1.0
    assert result["exact_match"] == 1.0
    assert result["avg_numeric_error"] == 0.0
    assert result["map_strict"] == 1.0
    assert result["map_slight"] == 1.0
    assert result["map_high"] == 1.0


def test_row_order_swap_still_scores_full() -> None:
    prediction = """| year | revenue | profit |
| --- | ---: | ---: |
| 2024 | 120 | 30 |
| 2023 | 100 | 20 |"""

    result = evaluate_chart_table(prediction, BASE_TABLE, tolerance="strict")

    assert result["triple_f1"] == 1.0
    assert result["exact_match"] == 1.0


def test_column_order_swap_still_scores_full() -> None:
    prediction = """| year | profit | revenue |
| --- | ---: | ---: |
| 2023 | 20 | 100 |
| 2024 | 30 | 120 |"""

    result = evaluate_chart_table(prediction, BASE_TABLE, tolerance="strict")

    assert result["triple_f1"] == 1.0
    assert result["exact_match"] == 1.0


def test_transposed_table_scores_full() -> None:
    prediction = """| metric | 2023 | 2024 |
| --- | ---: | ---: |
| revenue | 100 | 120 |
| profit | 20 | 30 |"""

    result = evaluate_chart_table(prediction, BASE_TABLE, tolerance="strict", allow_transpose=True)

    assert result["triple_f1"] == 1.0
    assert result["exact_match"] == 1.0


def test_numeric_tolerance_levels_change_the_score() -> None:
    prediction = """| year | revenue | profit |
| --- | ---: | ---: |
| 2023 | 100 | 20 |
| 2024 | 127 | 30 |"""

    strict_result = evaluate_chart_table(prediction, BASE_TABLE, tolerance="strict")
    slight_result = evaluate_chart_table(prediction, BASE_TABLE, tolerance="slight")
    high_result = evaluate_chart_table(prediction, BASE_TABLE, tolerance="high")

    assert strict_result["triple_f1"] < 1.0
    assert slight_result["triple_f1"] == 1.0
    assert high_result["triple_f1"] == 1.0
    assert strict_result["map_strict"] < 1.0
    assert high_result["map_high"] == 1.0


def test_binding_error_is_penalized() -> None:
    prediction = """| year | revenue | profit |
| --- | ---: | ---: |
| 2023 | 20 | 100 |
| 2024 | 30 | 120 |"""

    result = evaluate_chart_table(prediction, BASE_TABLE, tolerance="strict")

    assert result["triple_f1"] == 0.0
    assert result["exact_match"] == 0.0


def test_missing_row_reduces_recall() -> None:
    prediction = """| year | revenue | profit |
| --- | ---: | ---: |
| 2023 | 100 | 20 |"""

    result = evaluate_chart_table(prediction, BASE_TABLE, tolerance="strict")

    assert result["triple_precision"] == 1.0
    assert result["triple_recall"] == 0.5


def test_extra_row_reduces_precision() -> None:
    prediction = """| year | revenue | profit |
| --- | ---: | ---: |
| 2023 | 100 | 20 |
| 2024 | 120 | 30 |
| 2025 | 150 | 40 |"""

    result = evaluate_chart_table(prediction, BASE_TABLE, tolerance="strict")

    assert result["triple_recall"] == 1.0
    assert result["triple_precision"] == 0.6667


def test_unparseable_numeric_value_affects_metrics() -> None:
    prediction = """| year | revenue | profit |
| --- | ---: | ---: |
| 2023 | one hundred | 20 |
| 2024 | 120 | 30 |"""

    result = evaluate_chart_table(prediction, BASE_TABLE, tolerance="strict")

    assert result["triple_f1"] < 1.0
    assert result["avg_numeric_error"] > 0.0


def test_invalid_markdown_or_empty_output_does_not_crash() -> None:
    result = evaluate_chart_table("", BASE_TABLE, tolerance="strict")

    assert result["parse_success"] == 0.0
    assert result["triple_f1"] == 0.0
    assert result["errors"]


def test_numeric_normalization_handles_percent_currency_and_thousands() -> None:
    ground_truth = """| item | value |
| --- | ---: |
| revenue | 1200 |
| margin | 12 |"""
    prediction = """| item | value |
| --- | ---: |
| revenue | $1,200 |
| margin | 12% |"""

    result = evaluate_chart_table(prediction, ground_truth, tolerance="strict")

    assert result["map_high"] == 1.0
    assert result["chartarena_eval_logs"]


def test_chinese_headers_and_categories_are_supported() -> None:
    ground_truth = """| 年份 | 收入 | 利润 |
| --- | ---: | ---: |
| 2023 | 100 | 20 |
| 2024 | 120 | 30 |"""
    prediction = """| 指标 | 2023 | 2024 |
| --- | ---: | ---: |
| 收入 | 100 | 120 |
| 利润 | 20 | 30 |"""

    result = evaluate_chart_table(prediction, ground_truth, tolerance="strict")

    assert result["triple_f1"] == 1.0
    assert result["exact_match"] == 1.0


def test_extract_chart_tables_from_record_matches_real_0json_structure() -> None:
    record = json.loads(Path("outputs/final/0.json").read_text(encoding="utf-8"))

    tables = extract_chart_tables_from_record(record)

    assert len(tables) >= 2
    assert tables[0]["path"] == "parsed.extraction_results[0].json_res[0].content"
    assert tables[1]["path"] == "parsed.extraction_results[0].json_res[2].content"
    assert tables[0]["content"].startswith("| Sessions |")


def test_evaluate_from_record_self_gold() -> None:
    record = json.loads(Path("outputs/final/0.json").read_text(encoding="utf-8"))

    result = evaluate_from_record(record, chart_index=0, tolerance="strict")

    assert result["prediction_field_path"] == "parsed.extraction_results[0].json_res[0].content"
    assert result["groundtruth_field_path"] == "parsed.extraction_results[0].json_res[0].content"
    assert result["triple_f1"] == 1.0
    assert result["exact_match"] == 1.0


def test_evaluate_from_record_supports_actual_shape_sample() -> None:
    sample_record = {
        "parsed": {
            "extraction_results": [
                {
                    "page": 0,
                    "json_res": [
                        {
                            "type": "chart",
                            "content": BASE_TABLE,
                        }
                    ],
                }
            ]
        }
    }

    result = evaluate_from_record(sample_record, chart_index=0, tolerance="strict")

    assert result["prediction_field_path"] == "parsed.extraction_results[0].json_res[0].content"
    assert result["triple_f1"] == 1.0
