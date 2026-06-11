from __future__ import annotations

from pathlib import Path
import json
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_dataset.chart_td_f1.build_dataset import OUTPUT_PATH, build_dataset
from eval_dataset.chart_td_f1.evaluator import evaluate_chart_table


def load_dataset() -> list[dict[str, object]]:
    if not OUTPUT_PATH.exists():
        build_dataset()
    return json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))


def print_metrics(title: str, result: dict[str, object]) -> None:
    print(title)
    for key in (
        "parse_success",
        "triple_precision",
        "triple_recall",
        "triple_f1",
        "triple_iou",
        "exact_match",
        "avg_numeric_error",
        "map_strict",
        "map_slight",
        "map_high",
    ):
        print(f"  {key}: {float(result[key]):.4f}")
    print(f"  matched: {int(result['matched'])}")
    print(f"  pred_count: {int(result['pred_count'])}")
    print(f"  gt_count: {int(result['gt_count'])}")
    if result.get("errors"):
        print(f"  errors: {result['errors']}")


def evaluate_dataset_samples() -> None:
    dataset = load_dataset()
    totals = {
        "parse_success": 0.0,
        "triple_precision": 0.0,
        "triple_recall": 0.0,
        "triple_f1": 0.0,
        "triple_iou": 0.0,
        "exact_match": 0.0,
        "avg_numeric_error": 0.0,
        "map_strict": 0.0,
        "map_slight": 0.0,
        "map_high": 0.0,
    }
    print("Dataset evaluation")
    for sample in dataset:
        result = evaluate_chart_table(
            prediction=sample["prediction"],
            ground_truth=sample["groundtruth"],
        )
        print_metrics(f"- Sample {sample['id']}", result)
        for key in totals:
            totals[key] += float(result[key])
    count = max(len(dataset), 1)
    print("Average scores")
    for key, total in totals.items():
        print(f"  Average {key}: {total / count:.4f}")


def run_demo_tests() -> None:
    gold = """| year | revenue | profit |
| --- | ---: | ---: |
| 2023 | 100 | 20 |
| 2024 | 120 | 30 |"""
    pred_row_reordered = """| year | revenue | profit |
| --- | ---: | ---: |
| 2024 | 120 | 30 |
| 2023 | 100 | 20 |"""
    pred_transposed = """| metric | 2023 | 2024 |
| --- | ---: | ---: |
| revenue | 100 | 120 |
| profit | 20 | 30 |"""
    pred_slight_error = """| year | revenue | profit |
| --- | ---: | ---: |
| 2023 | 100 | 20 |
| 2024 | 126 | 30 |"""
    cases = [
        ("Test 1: prediction == groundtruth", gold, gold, "strict"),
        ("Test 2: row order swapped", pred_row_reordered, gold, "strict"),
        ("Test 3: transposed table", pred_transposed, gold, "strict"),
        ("Test 4: slight numeric error", pred_slight_error, gold, "slight"),
    ]
    print("Demo tests")
    for title, prediction, ground_truth, tolerance in cases:
        result = evaluate_chart_table(
            prediction=prediction,
            ground_truth=ground_truth,
            tolerance=tolerance,
        )
        print_metrics(f"- {title}", result)


def main() -> None:
    evaluate_dataset_samples()
    print()
    run_demo_tests()


if __name__ == "__main__":
    main()
