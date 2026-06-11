from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_dataset.chart_td_f1.evaluator import evaluate_chart_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate two chart-table files.")
    parser.add_argument("--pred", required=True, type=Path, help="Prediction table file")
    parser.add_argument("--gold", required=True, type=Path, help="Ground truth table file")
    parser.add_argument("--output", required=True, type=Path, help="Output result JSON path")
    parser.add_argument(
        "--tolerance",
        default="slight",
        choices=("strict", "slight", "high"),
        help="Matching tolerance level",
    )
    parser.add_argument(
        "--disable-transpose",
        action="store_true",
        help="Disable transpose fallback",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction = args.pred.read_text(encoding="utf-8")
    ground_truth = args.gold.read_text(encoding="utf-8")
    result = evaluate_chart_table(
        prediction=prediction,
        ground_truth=ground_truth,
        tolerance=args.tolerance,
        allow_transpose=not args.disable_transpose,
    )
    payload = {
        "pred_file": str(args.pred),
        "gold_file": str(args.gold),
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
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
        print(f"{key}: {float(result[key]):.4f}")
    print("Evaluation completed.")
    print(f"Result saved to: {args.output}")


if __name__ == "__main__":
    main()
