from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_dataset.mermaid_td_f1.evaluator import evaluate_mermaid_flowchart


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate two Mermaid flowchart files.")
    parser.add_argument("--pred", required=True, type=Path, help="Prediction Mermaid file")
    parser.add_argument("--gold", required=True, type=Path, help="Ground truth Mermaid file")
    parser.add_argument("--output", required=True, type=Path, help="Output result JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_mermaid = args.pred.read_text(encoding="utf-8")
    gold_mermaid = args.gold.read_text(encoding="utf-8")
    result = evaluate_mermaid_flowchart(pred_mermaid=pred_mermaid, gold_mermaid=gold_mermaid)
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
        "structure_f1",
        "node_text_f1",
        "edge_text_f1",
        "binding_f1",
        "semantic_f1",
        "final_td_f1",
    ):
        print(f"{key}: {float(result[key]):.4f}")
    print("Evaluation completed.")
    print(f"Result saved to: {args.output}")


if __name__ == "__main__":
    main()
