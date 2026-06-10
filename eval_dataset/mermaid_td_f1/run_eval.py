from __future__ import annotations

from pathlib import Path
import json
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_dataset.mermaid_td_f1.build_dataset import OUTPUT_PATH, build_dataset
from eval_dataset.mermaid_td_f1.evaluator import evaluate_mermaid_flowchart

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_dataset() -> list[dict[str, str]]:
    if not OUTPUT_PATH.exists():
        build_dataset()
    return json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))


def print_metrics(title: str, result: dict[str, object]) -> None:
    print(title)
    for key in (
        "structure_f1",
        "node_text_f1",
        "edge_text_f1",
        "binding_f1",
        "semantic_f1",
        "final_td_f1",
    ):
        print(f"  {key}: {float(result[key]):.4f}")
    print(f"  penalty: {float(result['penalty']):.4f}")
    debug = result.get("debug", {})
    errors = debug.get("errors", []) if isinstance(debug, dict) else []
    print(f"  debug.errors: {errors}")


def evaluate_dataset_samples() -> None:
    dataset = load_dataset()
    totals = {
        "structure_f1": 0.0,
        "node_text_f1": 0.0,
        "edge_text_f1": 0.0,
        "binding_f1": 0.0,
        "semantic_f1": 0.0,
        "final_td_f1": 0.0,
    }
    print("Dataset evaluation")
    for sample in dataset:
        result = evaluate_mermaid_flowchart(
            pred_mermaid=sample["prediction"],
            gold_mermaid=sample["groundtruth"],
        )
        print_metrics(f"- Sample {sample['id']}", result)
        for key in totals:
            totals[key] += float(result[key])
    count = max(len(dataset), 1)
    print("Average scores")
    for key, total in totals.items():
        print(f"  Average {key}: {total / count:.4f}")


def run_demo_tests() -> None:
    gold_branch = """flowchart TD
  A[开始] --> B{是否通过}
  B -- 是 --> C[通过]
  B -- 否 --> D[驳回]
"""
    pred_same = gold_branch
    pred_reordered = """flowchart TD
  A[开始] --> B{是否通过}
  B -- 否 --> D[驳回]
  B -- 是 --> C[通过]
"""
    pred_reversed = """flowchart TD
  A[开始] --> B{是否通过}
  B -- 是 --> D[驳回]
  B -- 否 --> C[通过]
"""
    gold_type = """flowchart TD
  A[开始] --> B{是否通过}
  B -- 是 --> C[通过]
  B -- 否 --> D[驳回]
  C --> E[结束]
"""
    pred_type = """flowchart TD
  A[开始] --> B[是否通过]
  B -- 是 --> C[通过]
  B -- 否 --> D[驳回]
  C --> E[结束]
"""
    cycle_graph = """flowchart TD
  A[开始] --> B[处理]
  B --> C{是否完成}
  C -- 否 --> B
  C -- 是 --> D[结束]
"""
    gold_virtual_routing = """flowchart TD
  A[处理A] --> C[处理C]
  A[处理A] --> D[处理D]
  B[处理B] --> C[处理C]
  B[处理B] --> D[处理D]
"""
    pred_virtual_routing = """flowchart TD
  A[处理A] --> V[ ]
  B[处理B] --> V
  V --> C[处理C]
  V --> D[处理D]
"""
    gold_virtual_chain = """flowchart TD
  A[开始] --> B[处理]
"""
    pred_virtual_chain = """flowchart TD
  A[开始] --> V1[ ]
  V1 --> V2[ ]
  V2 --> B[处理]
"""
    cases = [
        ("Test 1: prediction == groundtruth", pred_same, gold_branch),
        ("Test 2: sibling order reversed", pred_reordered, gold_branch),
        ("Test 3: polarity reversed", pred_reversed, gold_branch),
        ("Test 4: local node type mismatch", pred_type, gold_type),
        ("Test 5: cyclic graph", cycle_graph, cycle_graph),
        ("Test 6: virtual routing node normalization", pred_virtual_routing, gold_virtual_routing),
        ("Test 7: chained virtual node normalization", pred_virtual_chain, gold_virtual_chain),
    ]
    print("Demo tests")
    for title, pred_mermaid, gold_mermaid in cases:
        result = evaluate_mermaid_flowchart(
            pred_mermaid=pred_mermaid,
            gold_mermaid=gold_mermaid,
        )
        print_metrics(f"- {title}", result)


def main() -> None:
    evaluate_dataset_samples()
    print()
    run_demo_tests()


if __name__ == "__main__":
    main()
