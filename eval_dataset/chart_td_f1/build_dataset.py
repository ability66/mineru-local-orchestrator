from __future__ import annotations

from pathlib import Path
import json
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_dataset.chart_td_f1.evaluator import extract_chart_tables_from_record

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "outputs" / "final" / "0.json"
OUTPUT_PATH = REPO_ROOT / "eval_dataset" / "chart_td_f1" / "dataset.json"


def build_dataset() -> list[dict[str, str | int]]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    record = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    chart_blocks = extract_chart_tables_from_record(record)
    if not chart_blocks:
        raise ValueError("No chart table content found in outputs/final/0.json")

    dataset: list[dict[str, str | int]] = []
    for chart_index, chart_block in enumerate(chart_blocks):
        dataset.append(
            {
                "id": f"0_chart_{chart_index}",
                "prediction": chart_block["content"],
                "groundtruth": chart_block["content"],
                "source": "outputs/final/0.json",
                "prediction_field_path": chart_block["path"],
                "groundtruth_field_path": chart_block["path"],
                "chart_index": chart_index,
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return dataset


def main() -> None:
    dataset = build_dataset()
    print(f"Dataset built with {len(dataset)} sample(s).")
    print(f"Saved to: {OUTPUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
