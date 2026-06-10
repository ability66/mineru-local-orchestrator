from __future__ import annotations

from pathlib import Path
import json
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_dataset.mermaid_td_f1.evaluator import strip_mermaid_fence

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "outputs" / "final" / "figure2.json"
OUTPUT_PATH = REPO_ROOT / "eval_dataset" / "mermaid_td_f1" / "dataset.json"


def find_mermaid_fields(obj: Any) -> list[str]:
    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "mermaid" and isinstance(value, str):
                found.append(value)
            found.extend(find_mermaid_fields(value))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(find_mermaid_fields(item))
    return found


def find_content_fields(obj: Any) -> list[str]:
    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "content" and isinstance(value, str) and looks_like_mermaid(value):
                found.append(value)
            found.extend(find_content_fields(value))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(find_content_fields(item))
    return found


def looks_like_mermaid(text: str) -> bool:
    value = strip_mermaid_fence(text).strip()
    if not value:
        return False
    lowered = value.lower()
    if lowered.startswith("flowchart ") or lowered.startswith("graph "):
        return True
    return any(signal in value for signal in ("-->", "-.->", "==>"))


def build_dataset() -> list[dict[str, str]]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")
    payload = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    mermaid_candidates = find_mermaid_fields(payload)
    if mermaid_candidates:
        if len(mermaid_candidates) > 1:
            print(f"Found {len(mermaid_candidates)} mermaid fields; using the first one.")
        chosen = mermaid_candidates[0]
    else:
        content_candidates = find_content_fields(payload)
        if not content_candidates:
            raise ValueError("No Mermaid content found in figure2.json")
        if len(content_candidates) > 1:
            print(f"Found {len(content_candidates)} Mermaid-like content fields; using the first one.")
        chosen = content_candidates[0]

    sample = {
        "id": "figure2",
        "prediction": chosen,
        "groundtruth": chosen,
        "source": "outputs/final/figure2.json",
    }
    dataset = [sample]
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
