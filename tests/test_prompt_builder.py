from __future__ import annotations

from src.prompt_builder import load_prompt


def test_load_prompt_reads_named_prompts(tmp_path) -> None:
    prompt_path = tmp_path / "prompts.yaml"
    prompt_path.write_text(
        "\n".join(
            [
                "default_prompt: |",
                "  default body",
                "qwen_recognition_prompt: |",
                "  recognition body",
                "qwen_adjudication_prompt: |",
                "  adjudication body",
                "qwen_flowchart_adjudication_prompt: |",
                "  flowchart adjudication body",
                "qwen_table_adjudication_prompt: |",
                "  table adjudication body",
            ]
        ),
        encoding="utf-8",
    )

    assert load_prompt(prompt_path, "default_prompt") == "default body"
    assert load_prompt(prompt_path, "qwen_recognition_prompt") == "recognition body"
    assert load_prompt(prompt_path, "qwen_adjudication_prompt") == "adjudication body"
    assert load_prompt(prompt_path, "qwen_flowchart_adjudication_prompt") == "flowchart adjudication body"
    assert load_prompt(prompt_path, "qwen_table_adjudication_prompt") == "table adjudication body"
