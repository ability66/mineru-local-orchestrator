from __future__ import annotations

from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def load_default_prompt(config_path: Path) -> str:
    return load_prompt(config_path=config_path, prompt_name="default_prompt")


def load_prompt(config_path: Path, prompt_name: str) -> str:
    if not config_path.exists():
        raise FileNotFoundError(f"Prompt config not found: {config_path}")

    raw_text = config_path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(raw_text) or {}
    else:
        data = _fallback_prompt_parser(raw_text)

    prompt = data.get(prompt_name)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Missing non-empty '{prompt_name}' in prompts config")

    return prompt.strip()


def _fallback_prompt_parser(raw_text: str) -> dict[str, str]:
    lines = raw_text.splitlines()
    prompts: dict[str, str] = {}
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.endswith(": |") and not stripped.endswith(":|") and ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        if not key.endswith("_prompt") and key != "default_prompt":
            continue

        if stripped.endswith("|"):
            block: list[str] = []
            for subline in lines[index + 1 :]:
                if subline.startswith("  "):
                    block.append(subline[2:])
                    continue
                if not subline.strip():
                    block.append("")
                    continue
                break
            prompts[key] = "\n".join(block).rstrip()
            continue

        prompts[key] = value.strip().strip("'").strip('"')

    if not prompts:
        raise ValueError("Missing prompt entries in prompts config")
    return prompts
