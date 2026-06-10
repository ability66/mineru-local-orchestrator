from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
from typing import Any
from urllib import error, request

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_dataset.mermaid_td_f1.evaluator import (
    evaluate_mermaid_flowchart,
    strip_mermaid_fence,
)

DEFAULT_BASE_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_OUTPUT_DIR = (
    Path("eval_dataset") / "mermaid_td_f1" / "qwen_debug_outputs"
)
DEFAULT_FAIL_THRESHOLD = 0.999999
DEFAULT_TIMEOUT = 120
DEFAULT_MODEL = "qwen"
DEFAULT_TEMPERATURE = 0.9
DEFAULT_MAX_TOKENS = 8192
BUSINESS_SCENARIOS = [
    "审批流程",
    "贷款流程",
    "保险理赔流程",
    "采购流程",
    "订单流程",
    "工单流程",
    "招聘流程",
    "报销流程",
    "售后流程",
    "发票审核流程",
    "入职流程",
    "合同审批流程",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use local Qwen to generate equivalent Mermaid pairs and stress-test TD-F1."
    )
    parser.add_argument("--num-tests", type=int, default=20)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fail-threshold", type=float, default=DEFAULT_FAIL_THRESHOLD)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def build_qwen_prompt(iteration: int, self_check: bool = False) -> str:
    scenario_hint = random.choice(BUSINESS_SCENARIOS)
    self_check_block = ""
    if self_check:
        self_check_block = """
额外要求：
1. 先用自然语言解释为什么这两张图业务语义完全一致。
2. 将这段解释写入 JSON 的 "self_check_explanation" 字段。
3. "equivalence_reason" 至少给出 6 条逐项对应理由。
"""
    return f"""
你在做 Mermaid flowchart 评测器的压力测试，不是在做摘要，也不是在做解释题。

任务：
你必须自己创造一个新的复杂业务流程，并同时输出两张 Mermaid：
1. gold_mermaid：相对标准、直接的写法
2. pred_mermaid：业务语义完全一致，但 Mermaid 表达方式尽可能不同、尽可能极端

本轮编号：{iteration}
建议业务主题：{scenario_hint}

强制要求：
1. 两张图业务语义完全一致。
2. 两张图流程逻辑完全一致。
3. 节点数至少 20。
4. decision 节点至少 5。
5. 必须包含至少一个 loop。
6. 必须包含至少一个 merge。
7. 必须包含至少一个 yes/no 分支。
8. 必须包含至少一个失败路径。
9. 必须包含至少一个成功路径。
10. 必须包含至少一个多父节点汇聚。

你必须主动制造这些评测挑战：
- 节点 ID 全部重命名
- 边声明顺序完全打乱
- 同一 decision 的分支顺序交换
- 插入虚拟空节点
- 插入多个连续虚拟节点
- 使用汇聚节点改写 merge
- 使用分发节点改写 fan-out
- 调整缩进
- 调整节点定义顺序
- root 后面的边顺序随机

禁止事项：
1. 不要只改格式或只改缩进。
2. 不要输出简单流程。
3. 不要让 pred_mermaid 比 gold_mermaid 仅仅多几个空格。
4. 不要删除或新增业务语义节点。
5. 不要改变 yes/no、成功/失败、通过/拒绝 等条件含义。
6. 不要把不等价图伪装成等价图。
7. 不要输出任何 Markdown 解释性正文。

业务文字要求：
1. 业务节点的显示文字必须保持一致，便于对齐。
2. pred_mermaid 中可以重命名节点 ID，但业务节点文字必须与 gold_mermaid 对应一致。
3. yes/no、成功/失败等关键条件文字必须语义一致。

输出格式：
你只能输出一个 JSON object，不要输出任何额外说明，不要输出 Markdown 标题。
允许 JSON 中的 Mermaid 使用多行字符串。

JSON schema:
{{
  "scenario": "字符串，说明业务场景，例如贷款审批流程",
  "gold_mermaid": "Mermaid flowchart 字符串，必须是可被 JSON 解析的字符串；换行请写成 \\n",
  "pred_mermaid": "与 gold_mermaid 语义完全一致但表达差异极大的 Mermaid flowchart 字符串，必须是可被 JSON 解析的字符串；换行请写成 \\n",
  "equivalence_reason": [
    "字符串1",
    "字符串2"
  ]
{',"self_check_explanation": "详细说明两图为什么等价"' if self_check else ""}
}}

必须确保：
1. gold_mermaid 和 pred_mermaid 都是有效的 Mermaid flowchart。
2. 两张图都使用 flowchart TD 或 graph TD。
3. 两张图都可以独立阅读，不依赖外部说明。
4. pred_mermaid 必须尽可能挑战评测器，而不是接近 gold_mermaid。
{self_check_block}

只输出 JSON。
""".strip()


def parse_qwen_case(text: str) -> dict[str, Any]:
    json_text = extract_first_json_object(text)
    if json_text is None:
        raise ValueError("No JSON object found in Qwen output")
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        try:
            payload = json.loads(sanitize_jsonish_text(json_text))
        except json.JSONDecodeError as inner_exc:
            raise ValueError(f"Invalid JSON from Qwen: {inner_exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Qwen output JSON must be an object")

    scenario = str(payload.get("scenario", "") or "").strip()
    gold_mermaid = normalize_mermaid_text_field(payload.get("gold_mermaid"))
    pred_mermaid = normalize_mermaid_text_field(payload.get("pred_mermaid"))
    equivalence_reason = payload.get("equivalence_reason")
    self_check_explanation = str(payload.get("self_check_explanation", "") or "").strip()

    if not scenario:
        raise ValueError("Missing scenario")
    if not gold_mermaid:
        raise ValueError("Missing gold_mermaid")
    if not pred_mermaid:
        raise ValueError("Missing pred_mermaid")
    if not isinstance(equivalence_reason, list):
        raise ValueError("equivalence_reason must be a list")

    normalized_reasons = [
        str(item).strip()
        for item in equivalence_reason
        if str(item).strip()
    ]
    if not normalized_reasons:
        raise ValueError("equivalence_reason must contain at least one item")

    return {
        "scenario": scenario,
        "gold_mermaid": gold_mermaid,
        "pred_mermaid": pred_mermaid,
        "equivalence_reason": normalized_reasons,
        "self_check_explanation": self_check_explanation,
    }


def normalize_mermaid_text_field(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return strip_mermaid_fence(text).strip()


def extract_first_json_object(text: str) -> str | None:
    raw_text = str(text or "").strip()
    if not raw_text:
        return None
    fence_match = None
    if raw_text.startswith("```"):
        fence_match = raw_text
    if fence_match is not None:
        lines = raw_text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            raw_text = "\n".join(lines[:-1]).strip()
        else:
            raw_text = "\n".join(lines).strip()

    start_index: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(raw_text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start_index = index
            depth += 1
            continue
        if char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start_index is not None:
                return raw_text[start_index : index + 1]
    return None


def sanitize_jsonish_text(text: str) -> str:
    chars: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                chars.append(char)
                escaped = False
                continue
            if char == "\\":
                chars.append(char)
                escaped = True
                continue
            if char == '"':
                chars.append(char)
                in_string = False
                continue
            if char == "\n":
                chars.append("\\n")
                continue
            if char == "\r":
                chars.append("\\r")
                continue
            if char == "\t":
                chars.append("\\t")
                continue
            chars.append(char)
            continue
        chars.append(char)
        if char == '"':
            in_string = True
    return "".join(chars)


def call_local_qwen(
    *,
    model: str,
    base_url: str,
    api_key: str,
    prompt: str,
    temperature: float,
    timeout: int,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        base_url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw_response = response.read().decode("utf-8")
    except error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {error_body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"URL error: {exc}") from exc

    try:
        response_json = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from Qwen: {exc}") from exc

    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Qwen response missing choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError("Qwen response missing message")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "") or ""))
        joined = "".join(chunks).strip()
        if joined:
            return joined
    raise RuntimeError("Qwen response message.content is empty")


def run_stress_test(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_api_key = resolve_api_key(args.api_key)
    log_path = output_dir / "stress_test_log.jsonl"

    for iteration in range(1, args.num_tests + 1):
        prompt = build_qwen_prompt(iteration, self_check=args.self_check)
        raw_output = ""
        case: dict[str, Any] | None = None
        result: dict[str, Any] | None = None
        error_message = ""

        try:
            raw_output = call_local_qwen(
                model=args.model,
                base_url=args.base_url,
                api_key=resolved_api_key,
                prompt=prompt,
                temperature=args.temperature,
                timeout=args.timeout,
            )
            case = parse_qwen_case(raw_output)
            result = evaluate_mermaid_flowchart(
                pred_mermaid=case["pred_mermaid"],
                gold_mermaid=case["gold_mermaid"],
            )
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"

        record = build_log_record(
            iteration=iteration,
            case=case,
            result=result,
            error_message=error_message,
            self_check=args.self_check,
        )
        append_jsonl(log_path, record)

        if case is not None and result is not None:
            print_iteration_result(iteration=iteration, case=case, result=result)
        else:
            print(f"[{iteration}/{args.num_tests}] failed before scoring: {error_message}")

        if should_stop(
            result=result,
            error_message=error_message,
            fail_threshold=args.fail_threshold,
        ):
            save_failure(
                output_dir=output_dir,
                iteration=iteration,
                prompt=prompt,
                raw_output=raw_output,
                case=case,
                result=result,
                error_message=error_message,
                self_check=args.self_check,
            )
            print(f"Stopped at iteration {iteration}. Failure artifacts saved to: {output_dir}")
            return 1

    print(f"Stress test completed without failures. Log saved to: {log_path}")
    return 0


def build_log_record(
    *,
    iteration: int,
    case: dict[str, Any] | None,
    result: dict[str, Any] | None,
    error_message: str,
    self_check: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "iteration": iteration,
        "error": error_message,
        "self_check": self_check,
    }
    if case is not None:
        record.update(
            {
                "scenario": case["scenario"],
                "equivalence_reason": case["equivalence_reason"],
                "gold_mermaid": case["gold_mermaid"],
                "pred_mermaid": case["pred_mermaid"],
            }
        )
        if case.get("self_check_explanation"):
            record["self_check_explanation"] = case["self_check_explanation"]
    if result is not None:
        record["scores"] = summarize_scores(result)
        record["parse_valid"] = bool(result.get("parse_valid", False))
    return record


def print_iteration_result(
    *,
    iteration: int,
    case: dict[str, Any],
    result: dict[str, Any],
) -> None:
    print(
        f"[{iteration}] "
        f"scenario={case['scenario']} "
        f"structure_f1={float(result['structure_f1']):.6f} "
        f"semantic_f1={float(result['semantic_f1']):.6f} "
        f"binding_f1={float(result['binding_f1']):.6f} "
        f"final_td_f1={float(result['final_td_f1']):.6f}"
    )


def should_stop(
    *,
    result: dict[str, Any] | None,
    error_message: str,
    fail_threshold: float,
) -> bool:
    if error_message:
        return True
    if result is None:
        return True
    if not bool(result.get("parse_valid", False)):
        return True
    return any(
        float(result[key]) < fail_threshold
        for key in (
            "structure_f1",
            "semantic_f1",
            "binding_f1",
            "final_td_f1",
        )
    )


def save_failure(
    *,
    output_dir: Path,
    iteration: int,
    prompt: str,
    raw_output: str,
    case: dict[str, Any] | None,
    result: dict[str, Any] | None,
    error_message: str,
    self_check: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"failure_iter_{iteration}"
    payload: dict[str, Any] = {
        "iteration": iteration,
        "error": error_message,
        "self_check": self_check,
        "scores": summarize_scores(result) if result is not None else None,
    }
    if case is not None:
        payload.update(
            {
                "scenario": case["scenario"],
                "equivalence_reason": case["equivalence_reason"],
                "gold_mermaid": case["gold_mermaid"],
                "pred_mermaid": case["pred_mermaid"],
            }
        )
        if case.get("self_check_explanation"):
            payload["self_check_explanation"] = case["self_check_explanation"]
    if result is not None:
        payload["result"] = result

    (output_dir / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / f"{stem}_prompt.txt").write_text(prompt, encoding="utf-8")
    (output_dir / f"{stem}_raw_output.txt").write_text(raw_output, encoding="utf-8")
    if case is not None:
        (output_dir / f"{stem}_gold.mmd").write_text(
            case["gold_mermaid"],
            encoding="utf-8",
        )
        (output_dir / f"{stem}_pred.mmd").write_text(
            case["pred_mermaid"],
            encoding="utf-8",
        )


def summarize_scores(result: dict[str, Any] | None) -> dict[str, float] | None:
    if result is None:
        return None
    return {
        "structure_f1": float(result.get("structure_f1", 0.0)),
        "node_text_f1": float(result.get("node_text_f1", 0.0)),
        "edge_text_f1": float(result.get("edge_text_f1", 0.0)),
        "binding_f1": float(result.get("binding_f1", 0.0)),
        "semantic_f1": float(result.get("semantic_f1", 0.0)),
        "final_td_f1": float(result.get("final_td_f1", 0.0)),
        "penalty": float(result.get("penalty", 0.0)),
    }


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def resolve_api_key(api_key_arg: str | None) -> str:
    if api_key_arg is not None and api_key_arg.strip():
        return api_key_arg.strip()
    for env_name in ("LOCAL_QWEN_API_KEY", "QWEN_LOCAL_API_KEY"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return "EMPTY"


def main() -> None:
    args = parse_args()
    raise SystemExit(run_stress_test(args))


if __name__ == "__main__":
    main()
