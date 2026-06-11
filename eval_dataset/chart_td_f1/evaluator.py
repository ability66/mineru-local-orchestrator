from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


def _ensure_runtime_paths() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    venv_lib = repo_root / ".venv" / "lib"
    if venv_lib.exists():
        for site_packages in sorted(venv_lib.glob("python*/site-packages")):
            path_text = str(site_packages)
            if path_text not in sys.path:
                sys.path.append(path_text)


_ensure_runtime_paths()

try:
    import Levenshtein
    from .chartarena_vendor.methods.normalize import normalize_prediction_for_data, normalize_to_csv
    from .chartarena_vendor.methods.parsers.parse_json import json_to_internal_csv
    from .chartarena_vendor.metrics.SCRM import (
        _is_empty_header,
        csv2triples,
        csv_eval,
        process_triplets,
        union_with_tolerance,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "chart_td_f1 requires vendored ChartArena dependencies. "
        "Install them into .venv first, for example: "
        "./.venv/bin/pip install numpy python-Levenshtein pytest"
    ) from exc

TOLERANCE_PRESETS = {
    "strict": {"tol_word": 0, "tol_num": 0.05, "score_key": "map_strict"},
    "slight": {"tol_word": 1, "tol_num": 0.1, "score_key": "map_slight"},
    "high": {"tol_word": 5, "tol_num": 0.2, "score_key": "map_high"},
}


def evaluate_chart_table(
    prediction: Any,
    ground_truth: Any,
    tolerance: str = "slight",
    allow_transpose: bool = True,
) -> dict[str, Any]:
    config = _get_tolerance_config(tolerance)
    prediction_variants = _build_csv_variants(prediction, allow_transpose=allow_transpose)
    ground_truth_variants = _build_csv_variants(ground_truth, allow_transpose=False)

    if not prediction_variants or not ground_truth_variants:
        return _empty_result(
            tolerance=tolerance,
            parse_success=0.0,
            errors=[
                *([] if prediction_variants else ["prediction_parse_failed"]),
                *([] if ground_truth_variants else ["ground_truth_parse_failed"]),
            ],
        )

    best_result: dict[str, Any] | None = None
    for pred_variant_name, pred_csv in prediction_variants:
        for gt_variant_name, gt_csv in ground_truth_variants:
            candidate = _evaluate_csv_pair(
                pred_csv=pred_csv,
                gt_csv=gt_csv,
                pred_variant_name=pred_variant_name,
                gt_variant_name=gt_variant_name,
                tolerance=tolerance,
                config=config,
            )
            if best_result is None or _is_better_result(candidate, best_result):
                best_result = candidate

    assert best_result is not None
    best_result["parse_success"] = 1.0
    return best_result


def evaluate_from_record(
    record: dict[str, Any],
    tolerance: str = "slight",
    allow_transpose: bool = True,
    chart_index: int = 0,
    ground_truth_record: dict[str, Any] | None = None,
    ground_truth_chart_index: int | None = None,
) -> dict[str, Any]:
    prediction_blocks = extract_chart_tables_from_record(record)
    if chart_index < 0 or chart_index >= len(prediction_blocks):
        return _empty_result(
            tolerance=tolerance,
            parse_success=0.0,
            errors=[f"chart_index_out_of_range:{chart_index}"],
        )

    ground_truth_blocks = (
        extract_chart_tables_from_record(ground_truth_record)
        if ground_truth_record is not None
        else prediction_blocks
    )
    target_gt_index = chart_index if ground_truth_chart_index is None else ground_truth_chart_index
    if target_gt_index < 0 or target_gt_index >= len(ground_truth_blocks):
        return _empty_result(
            tolerance=tolerance,
            parse_success=0.0,
            errors=[f"ground_truth_chart_index_out_of_range:{target_gt_index}"],
        )

    prediction_item = prediction_blocks[chart_index]
    ground_truth_item = ground_truth_blocks[target_gt_index]
    result = evaluate_chart_table(
        prediction=prediction_item["content"],
        ground_truth=ground_truth_item["content"],
        tolerance=tolerance,
        allow_transpose=allow_transpose,
    )
    result["prediction_field_path"] = prediction_item["path"]
    result["groundtruth_field_path"] = ground_truth_item["path"]
    result["chart_index"] = chart_index
    result["groundtruth_chart_index"] = target_gt_index
    result["chart_count"] = len(prediction_blocks)
    result["groundtruth_chart_count"] = len(ground_truth_blocks)
    return result


def extract_chart_tables_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(record, dict):
        return []

    root = record
    prefix = ""
    parsed = record.get("parsed")
    if isinstance(parsed, dict) and isinstance(parsed.get("extraction_results"), list):
        root = parsed
        prefix = "parsed"

    extraction_results = root.get("extraction_results")
    if not isinstance(extraction_results, list):
        return []

    tables: list[dict[str, Any]] = []
    for page_index, page in enumerate(extraction_results):
        if not isinstance(page, dict):
            continue
        json_res = page.get("json_res")
        if not isinstance(json_res, list):
            continue
        for block_index, block in enumerate(json_res):
            if not isinstance(block, dict):
                continue
            if str(block.get("type", "")).strip().lower() != "chart":
                continue
            content = block.get("content")
            if isinstance(content, str) and content.strip():
                tables.append(
                    {
                        "page_index": page_index,
                        "block_index": block_index,
                        "path": _join_path(
                            prefix,
                            f"extraction_results[{page_index}]",
                            f"json_res[{block_index}]",
                            "content",
                        ),
                        "content": content,
                    }
                )
                continue
            if isinstance(content, dict):
                nested_content = str(content.get("content", "") or "").strip()
                if nested_content:
                    tables.append(
                        {
                            "page_index": page_index,
                            "block_index": block_index,
                            "path": _join_path(
                                prefix,
                                f"extraction_results[{page_index}]",
                                f"json_res[{block_index}]",
                                "content.content",
                            ),
                            "content": nested_content,
                        }
                    )
    return tables


def _build_csv_variants(value: Any, allow_transpose: bool) -> list[tuple[str, str]]:
    variants: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add_variant(name: str, csv_text: str) -> None:
        normalized = str(csv_text or "").strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        variants.append((name, normalized))

    if isinstance(value, (dict, list)):
        json_text = json.dumps(value, ensure_ascii=False)
        add_variant("json", json_to_internal_csv(json_text))
    else:
        text = str(value or "").strip()
        if text:
            if text.startswith(("{", "[")):
                add_variant("json", normalize_prediction_for_data(text, "SE_JSON"))
            add_variant("csv", normalize_prediction_for_data(text, "SE_CSV"))
            add_variant("code", normalize_prediction_for_data(text, "SE_CODE"))
            add_variant("markdown", normalize_prediction_for_data(text, "SE_MD"))
            add_variant("normalize_to_csv", normalize_to_csv(text))
            if "\\t" in text and "\\n" in text:
                add_variant("internal_csv", text)

    if allow_transpose:
        original_variants = list(variants)
        for name, csv_text in original_variants:
            transposed = _transpose_csv(csv_text)
            if transposed and transposed != csv_text:
                add_variant(f"{name}:transposed", transposed)

    return variants


def _evaluate_csv_pair(
    pred_csv: str,
    gt_csv: str,
    pred_variant_name: str,
    gt_variant_name: str,
    tolerance: str,
    config: dict[str, str | int | float],
) -> dict[str, Any]:
    scores, eval_logs = csv_eval([pred_csv], [gt_csv], easy=1)
    (
        em,
        map_strict,
        map_slight,
        map_high,
        ap_50_strict,
        ap_75_strict,
        ap_90_strict,
        ap_50_slight,
        ap_75_slight,
        ap_90_slight,
        ap_50_high,
        ap_75_high,
        ap_90_high,
    ) = scores

    aligned_pred_csv, aligned_gt_csv = _align_csv_pair(pred_csv, gt_csv)
    pred_triples = _unique_triples(aligned_pred_csv)
    gt_triples = _unique_triples(aligned_gt_csv)
    tol_word = int(config["tol_word"])
    tol_num = float(config["tol_num"])
    matched_pairs = _greedy_match_triples(pred_triples, gt_triples, tol_word=tol_word, tol_num=tol_num)
    matched = len(matched_pairs)
    pred_count = len(pred_triples)
    gt_count = len(gt_triples)
    triple_precision = _safe_div(matched, pred_count)
    triple_recall = _safe_div(matched, gt_count)
    triple_f1 = _safe_f1(triple_precision, triple_recall)
    triple_iou = _safe_div(
        matched,
        len(union_with_tolerance(pred_triples, gt_triples, tol_word, tol_num)),
    )

    return {
        "parse_success": 1.0,
        "triple_precision": round(triple_precision, 4),
        "triple_recall": round(triple_recall, 4),
        "triple_f1": round(triple_f1, 4),
        "triple_iou": round(triple_iou, 4),
        "exact_match": round(float(em), 4),
        "avg_numeric_error": round(
            _average_numeric_error(pred_triples, gt_triples, matched_pairs),
            4,
        ),
        "matched": matched,
        "pred_count": pred_count,
        "gt_count": gt_count,
        "tolerance": tolerance,
        "pred_format": "chartarena_internal_csv",
        "gt_format": "chartarena_internal_csv",
        "pred_variant": pred_variant_name,
        "gt_variant": gt_variant_name,
        "map_strict": round(float(map_strict), 4),
        "map_slight": round(float(map_slight), 4),
        "map_high": round(float(map_high), 4),
        "ap_50_strict": round(float(ap_50_strict), 4),
        "ap_75_strict": round(float(ap_75_strict), 4),
        "ap_90_strict": round(float(ap_90_strict), 4),
        "ap_50_slight": round(float(ap_50_slight), 4),
        "ap_75_slight": round(float(ap_75_slight), 4),
        "ap_90_slight": round(float(ap_90_slight), 4),
        "ap_50_high": round(float(ap_50_high), 4),
        "ap_75_high": round(float(ap_75_high), 4),
        "ap_90_high": round(float(ap_90_high), 4),
        "chartarena_eval_logs": list(eval_logs or []),
    }


def _unique_triples(csv_text: str) -> list[tuple[str, str, Any]]:
    triples = process_triplets(csv2triples(csv_text, norm_logs=[]))
    unique = sorted(set(triples), key=lambda item: tuple(str(part) for part in item))
    return list(unique)


def _align_csv_pair(pred_csv: str, gt_csv: str, separator: str = "\\t", delimiter: str = "\\n") -> tuple[str, str]:
    def _parse_rows(csv_str: str) -> list[list[str]]:
        if not csv_str:
            return []
        trimmed = csv_str.rstrip("\n\r")
        if not trimmed.strip():
            return []
        return [line.split(separator) for line in trimmed.split(delimiter)]

    pred_rows = _parse_rows(pred_csv)
    gt_rows = _parse_rows(gt_csv)
    pred_header = pred_rows[0] if pred_rows else []
    gt_header = gt_rows[0] if gt_rows else []
    pred_headerless = _is_empty_header(pred_header) if pred_header else False
    gt_headerless = _is_empty_header(gt_header) if gt_header else False

    if gt_headerless and not pred_headerless and len(pred_rows) > 1:
        empty_header = separator.join([" "] * len(pred_header))
        rest = pred_csv.rstrip("\n\r").split(delimiter)[1:]
        return empty_header + delimiter + delimiter.join(rest), gt_csv

    if pred_headerless and not gt_headerless and len(gt_rows) > 1:
        pred_data_rows = len(pred_rows) - 1
        gt_data_rows = len(gt_rows) - 1
        pred_columns = len(pred_rows[1]) if len(pred_rows) > 1 else len(pred_header)
        gt_columns = len(gt_rows[1]) if len(gt_rows) > 1 else len(gt_header)
        if abs(pred_data_rows - gt_data_rows) <= 1 and pred_columns == gt_columns and gt_columns >= 2:
            empty_header = separator.join([" "] * len(gt_header))
            rest = gt_csv.rstrip("\n\r").split(delimiter)[1:]
            return pred_csv, empty_header + delimiter + delimiter.join(rest)
        return pred_csv, gt_csv

    try_branch_c = not pred_headerless and not gt_headerless and len(pred_rows) > 1 and len(gt_rows) > 1
    if not try_branch_c:
        return pred_csv, gt_csv

    pred_data_rows = len(pred_rows) - 1
    gt_data_rows = len(gt_rows) - 1
    pred_columns = len(pred_rows[1])
    gt_columns = len(gt_rows[1])
    gt_first_col_empty = bool(gt_header) and gt_header[0].strip() == ""
    row_diff_ok = abs(pred_data_rows - gt_data_rows) <= 1
    col_match_ok = pred_columns == gt_columns + 1 and gt_columns >= 2
    pred_first_col_named = bool(pred_header) and pred_header[0].strip() != ""

    def _is_auto_index_column() -> bool:
        if len(pred_rows) < 2:
            return False
        if not pred_header or pred_header[0].strip() != "":
            return False
        first_column_values = [row[0].strip() for row in pred_rows[1:] if row]
        if not first_column_values:
            return False
        try:
            numbers = [int(value) for value in first_column_values]
        except ValueError:
            return False
        if len(numbers) < 2 or numbers[0] not in {0, 1}:
            return False
        return numbers == list(range(numbers[0], numbers[0] + len(numbers)))

    trigger_c1 = row_diff_ok and col_match_ok and gt_first_col_empty and pred_first_col_named
    trigger_c2 = row_diff_ok and col_match_ok and _is_auto_index_column()
    if trigger_c1 or trigger_c2:
        raw_lines = pred_csv.rstrip("\n\r").split(delimiter)
        new_lines: list[str] = []
        for line in raw_lines:
            cells = line.split(separator)
            if len(cells) <= 1:
                new_lines.append(line)
            else:
                new_lines.append(separator.join(cells[1:]))
        return delimiter.join(new_lines), gt_csv

    if pred_columns == 2 and gt_columns == 2 and row_diff_ok:
        pred_first_empty = bool(pred_header) and pred_header[0].strip() == ""
        pred_col_named = len(pred_header) >= 2 and pred_header[1].strip() != ""
        gt_col_named = len(gt_header) >= 2 and gt_header[1].strip() != ""
        first_col_sym = pred_first_empty == gt_first_col_empty
        first_col_pred_named_ref_empty = (not pred_first_empty) and gt_first_col_empty
        if pred_col_named and gt_col_named and (first_col_sym or first_col_pred_named_ref_empty):
            empty_header = separator.join([" "] * 2)
            pred_rest = pred_csv.rstrip("\n\r").split(delimiter)[1:]
            gt_rest = gt_csv.rstrip("\n\r").split(delimiter)[1:]
            return (
                empty_header + delimiter + delimiter.join(pred_rest),
                empty_header + delimiter + delimiter.join(gt_rest),
            )

    return pred_csv, gt_csv


def _greedy_match_triples(
    pred_triples: list[tuple[str, str, Any]],
    gt_triples: list[tuple[str, str, Any]],
    tol_word: int,
    tol_num: float,
) -> list[tuple[int, int, float]]:
    candidates: list[tuple[float, int, int]] = []
    for pred_index, pred_triple in enumerate(pred_triples):
        for gt_index, gt_triple in enumerate(gt_triples):
            score = _match_score(pred_triple, gt_triple, tol_word=tol_word, tol_num=tol_num)
            if score is None:
                continue
            candidates.append((score, pred_index, gt_index))

    candidates.sort(reverse=True)
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, pred_index, gt_index in candidates:
        if pred_index in used_pred or gt_index in used_gt:
            continue
        used_pred.add(pred_index)
        used_gt.add(gt_index)
        matches.append((pred_index, gt_index, score))
    return matches


def _match_score(
    pred_triple: tuple[str, str, Any],
    gt_triple: tuple[str, str, Any],
    tol_word: int,
    tol_num: float,
) -> float | None:
    pred_value = pred_triple[-1]
    gt_value = gt_triple[-1]

    if isinstance(pred_value, float) and isinstance(gt_value, float):
        key_distance = Levenshtein.distance("".join(pred_triple[:-1]), "".join(gt_triple[:-1]))
        if key_distance > tol_word:
            return None
        relative_error = abs(pred_value - gt_value) / (gt_value + 0.000001)
        if relative_error > tol_num:
            return None
        return 10.0 - key_distance - relative_error

    full_distance = Levenshtein.distance(
        "".join(str(part) for part in pred_triple),
        "".join(str(part) for part in gt_triple),
    )
    if full_distance > tol_word:
        return None
    return 10.0 - full_distance


def _average_numeric_error(
    pred_triples: list[tuple[str, str, Any]],
    gt_triples: list[tuple[str, str, Any]],
    matched_pairs: list[tuple[int, int, float]],
) -> float:
    penalties: list[float] = []
    matched_pred = {pred_index for pred_index, _, _ in matched_pairs}
    matched_gt = {gt_index for _, gt_index, _ in matched_pairs}

    for pred_index, gt_index, _ in matched_pairs:
        pred_value = pred_triples[pred_index][-1]
        gt_value = gt_triples[gt_index][-1]
        if isinstance(pred_value, float) and isinstance(gt_value, float):
            penalties.append(abs(pred_value - gt_value) / (abs(gt_value) + 1.0))
        elif isinstance(pred_value, float) or isinstance(gt_value, float):
            penalties.append(1.0)

    for pred_index, pred_triple in enumerate(pred_triples):
        if pred_index not in matched_pred and isinstance(pred_triple[-1], float):
            penalties.append(1.0)
    for gt_index, gt_triple in enumerate(gt_triples):
        if gt_index not in matched_gt and isinstance(gt_triple[-1], float):
            penalties.append(1.0)

    if not penalties:
        return 0.0
    return sum(min(1.0, penalty) for penalty in penalties) / len(penalties)


def _transpose_csv(csv_str: str, separator: str = " \\t ", delimiter: str = " \\n ") -> str:
    if not csv_str or not csv_str.strip():
        return ""

    rows = csv_str.split(delimiter)
    if len(rows) < 2:
        return ""

    matrix = [row.split(separator) for row in rows]
    column_count = len(matrix[0])
    if column_count < 2:
        return ""
    if not all(len(row) == column_count for row in matrix):
        min_columns = min(len(row) for row in matrix)
        if min_columns < 2:
            return ""
        matrix = [row[:min_columns] for row in matrix]
        column_count = min_columns

    transposed = []
    for column_index in range(column_count):
        transposed.append([matrix[row_index][column_index] for row_index in range(len(matrix))])
    return delimiter.join(separator.join(row) for row in transposed)


def _get_tolerance_config(tolerance: str) -> dict[str, str | int | float]:
    if tolerance not in TOLERANCE_PRESETS:
        supported = ", ".join(sorted(TOLERANCE_PRESETS))
        raise ValueError(f"Unsupported tolerance: {tolerance}. Supported: {supported}")
    return TOLERANCE_PRESETS[tolerance]


def _is_better_result(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_key = (
        float(left["map_high"]),
        float(left["map_slight"]),
        float(left["map_strict"]),
        float(left["exact_match"]),
        float(left["triple_f1"]),
        -float(left["avg_numeric_error"]),
    )
    right_key = (
        float(right["map_high"]),
        float(right["map_slight"]),
        float(right["map_strict"]),
        float(right["exact_match"]),
        float(right["triple_f1"]),
        -float(right["avg_numeric_error"]),
    )
    return left_key > right_key


def _empty_result(
    tolerance: str,
    parse_success: float,
    errors: list[str],
) -> dict[str, Any]:
    return {
        "parse_success": parse_success,
        "triple_precision": 0.0,
        "triple_recall": 0.0,
        "triple_f1": 0.0,
        "triple_iou": 0.0,
        "exact_match": 0.0,
        "avg_numeric_error": 1.0 if parse_success == 0.0 else 0.0,
        "matched": 0,
        "pred_count": 0,
        "gt_count": 0,
        "tolerance": tolerance,
        "pred_format": "chartarena_internal_csv",
        "gt_format": "chartarena_internal_csv",
        "pred_variant": "",
        "gt_variant": "",
        "map_strict": 0.0,
        "map_slight": 0.0,
        "map_high": 0.0,
        "ap_50_strict": 0.0,
        "ap_75_strict": 0.0,
        "ap_90_strict": 0.0,
        "ap_50_slight": 0.0,
        "ap_75_slight": 0.0,
        "ap_90_slight": 0.0,
        "ap_50_high": 0.0,
        "ap_75_high": 0.0,
        "ap_90_high": 0.0,
        "chartarena_eval_logs": [],
        "errors": errors,
    }


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _safe_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _join_path(*parts: str) -> str:
    filtered = [part for part in parts if part]
    if not filtered:
        return ""
    path = filtered[0]
    for part in filtered[1:]:
        path += f".{part}"
    return path


def main() -> None:
    sample_path = Path("outputs/final/0.json")
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)
    record = json.loads(sample_path.read_text(encoding="utf-8"))
    result = evaluate_from_record(record, tolerance="slight", chart_index=0)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
