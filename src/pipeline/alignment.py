from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from src.schema import CanonicalBlock

_VISUAL_TYPES = {"chart", "image", "table"}
_TEXT_TYPES = {"title", "paragraph", "list", "page_header", "page_footer", "page_footnote"}


@dataclass
class BlockMatch:
    base_index: int
    candidate_index: int
    score: float
    bbox_iou: float
    text_similarity: float


def align_blocks(
    base_blocks: list[CanonicalBlock],
    candidate_blocks: list[CanonicalBlock],
    min_score: float = 0.35,
) -> list[BlockMatch]:
    scored_pairs: list[BlockMatch] = []
    for base_index, base_block in enumerate(base_blocks):
        for candidate_index, candidate_block in enumerate(candidate_blocks):
            if base_block.page_idx != candidate_block.page_idx:
                continue
            bbox_score = bbox_iou(base_block.bbox, candidate_block.bbox)
            text_score = text_similarity(base_block.text, candidate_block.text)
            type_score = _type_compatibility(base_block.type, candidate_block.type)
            score = round(0.50 * bbox_score + 0.30 * text_score + 0.20 * type_score, 4)
            if score < min_score:
                continue
            scored_pairs.append(
                BlockMatch(
                    base_index=base_index,
                    candidate_index=candidate_index,
                    score=score,
                    bbox_iou=bbox_score,
                    text_similarity=text_score,
                )
            )

    scored_pairs.sort(key=lambda item: (item.score, item.bbox_iou, item.text_similarity), reverse=True)
    matched_base: set[int] = set()
    matched_candidate: set[int] = set()
    selected: list[BlockMatch] = []

    for pair in scored_pairs:
        if pair.base_index in matched_base or pair.candidate_index in matched_candidate:
            continue
        matched_base.add(pair.base_index)
        matched_candidate.add(pair.candidate_index)
        selected.append(pair)

    selected.sort(key=lambda item: item.base_index)
    return selected


def bbox_iou(left: list[int], right: list[int]) -> float:
    if len(left) != 4 or len(right) != 4:
        return 0.0

    inter_left = max(left[0], right[0])
    inter_top = max(left[1], right[1])
    inter_right = min(left[2], right[2])
    inter_bottom = min(left[3], right[3])
    inter_width = max(0, inter_right - inter_left)
    inter_height = max(0, inter_bottom - inter_top)
    inter_area = inter_width * inter_height
    if inter_area <= 0:
        return 0.0

    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - inter_area
    if union <= 0:
        return 0.0
    return round(inter_area / union, 4)


def text_similarity(left: str, right: str) -> float:
    normalized_left = _normalize_text(left)
    normalized_right = _normalize_text(right)
    if not normalized_left and not normalized_right:
        return 1.0
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left == normalized_right:
        return 1.0

    left_bigrams = _char_bigrams(normalized_left)
    right_bigrams = _char_bigrams(normalized_right)
    union = left_bigrams | right_bigrams
    if not union:
        return 0.0
    return round(len(left_bigrams & right_bigrams) / len(union), 4)


def _type_compatibility(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if left in _VISUAL_TYPES and right in _VISUAL_TYPES:
        return 0.75
    if left in _TEXT_TYPES and right in _TEXT_TYPES:
        return 0.65
    return 0.0


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    return "".join(normalized.split()).lower()


def _char_bigrams(text: str) -> set[str]:
    if len(text) < 2:
        return {text} if text else set()
    return {text[index : index + 2] for index in range(len(text) - 1)}

