from __future__ import annotations

from typing import Any

from src.preprocess.schema import CropGroup, LayoutBlock

PRIMARY_TYPES = {"chart", "image", "table"}
CONTEXT_TYPES = {
    "image_caption",
    "image_footnote",
    "table_caption",
    "table_footnote",
    "chart_caption",
    "chart_footnote",
}


def normalize_layout_blocks(raw_blocks: list[dict[str, Any]]) -> list[LayoutBlock]:
    blocks: list[LayoutBlock] = []
    for index, raw_block in enumerate(raw_blocks):
        block_type = str(raw_block.get("type", "") or "").strip().lower()
        bbox = _normalize_bbox(raw_block.get("bbox"))
        if not block_type or not bbox:
            continue
        blocks.append(
            LayoutBlock(
                index=index,
                block_type=block_type,
                bbox=bbox,
                angle=_coerce_float(raw_block.get("angle")),
                content=str(raw_block.get("content", "") or "").strip(),
                sub_type=str(raw_block.get("sub_type", "") or "").strip().lower(),
                merge_prev=bool(raw_block.get("merge_prev", False)),
                raw=dict(raw_block),
            )
        )
    return blocks


def build_crop_groups(
    page_stem: str,
    blocks: list[LayoutBlock],
    min_horizontal_overlap: float = 0.25,
    max_context_gap: float = 0.08,
) -> list[CropGroup]:
    groups: list[CropGroup] = []
    used_context_indexes: set[int] = set()

    for block in blocks:
        if block.block_type not in PRIMARY_TYPES:
            continue
        attached_blocks = [block]

        previous_index = block.index - 1
        while previous_index >= 0:
            previous_block = blocks[previous_index]
            if previous_block.block_type in PRIMARY_TYPES:
                break
            if previous_block.block_type not in CONTEXT_TYPES:
                break
            if previous_block.index in used_context_indexes:
                previous_index -= 1
                continue
            if not _can_attach_context(
                context_block=previous_block,
                primary_block=block,
                min_horizontal_overlap=min_horizontal_overlap,
                max_context_gap=max_context_gap,
            ):
                break
            attached_blocks.append(previous_block)
            used_context_indexes.add(previous_block.index)
            previous_index -= 1

        next_index = block.index + 1
        while next_index < len(blocks):
            next_block = blocks[next_index]
            if next_block.block_type in PRIMARY_TYPES:
                break
            if next_block.block_type not in CONTEXT_TYPES:
                break
            if next_block.index in used_context_indexes:
                next_index += 1
                continue
            if not _can_attach_context(
                context_block=next_block,
                primary_block=block,
                min_horizontal_overlap=min_horizontal_overlap,
                max_context_gap=max_context_gap,
            ):
                break
            attached_blocks.append(next_block)
            used_context_indexes.add(next_block.index)
            next_index += 1

        attached_blocks.sort(key=lambda item: item.index)
        groups.append(
            CropGroup(
                group_index=len(groups) + 1,
                page_stem=page_stem,
                primary_block=block,
                blocks=attached_blocks,
                merged_bbox=_merge_bboxes([item.bbox for item in attached_blocks]),
            )
        )

    return groups


def _can_attach_context(
    context_block: LayoutBlock,
    primary_block: LayoutBlock,
    min_horizontal_overlap: float,
    max_context_gap: float,
) -> bool:
    overlap = _horizontal_overlap_ratio(context_block.bbox, primary_block.bbox)
    if overlap < min_horizontal_overlap:
        return False

    gap = _vertical_gap(context_block.bbox, primary_block.bbox)
    if gap > max_context_gap:
        return False
    return True


def _horizontal_overlap_ratio(left_bbox: list[float], right_bbox: list[float]) -> float:
    left = max(left_bbox[0], right_bbox[0])
    right = min(left_bbox[2], right_bbox[2])
    overlap = max(0.0, right - left)
    left_width = max(1e-6, left_bbox[2] - left_bbox[0])
    right_width = max(1e-6, right_bbox[2] - right_bbox[0])
    return overlap / min(left_width, right_width)


def _vertical_gap(left_bbox: list[float], right_bbox: list[float]) -> float:
    if left_bbox[3] < right_bbox[1]:
        return right_bbox[1] - left_bbox[3]
    if right_bbox[3] < left_bbox[1]:
        return left_bbox[1] - right_bbox[3]
    return 0.0


def _merge_bboxes(bboxes: list[list[float]]) -> list[float]:
    if not bboxes:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        min(item[0] for item in bboxes),
        min(item[1] for item in bboxes),
        max(item[2] for item in bboxes),
        max(item[3] for item in bboxes),
    ]


def _normalize_bbox(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        return []
    coordinates: list[float] = []
    for item in value:
        try:
            coordinates.append(float(item))
        except (TypeError, ValueError):
            return []
    x0, y0, x1, y1 = coordinates
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    return [left, top, right, bottom]


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
