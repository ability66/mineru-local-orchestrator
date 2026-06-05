from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LayoutBlock:
    index: int
    block_type: str
    bbox: list[float]
    angle: float
    content: str
    sub_type: str = ""
    merge_prev: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class CropGroup:
    group_index: int
    page_stem: str
    primary_block: LayoutBlock
    blocks: list[LayoutBlock]
    merged_bbox: list[float]


@dataclass
class CropManifestEntry:
    crop_file: str
    source_image: str
    group_index: int
    main_type: str
    sub_type: str
    merged_block_indexes: list[int]
    merged_block_types: list[str]
    bbox_norm: list[float]
    bbox_px: list[int]
