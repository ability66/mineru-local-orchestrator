from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from src.preprocess.schema import CropGroup, CropManifestEntry


def write_page_crops(
    image_path: Path,
    page_output_dir: Path,
    crop_groups: list[CropGroup],
    padding_px: int = 16,
) -> dict[str, Any]:
    page_output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "source_image": image_path.name,
        "source_path": str(image_path),
        "crops": [],
    }

    with Image.open(image_path) as image:
        width, height = image.size
        manifest["page_width"] = width
        manifest["page_height"] = height

        for crop_group in crop_groups:
            bbox_px = _bbox_to_pixels(
                bbox=crop_group.merged_bbox,
                width=width,
                height=height,
                padding_px=padding_px,
            )
            crop_file_name = _build_crop_file_name(crop_group)
            crop_image = image.crop(tuple(bbox_px))
            crop_image.save(page_output_dir / crop_file_name)
            entry = CropManifestEntry(
                crop_file=crop_file_name,
                source_image=image_path.name,
                group_index=crop_group.group_index,
                main_type=crop_group.primary_block.block_type,
                sub_type=crop_group.primary_block.sub_type,
                merged_block_indexes=[item.index for item in crop_group.blocks],
                merged_block_types=[item.block_type for item in crop_group.blocks],
                bbox_norm=[round(item, 6) for item in crop_group.merged_bbox],
                bbox_px=bbox_px,
            )
            manifest["crops"].append(
                {
                    "crop_file": entry.crop_file,
                    "source_image": entry.source_image,
                    "group_index": entry.group_index,
                    "main_type": entry.main_type,
                    "sub_type": entry.sub_type,
                    "merged_block_indexes": entry.merged_block_indexes,
                    "merged_block_types": entry.merged_block_types,
                    "bbox_norm": entry.bbox_norm,
                    "bbox_px": entry.bbox_px,
                }
            )

    manifest_path = page_output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _build_crop_file_name(crop_group: CropGroup) -> str:
    page_prefix = _slugify(crop_group.page_stem)
    block_type = _slugify(crop_group.primary_block.block_type) or "block"
    sub_type = _slugify(crop_group.primary_block.sub_type)
    suffix = f"_{sub_type}" if sub_type else ""
    return f"{page_prefix}_{crop_group.group_index:03d}_{block_type}{suffix}.png"


def _bbox_to_pixels(
    bbox: list[float],
    width: int,
    height: int,
    padding_px: int,
) -> list[int]:
    x0, y0, x1, y1 = bbox
    if max(abs(item) for item in bbox) <= 1.5:
        left = int(x0 * width)
        top = int(y0 * height)
        right = int(x1 * width)
        bottom = int(y1 * height)
    else:
        left = int(x0)
        top = int(y0)
        right = int(x1)
        bottom = int(y1)

    left = max(0, left - max(0, padding_px))
    top = max(0, top - max(0, padding_px))
    right = min(width, right + max(0, padding_px))
    bottom = min(height, bottom + max(0, padding_px))

    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)
    return [left, top, right, bottom]


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", str(value or "").strip().lower())
    return normalized.strip("_")
