from __future__ import annotations

import re
from pathlib import Path

from src.schema import ImageTask

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_PAGE_CROP_PATTERN = re.compile(
    r"^(?P<image_name>.+)_(?P<page_tag>p_?\d+)_(?P<region_tag>r_?[0-9A-Za-z]+)_(?P<suffix>.+)$"
)


def _build_image_id(relative_path: Path) -> str:
    return relative_path.stem


def _parse_page_crop_metadata(image_id: str) -> tuple[str, str, bool]:
    structured_match = _PAGE_CROP_PATTERN.match(str(image_id or "").strip())
    if structured_match is not None:
        image_name = str(structured_match.group("image_name") or "").strip()
        page_tag = str(structured_match.group("page_tag") or "").strip()
        region_tag = str(structured_match.group("region_tag") or "").strip()
        suffix = str(structured_match.group("suffix") or "").strip()
        merge_order = region_tag[1:].removeprefix("_").strip()
        if all((image_name, page_tag, merge_order, suffix)):
            return f"{image_name}_{page_tag}", merge_order, True

    parts = str(image_id or "").split("_", 3)
    if len(parts) < 4:
        return "", "", False
    first, second, third, remainder = parts
    if not all((first, second, third, remainder)) or not third.isdigit():
        return "", "", False
    return f"{first}_{second}", third, True


def load_image_tasks(data_dir: Path) -> list[ImageTask]:
    if not data_dir.exists() or not data_dir.is_dir():
        return []

    tasks: list[ImageTask] = []
    seen_image_ids: dict[str, Path] = {}
    image_paths = [
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    image_paths.sort(
        key=lambda path: (
            path.name,
            path.relative_to(data_dir).as_posix(),
        )
    )

    for path in image_paths:
        relative_path = path.relative_to(data_dir)
        image_id = _build_image_id(relative_path)
        page_output_id, merge_order, is_page_crop = _parse_page_crop_metadata(image_id)
        previous_path = seen_image_ids.get(image_id)
        if previous_path is not None:
            raise ValueError(
                "Duplicate image file stem detected for output naming: "
                f"'{image_id}' from '{previous_path.as_posix()}' and '{relative_path.as_posix()}'"
            )
        seen_image_ids[image_id] = relative_path
        tasks.append(
            ImageTask(
                image_id=image_id,
                image_path=str(path),
                file_name=path.name,
                file_ext=path.suffix.lower(),
                page_output_id=page_output_id,
                merge_order=merge_order,
                is_page_crop=is_page_crop,
            )
        )

    return tasks
