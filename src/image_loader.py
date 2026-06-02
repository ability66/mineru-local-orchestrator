from __future__ import annotations

from pathlib import Path

from src.schema import ImageTask

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _build_image_id(relative_path: Path) -> str:
    return relative_path.stem


def load_image_tasks(data_dir: Path) -> list[ImageTask]:
    if not data_dir.exists() or not data_dir.is_dir():
        return []

    tasks: list[ImageTask] = []
    seen_image_ids: dict[str, Path] = {}
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        relative_path = path.relative_to(data_dir)
        image_id = _build_image_id(relative_path)
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
            )
        )

    return tasks
