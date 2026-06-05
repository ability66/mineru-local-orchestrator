from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.preprocess.client import JsonLayoutClient
from src.preprocess.cropper import write_page_crops
from src.preprocess.grouping import build_crop_groups, normalize_layout_blocks

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop page images into visual blocks with bbox json layouts."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--layout-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/preprocess"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--padding-px", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = discover_page_images(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        limit=args.limit,
    )
    client = JsonLayoutClient(layout_dir=args.layout_dir)

    results: list[dict[str, Any]] = []
    max_workers = max(1, int(args.workers or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                process_page_image,
                image_path=image_path,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                client=client,
                padding_px=args.padding_px,
            ): image_path
            for image_path in image_paths
        }
        for future in as_completed(future_map):
            image_path = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "image_path": str(image_path),
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(result)
            status = "ok" if result.get("success") else "failed"
            detail = (
                f"{result.get('crop_count', 0)} crops"
                if result.get("success")
                else str(result.get("error", "unknown_error"))
            )
            print(f"[preprocess] {status} {image_path.name}: {detail}")

    succeeded = sum(1 for item in results if item.get("success"))
    failed = len(results) - succeeded
    print(
        f"[preprocess] finished total={len(results)} succeeded={succeeded} failed={failed}"
    )


def discover_page_images(
    data_dir: Path,
    output_dir: Path,
    limit: int | None = None,
) -> list[Path]:
    if not data_dir.exists() or not data_dir.is_dir():
        return []

    paths: list[Path] = []
    seen_stems: dict[str, Path] = {}
    resolved_output_dir = output_dir.resolve()
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if resolved_output_dir in path.resolve().parents:
            continue
        relative_path = path.relative_to(data_dir)
        previous_path = seen_stems.get(relative_path.stem)
        if previous_path is not None:
            raise ValueError(
                "Duplicate image file stem detected for preprocess output naming: "
                f"'{relative_path.stem}' from '{previous_path.as_posix()}' and '{relative_path.as_posix()}'"
            )
        seen_stems[relative_path.stem] = relative_path
        paths.append(path)
        if limit is not None and len(paths) >= limit:
            break
    return paths


def process_page_image(
    image_path: Path,
    data_dir: Path,
    output_dir: Path,
    client: JsonLayoutClient,
    padding_px: int,
) -> dict[str, Any]:
    relative_path = image_path.relative_to(data_dir)
    page_stem = relative_path.stem
    raw_blocks = client.fetch_blocks(image_path=image_path, relative_path=relative_path)
    layout_blocks = normalize_layout_blocks(raw_blocks)
    crop_groups = build_crop_groups(page_stem=page_stem, blocks=layout_blocks)
    page_output_dir = output_dir / page_stem
    manifest = write_page_crops(
        image_path=image_path,
        page_output_dir=page_output_dir,
        crop_groups=crop_groups,
        padding_px=padding_px,
    )
    return {
        "image_path": str(image_path),
        "success": True,
        "page_output_dir": str(page_output_dir),
        "crop_count": len(manifest.get("crops", [])),
    }


if __name__ == "__main__":
    main()
