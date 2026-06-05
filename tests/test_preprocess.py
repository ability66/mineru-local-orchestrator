from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.preprocess.client import JsonLayoutClient
from src.preprocess.cropper import write_page_crops
from src.preprocess.grouping import build_crop_groups, normalize_layout_blocks


def test_build_crop_groups_merges_chart_caption_and_footnote() -> None:
    raw_blocks = [
        {
            "type": "header",
            "bbox": [0.418, 0.05, 0.849, 0.064],
            "angle": 0,
            "content": "header",
        },
        {
            "type": "text",
            "bbox": [0.146, 0.094, 0.853, 0.375],
            "angle": 0,
            "content": "paragraph",
        },
        {
            "type": "image_caption",
            "bbox": [0.279, 0.387, 0.719, 0.404],
            "angle": 0,
            "content": "chart caption",
        },
        {
            "type": "chart",
            "bbox": [0.22, 0.44, 0.825, 0.684],
            "angle": 0,
            "content": "| a | b |",
            "sub_type": "bar_line",
        },
        {
            "type": "image_footnote",
            "bbox": [0.146, 0.711, 0.603, 0.729],
            "angle": 0,
            "content": "footnote",
        },
        {
            "type": "text",
            "bbox": [0.146, 0.743, 0.854, 0.911],
            "angle": 0,
            "content": "tail paragraph",
        },
    ]

    blocks = normalize_layout_blocks(raw_blocks)
    groups = build_crop_groups(page_stem="changsha_page_48", blocks=blocks)

    assert len(groups) == 1
    group = groups[0]
    assert group.primary_block.block_type == "chart"
    assert group.primary_block.sub_type == "bar_line"
    assert [item.index for item in group.blocks] == [2, 3, 4]
    assert group.merged_bbox == [0.146, 0.387, 0.825, 0.729]


def test_write_page_crops_outputs_prefixed_files_and_manifest(tmp_path) -> None:
    image_path = tmp_path / "changsha_page_48.png"
    Image.new("RGB", (1000, 1000), color="white").save(image_path)

    blocks = normalize_layout_blocks(
        [
            {
                "type": "image_caption",
                "bbox": [0.279, 0.387, 0.719, 0.404],
                "angle": 0,
                "content": "chart caption",
            },
            {
                "type": "chart",
                "bbox": [0.22, 0.44, 0.825, 0.684],
                "angle": 0,
                "content": "| a | b |",
                "sub_type": "bar_line",
            },
            {
                "type": "image_footnote",
                "bbox": [0.146, 0.711, 0.603, 0.729],
                "angle": 0,
                "content": "footnote",
            },
        ]
    )
    groups = build_crop_groups(page_stem="changsha_page_48", blocks=blocks)

    manifest = write_page_crops(
        image_path=image_path,
        page_output_dir=tmp_path / "data" / "preprocess" / "changsha_page_48",
        crop_groups=groups,
        padding_px=0,
    )

    crop_path = tmp_path / "data" / "preprocess" / "changsha_page_48" / "changsha_page_48_001_chart_bar_line.png"
    assert crop_path.exists()
    with Image.open(crop_path) as crop_image:
        assert crop_image.size == (679, 342)

    manifest_path = tmp_path / "data" / "preprocess" / "changsha_page_48" / "manifest.json"
    assert manifest_path.exists()
    assert manifest["source_image"] == "changsha_page_48.png"
    assert manifest["crops"][0]["crop_file"] == "changsha_page_48_001_chart_bar_line.png"
    assert manifest["crops"][0]["merged_block_indexes"] == [0, 1, 2]
    persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted_manifest["crops"][0]["bbox_px"] == [146, 387, 825, 729]


def test_json_layout_client_reads_relative_layout_json(tmp_path) -> None:
    image_path = tmp_path / "pages" / "sample.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), color="white").save(image_path)

    layout_dir = tmp_path / "layouts"
    layout_path = layout_dir / "pages" / "sample.json"
    layout_path.parent.mkdir(parents=True)
    layout_path.write_text(
        json.dumps(
            [
                {
                    "type": "chart",
                    "bbox": [0.1, 0.1, 0.9, 0.9],
                    "angle": 0,
                    "content": "|a|b|",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    client = JsonLayoutClient(layout_dir=layout_dir)
    blocks = client.fetch_blocks(
        image_path=image_path,
        relative_path=Path("pages") / "sample.png",
    )

    assert len(blocks) == 1
    assert blocks[0]["type"] == "chart"
