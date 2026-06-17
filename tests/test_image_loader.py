from __future__ import annotations

from pathlib import Path

import pytest

from src.image_loader import load_image_tasks


def test_load_image_tasks_uses_file_stem_as_image_id(tmp_path: Path) -> None:
    image_path = tmp_path / "figure1.png"
    image_path.write_bytes(b"fake")

    tasks = load_image_tasks(tmp_path)

    assert len(tasks) == 1
    assert tasks[0].image_id == "figure1"
    assert tasks[0].file_name == "figure1.png"
    assert tasks[0].page_output_id == ""
    assert tasks[0].merge_order == ""
    assert tasks[0].is_page_crop is False


def test_load_image_tasks_parses_page_crop_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "doc1_02_003_flowchart_extra.jpg"
    image_path.write_bytes(b"fake")

    tasks = load_image_tasks(tmp_path)

    assert len(tasks) == 1
    assert tasks[0].image_id == "doc1_02_003_flowchart_extra"
    assert tasks[0].page_output_id == "doc1_02"
    assert tasks[0].merge_order == "003"
    assert tasks[0].is_page_crop is True


def test_load_image_tasks_parses_structured_page_crop_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "paper_section_a_p_007_r_012_chart.jpg"
    image_path.write_bytes(b"fake")

    tasks = load_image_tasks(tmp_path)

    assert len(tasks) == 1
    assert tasks[0].image_id == "paper_section_a_p_007_r_012_chart"
    assert tasks[0].page_output_id == "paper_section_a_p_007"
    assert tasks[0].merge_order == "012"
    assert tasks[0].is_page_crop is True


def test_load_image_tasks_parses_compact_page_crop_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "paper_section_a_p0007_r0012_chart.jpg"
    image_path.write_bytes(b"fake")

    tasks = load_image_tasks(tmp_path)

    assert len(tasks) == 1
    assert tasks[0].image_id == "paper_section_a_p0007_r0012_chart"
    assert tasks[0].page_output_id == "paper_section_a_p0007"
    assert tasks[0].merge_order == "0012"
    assert tasks[0].is_page_crop is True


def test_load_image_tasks_does_not_treat_non_numeric_third_segment_as_page_crop(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "table_markdown_flexible_demo.png"
    image_path.write_bytes(b"fake")

    tasks = load_image_tasks(tmp_path)

    assert len(tasks) == 1
    assert tasks[0].page_output_id == ""
    assert tasks[0].merge_order == ""
    assert tasks[0].is_page_crop is False


def test_load_image_tasks_rejects_duplicate_file_stems(tmp_path: Path) -> None:
    first_dir = tmp_path / "a"
    second_dir = tmp_path / "b"
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / "figure1.png").write_bytes(b"first")
    (second_dir / "figure1.jpg").write_bytes(b"second")

    with pytest.raises(ValueError, match="Duplicate image file stem detected"):
        load_image_tasks(tmp_path)
