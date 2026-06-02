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


def test_load_image_tasks_rejects_duplicate_file_stems(tmp_path: Path) -> None:
    first_dir = tmp_path / "a"
    second_dir = tmp_path / "b"
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / "figure1.png").write_bytes(b"first")
    (second_dir / "figure1.jpg").write_bytes(b"second")

    with pytest.raises(ValueError, match="Duplicate image file stem detected"):
        load_image_tasks(tmp_path)
