from __future__ import annotations

from src.pipeline.normalizers import normalize_mineru_payload
from src.schema import ImageTask, ModelOutput


def test_normalize_mineru_payload_unwraps_nested_data_container() -> None:
    image_task = ImageTask(
        image_id="img-1",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    model_output = ModelOutput(
        image_id="img-1",
        model_name="mineru",
        success=True,
        raw_text="",
        parsed={
            "code": 0,
            "data": {
                "content_list_v2": [[
                    {
                        "type": "image",
                        "sub_type": "seal",
                        "bbox": [0, 0, 1000, 1000],
                        "content": {
                            "image_caption": ["某某公司印章"],
                            "img_path": "data/demo.png",
                        },
                    }
                ]]
            },
        },
    )

    _, document, label = normalize_mineru_payload(
        image_task=image_task,
        model_output=model_output,
    )

    assert not document.warnings
    assert len(document.blocks) == 1
    assert document.blocks[0].type == "image"
    assert document.blocks[0].sub_type == "seal"
    assert document.blocks[0].text == "某某公司印章"
    assert label is not None
    assert any(region.role == "seal" for region in label.ocr_regions)
