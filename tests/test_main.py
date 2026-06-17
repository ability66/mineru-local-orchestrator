from __future__ import annotations

import argparse
import json
from typing import Any

from src.clients.base import BaseLocalClient
from src.main import (
    _build_seal_adjudication_candidates,
    _has_flowchart_path_hint,
    _pick_flowchart_reference_bundle,
    _pick_seal_reference_bundle,
    build_stage2_selection_record,
    process_image_task,
)
from src.schema import (
    CanonicalBlock,
    CanonicalDocument,
    CaptionStructured,
    ImageTask,
    ModelOutput,
    OcrRegion,
    ParsedLabel,
    SealSelectionDecision,
)


class StubClient(BaseLocalClient):
    def __init__(
        self,
        model_name: str,
        responses: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model_name=model_name, config=config)
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def _analyze_impl(
        self,
        image_task: ImageTask,
        prompt: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "image_id": image_task.image_id,
                "prompt": prompt,
                "context": context,
            }
        )
        if not self._responses:
            raise AssertionError(f"No stub response left for {self.model_name}")
        return self._responses.pop(0)


def _build_args() -> argparse.Namespace:
    return argparse.Namespace(retry=0, manual_compare_mode=False)


def _single_page_payload(blocks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return [blocks]


def _qwen_raw_payload(blocks: list[dict[str, Any]]) -> str:
    return json.dumps({"content_list_v2": [blocks]}, ensure_ascii=False)


def _table_block(
    block_id: str,
    table_markdown: str,
    block_type: str = "table",
    image_path: str = "data/demo.png",
) -> dict[str, Any]:
    content: dict[str, Any] = {"img_path": image_path}
    if block_type == "table":
        content["table_body"] = table_markdown
        content["table_caption"] = ["Markdown 表格"]
    else:
        content["content"] = table_markdown
        content["chart_caption"] = ["Markdown 表格"]
    block = {
        "block_id": block_id,
        "type": block_type,
        "bbox": [0, 0, 1000, 1000],
        "content": content,
    }
    return block


def _plain_chart_block(
    block_id: str,
    chart_text: str,
    image_path: str = "data/demo.png",
) -> dict[str, Any]:
    return {
        "block_id": block_id,
        "type": "chart",
        "sub_type": "bar_chart",
        "bbox": [0, 0, 1000, 1000],
        "content": {
            "img_path": image_path,
            "content": chart_text,
            "chart_caption": ["普通图表"],
        },
    }


def _blank_chart_block(
    block_id: str,
    image_path: str = "data/demo.png",
) -> dict[str, Any]:
    return {
        "block_id": block_id,
        "type": "chart",
        "sub_type": "scatter",
        "bbox": [0, 0, 1000, 1000],
        "content": {
            "img_path": image_path,
            "content": "",
        },
    }


def test_has_flowchart_path_hint_ignores_page_crop_type_suffix() -> None:
    crop_task = ImageTask(
        image_id="doc1_02_003_flowchart",
        image_path="data/doc1_02_003_flowchart.jpg",
        file_name="doc1_02_003_flowchart.jpg",
        file_ext=".jpg",
        page_output_id="doc1_02",
        merge_order="003",
        is_page_crop=True,
    )
    regular_task = ImageTask(
        image_id="demo-flowchart",
        image_path="data/demo-flowchart.png",
        file_name="demo-flowchart.png",
        file_ext=".png",
    )

    assert _has_flowchart_path_hint(crop_task) is False
    assert _has_flowchart_path_hint(regular_task) is True


def test_pick_seal_reference_bundle_prefers_richer_auxiliary_result() -> None:
    image_task = ImageTask(
        image_id="seal-1",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    mineru_document = CanonicalDocument(
        document_id="seal-1",
        source="mineru",
        blocks=[
            CanonicalBlock(
                block_id="m1",
                page_idx=0,
                order_index=1,
                type="image",
                bbox=[0, 0, 100, 100],
                text="",
                content={"img_path": "data/demo.png"},
                source="mineru",
                caption_structured=CaptionStructured(brief=""),
            )
        ],
    )
    paddle_document = CanonicalDocument(
        document_id="seal-1",
        source="paddle",
        blocks=[
            CanonicalBlock(
                block_id="p1",
                page_idx=0,
                order_index=1,
                type="image",
                sub_type="seal",
                bbox=[0, 0, 100, 100],
                text="某某公司印章",
                content={
                    "img_path": "data/demo.png",
                    "image_caption": ["某某公司印章"],
                },
                source="paddle",
                caption_structured=CaptionStructured(brief="某某公司印章"),
                ocr_regions=[
                    OcrRegion(role="seal", text="某某公司", confidence="high")
                ],
            )
        ],
    )
    glm_document = CanonicalDocument(
        document_id="seal-1",
        source="glm",
        blocks=[
            CanonicalBlock(
                block_id="g1",
                page_idx=0,
                order_index=1,
                type="image",
                sub_type="seal",
                bbox=[0, 0, 100, 100],
                text="",
                content={"img_path": "data/demo.png"},
                source="glm",
                caption_structured=CaptionStructured(brief=""),
            )
        ],
    )

    bundle, issues = _pick_seal_reference_bundle(
        image_task=image_task,
        mineru_document=mineru_document,
        auxiliary_bundles=[
            {
                "role": "glm",
                "output": ModelOutput(
                    image_id="seal-1",
                    model_name="glm-local",
                    success=True,
                    raw_text="{}",
                ),
                "document": glm_document,
                "label": ParsedLabel(image_type="seal"),
            },
            {
                "role": "paddle",
                "output": ModelOutput(
                    image_id="seal-1",
                    model_name="paddle-local",
                    success=True,
                    raw_text="{}",
                ),
                "document": paddle_document,
                "label": ParsedLabel(image_type="seal"),
            },
        ],
    )

    assert bundle is not None
    assert bundle["role"] == "paddle"
    assert len(issues) == 1
    assert issues[0].issue_type == "seal_type_disagreement"
    assert issues[0].candidate_payload is not None
    assert issues[0].candidate_payload["reference_model_role"] == "paddle"
    assert issues[0].candidate_payload["reference_model_name"] == "paddle-local"


def test_build_seal_adjudication_candidates_returns_full_text_candidates() -> None:
    image_task = ImageTask(
        image_id="seal-select-1",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    mineru_document = CanonicalDocument(
        document_id="seal-select-1",
        source="mineru",
        blocks=[
            CanonicalBlock(
                block_id="m1",
                page_idx=0,
                order_index=1,
                type="image",
                sub_type="seal",
                bbox=[0, 0, 999, 999],
                text="上海日轲电子有限公司",
                content={"img_path": "data/demo.png", "image_caption": ["上海日轲电子有限公司"]},
                source="mineru",
                caption_structured=CaptionStructured(brief="上海日轲电子有限公司"),
                ocr_regions=[OcrRegion(role="seal", text="上海日轲电子有限公司")],
            ),
            CanonicalBlock(
                block_id="m2",
                page_idx=0,
                order_index=2,
                type="paragraph",
                bbox=[194, 330, 543, 384],
                text="4541982082",
                content={"paragraph_content": [{"type": "text", "content": "4541982082"}]},
                source="mineru",
            ),
        ],
    )
    paddle_document = CanonicalDocument(
        document_id="seal-select-1",
        source="paddle",
        blocks=[
            CanonicalBlock(
                block_id="p1",
                page_idx=0,
                order_index=1,
                type="image",
                sub_type="seal",
                bbox=[0, 0, 999, 999],
                text="上海日轲电子有限公司",
                content={"img_path": "data/demo.png", "image_caption": ["上海日轲电子有限公司"]},
                source="paddle",
                caption_structured=CaptionStructured(brief="上海日轲电子有限公司"),
                ocr_regions=[OcrRegion(role="seal", text="上海日轲电子有限公司")],
            )
        ],
    )

    candidates, payload = _build_seal_adjudication_candidates(
        image_task=image_task,
        mineru_bundle={
            "output": ModelOutput(
                image_id="seal-select-1",
                model_name="mineru-local",
                success=True,
                raw_text="{}",
            ),
            "document": mineru_document,
            "label": ParsedLabel(image_type="seal", caption="上海日轲电子有限公司"),
        },
        auxiliary_bundles=[
            {
                "role": "paddle",
                "output": ModelOutput(
                    image_id="seal-select-1",
                    model_name="paddle-local",
                    success=True,
                    raw_text="{}",
                ),
                "document": paddle_document,
                "label": ParsedLabel(image_type="seal", caption="上海日轲电子有限公司"),
            }
        ],
    )

    assert len(candidates) == 2
    assert payload is not None
    mineru_payload = next(
        candidate for candidate in payload["candidates"] if candidate["candidate_id"] == "mineru"
    )
    assert mineru_payload["full_text"] == "上海日轲电子有限公司\n\n4541982082"
    assert mineru_payload["core_seal_text"] == "上海日轲电子有限公司"
    assert payload["comparisons"][0]["candidate_id"] == "paddle"
    assert payload["comparisons"][0]["issue_types"] == ["seal_ocr_conflict"]


def test_build_seal_adjudication_candidates_deduplicates_identical_candidates() -> None:
    image_task = ImageTask(
        image_id="seal-select-identical",
        image_path="data/demo.png",
        file_name="demo.png",
        file_ext=".png",
    )
    mineru_document = CanonicalDocument(
        document_id="seal-select-identical",
        source="mineru",
        blocks=[
            CanonicalBlock(
                block_id="m1",
                page_idx=0,
                order_index=1,
                type="image",
                sub_type="seal",
                bbox=[0, 0, 999, 999],
                text="上海日轲电子有限公司",
                content={"img_path": "data/demo.png", "image_caption": ["上海日轲电子有限公司"]},
                source="mineru",
                caption_structured=CaptionStructured(brief="上海日轲电子有限公司"),
                ocr_regions=[OcrRegion(role="seal", text="上海日轲电子有限公司")],
            )
        ],
    )
    paddle_document = CanonicalDocument(
        document_id="seal-select-identical",
        source="paddle",
        blocks=[
            CanonicalBlock(
                block_id="p1",
                page_idx=0,
                order_index=1,
                type="image",
                sub_type="seal",
                bbox=[0, 0, 999, 999],
                text="上海日轲电子有限公司",
                content={"img_path": "data/demo.png", "image_caption": ["上海日轲电子有限公司"]},
                source="paddle",
                caption_structured=CaptionStructured(brief="上海日轲电子有限公司"),
                ocr_regions=[OcrRegion(role="seal", text="上海日轲电子有限公司")],
            )
        ],
    )

    candidates, payload = _build_seal_adjudication_candidates(
        image_task=image_task,
        mineru_bundle={
            "output": ModelOutput(
                image_id="seal-select-identical",
                model_name="mineru-local",
                success=True,
                raw_text="{}",
            ),
            "document": mineru_document,
            "label": ParsedLabel(image_type="seal", caption="上海日轲电子有限公司"),
        },
        auxiliary_bundles=[
            {
                "role": "paddle",
                "output": ModelOutput(
                    image_id="seal-select-identical",
                    model_name="paddle-local",
                    success=True,
                    raw_text="{}",
                ),
                "document": paddle_document,
                "label": ParsedLabel(image_type="seal", caption="上海日轲电子有限公司"),
            }
        ],
    )

    assert len(candidates) == 1
    assert payload is None


def test_build_stage2_selection_record_keeps_selection_payload() -> None:
    record = build_stage2_selection_record(
        selection_payload={
            "image_id": "seal-select-2",
            "candidates": [{"candidate_id": "mineru"}, {"candidate_id": "paddle"}],
        },
        output=ModelOutput(
            image_id="seal-select-2",
            model_name="qwen-judge",
            success=True,
            raw_text='{"selected_candidate":"paddle"}',
            parsed={"usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}},
        ),
        selection_decision=SealSelectionDecision(
            selected_candidate="paddle",
            reason="paddle text is more complete",
            confidence="high",
        ),
        prompt="prompt body",
        mode="seal_adjudication",
    )

    assert record["mode"] == "seal_adjudication"
    assert record["selection_payload"]["candidates"][1]["candidate_id"] == "paddle"
    assert record["selection_decision"]["selected_candidate"] == "paddle"


def test_pick_flowchart_reference_bundle_ignores_non_qwen_candidates() -> None:
    image_task = ImageTask(
        image_id="flow-ref-1",
        image_path="data/flowchart_crops/figure1.png",
        file_name="figure1.png",
        file_ext=".png",
    )
    mineru_document = CanonicalDocument(
        document_id="flow-ref-1",
        source="mineru",
        blocks=[
            CanonicalBlock(
                block_id="m1",
                page_idx=0,
                order_index=1,
                type="chart",
                sub_type="flowchart",
                bbox=[0, 0, 1000, 1000],
                text="流程图",
                content={"img_path": "data/demo.png", "content": "flowchart TD\nA-->B"},
                source="mineru",
                caption_structured=CaptionStructured(brief="流程图"),
            )
        ],
    )
    qwen_document = CanonicalDocument(
        document_id="flow-ref-1",
        source="qwen",
        blocks=[
            CanonicalBlock(
                block_id="q1",
                page_idx=0,
                order_index=1,
                type="chart",
                sub_type="flowchart",
                bbox=[0, 0, 1000, 1000],
                text="流程图",
                content={"img_path": "data/demo.png", "content": "flowchart TD\nA-->B\nB-->C"},
                source="qwen",
                caption_structured=CaptionStructured(brief="流程图"),
            )
        ],
    )
    paddle_document = CanonicalDocument(
        document_id="flow-ref-1",
        source="paddle",
        blocks=[
            CanonicalBlock(
                block_id="p1",
                page_idx=0,
                order_index=1,
                type="chart",
                sub_type="flowchart",
                bbox=[0, 0, 1000, 1000],
                text="流程图",
                content={"img_path": "data/demo.png", "content": "flowchart TD\nX-->Y\nY-->Z"},
                source="paddle",
                caption_structured=CaptionStructured(brief="流程图"),
            )
        ],
    )

    bundle, issues = _pick_flowchart_reference_bundle(
        image_task=image_task,
        mineru_document=mineru_document,
        mineru_label=ParsedLabel(image_type="flowchart", caption="流程图"),
        candidate_bundles=[
            {
                "role": "paddle",
                "output": ModelOutput(
                    image_id="flow-ref-1",
                    model_name="paddle-local",
                    success=True,
                    raw_text="{}",
                ),
                "document": paddle_document,
                "label": ParsedLabel(image_type="flowchart", caption="流程图"),
            },
            {
                "role": "qwen",
                "output": ModelOutput(
                    image_id="flow-ref-1",
                    model_name="qwen-local",
                    success=True,
                    raw_text="{}",
                ),
                "document": qwen_document,
                "label": ParsedLabel(image_type="flowchart", caption="流程图"),
            },
        ],
    )

    assert bundle is not None
    assert bundle["role"] == "qwen"
    assert issues
    assert issues[0].candidate_payload is not None
    assert issues[0].candidate_payload["reference_model_role"] == "qwen"
    assert issues[0].candidate_payload["reference_model_name"] == "qwen-local"


def test_process_image_task_runs_qwen_first_pass_for_flowchart_branch(tmp_path) -> None:
    image_task = ImageTask(
        image_id="figure1",
        image_path="data/flowchart_crops/figure1.png",
        file_name="figure1.png",
        file_ext=".png",
    )
    mineru_block = {
        "block_id": "m1",
        "type": "chart",
        "sub_type": "flowchart",
        "bbox": [0, 0, 1000, 1000],
        "text": "流程图",
        "content": {
            "img_path": "data/flowchart_crops/figure1.png",
            "content": "flowchart TD\nA-->B",
        },
    }
    qwen_block = {
        "block_id": "q1",
        "type": "chart",
        "sub_type": "flowchart",
        "bbox": [0, 0, 1000, 1000],
        "text": "流程图",
        "content": {
            "img_path": "data/flowchart_crops/figure1.png",
            "content": "flowchart TD\nA-->B\nB-->C",
        },
    }
    mineru_client = StubClient(
        model_name="mineru-local",
        responses=[{"success": True, "parsed": _single_page_payload([mineru_block])}],
        config={"provider": "minerupro_local", "role": "mineru"},
    )
    paddle_client = StubClient(
        model_name="paddle-local",
        responses=[
            {
                "success": True,
                "parsed": _single_page_payload(
                    [
                        {
                            "block_id": "p1",
                            "type": "paragraph",
                            "bbox": [0, 0, 1000, 200],
                            "text": "审批通过",
                            "content": "审批通过",
                        }
                    ]
                ),
            }
        ],
        config={"provider": "paddle_local", "role": "paddle"},
    )
    glm_client = StubClient(
        model_name="glm-local",
        responses=[
            {
                "success": True,
                "raw_text": _qwen_raw_payload(
                    [
                        {
                            "block_id": "g1",
                            "type": "paragraph",
                            "bbox": [0, 220, 1000, 420],
                            "text": "人工复核",
                            "content": "人工复核",
                        }
                    ]
                ),
            }
        ],
        config={"provider": "glm_openai_compatible", "role": "glm"},
    )
    qwen_client = StubClient(
        model_name="qwen-local",
        responses=[
            {"success": True, "raw_text": _qwen_raw_payload([qwen_block])},
            {
                "success": True,
                "raw_text": json.dumps(
                    {
                        "decision": "use_qwen_fields",
                        "patch": {},
                        "reason": "qwen_flowchart_preferred_on_conflict",
                    },
                    ensure_ascii=False,
                ),
                "parsed": {
                    "choices": [{"finish_reason": "stop"}],
                    "_request_control": {
                        "mode": "flowchart_adjudication",
                        "thinking_mode": "disabled_requested",
                        "disable_thinking_requested": True,
                        "disable_thinking_applied": True,
                        "disable_thinking_fallback_used": False,
                    },
                },
            },
        ],
        config={"provider": "qwen_openai_compatible", "role": "judge"},
    )

    summary = process_image_task(
        image_task=image_task,
        args=_build_args(),
        mineru_client=mineru_client,
        paddle_client=paddle_client,
        glm_client=glm_client,
        qwen_client=qwen_client,
        recognition_prompt="recognition prompt",
        seal_adjudication_prompt="seal prompt",
        flowchart_adjudication_prompt="flow prompt",
        output_dir=tmp_path,
    )

    raw_qwen = json.loads(
        (tmp_path / "raw" / "qwen" / "figure1.json").read_text(encoding="utf-8")
    )
    raw_glm = json.loads(
        (tmp_path / "raw" / "glm" / "figure1.json").read_text(encoding="utf-8")
    )
    stage2_payload = json.loads(
        (tmp_path / "judge_stage2" / "figure1.json").read_text(encoding="utf-8")
    )
    assert summary["qwen_success"] is True
    assert raw_qwen["model_name"] == "qwen-local"
    assert raw_glm["model_name"] == "glm-local"
    assert len(qwen_client.calls) == 2
    assert len(glm_client.calls) == 1
    assert "审批通过" in qwen_client.calls[1]["context"]["issue_payload"]["ocr_reference_texts"]
    assert "人工复核" in qwen_client.calls[1]["context"]["issue_payload"]["ocr_reference_texts"]
    assert (
        qwen_client.calls[1]["context"]["issue_payload"]["ocr_reference_models"]
        == ["paddle-local", "glm-local"]
    )
    assert (
        qwen_client.calls[1]["context"]["issue_payload"]["ocr_reference_sources"][0][
            "reference_model_role"
        ]
        == "paddle"
    )
    assert (
        qwen_client.calls[1]["context"]["issue_payload"]["ocr_reference_sources"][1][
            "reference_model_role"
        ]
        == "glm"
    )
    assert stage2_payload["records"][0]["thinking_mode"] == "disabled_requested"


def test_process_image_task_keeps_non_flowchart_branch_without_qwen_first_pass(
    tmp_path,
) -> None:
    image_task = ImageTask(
        image_id="seal-plain",
        image_path="data/stamp/seal-plain.png",
        file_name="seal-plain.png",
        file_ext=".png",
    )
    seal_block = {
        "block_id": "s1",
        "type": "image",
        "sub_type": "seal",
        "bbox": [0, 0, 1000, 1000],
        "text": "上海日轲电子有限公司",
        "content": {
            "img_path": "data/stamp/seal-plain.png",
            "image_caption": ["上海日轲电子有限公司"],
        },
        "ocr_regions": [{"role": "seal", "text": "上海日轲电子有限公司"}],
    }
    mineru_client = StubClient(
        model_name="mineru-local",
        responses=[{"success": True, "parsed": _single_page_payload([seal_block])}],
        config={"provider": "minerupro_local", "role": "mineru"},
    )
    paddle_client = StubClient(
        model_name="paddle-local",
        responses=[{"success": True, "parsed": _single_page_payload([seal_block])}],
        config={"provider": "paddle_local", "role": "paddle"},
    )
    glm_client = StubClient(
        model_name="glm-local",
        responses=[{"success": True, "parsed": _single_page_payload([seal_block])}],
        config={"provider": "glm_openai_compatible", "role": "glm"},
    )
    qwen_client = StubClient(
        model_name="qwen-local",
        responses=[
            {
                "success": True,
                "raw_text": json.dumps(
                    {
                        "selected_candidate": "mineru",
                        "reason": "mineru text is correct",
                        "confidence": "high",
                    },
                    ensure_ascii=False,
                ),
            }
        ],
        config={"provider": "qwen_openai_compatible", "role": "judge"},
    )

    summary = process_image_task(
        image_task=image_task,
        args=_build_args(),
        mineru_client=mineru_client,
        paddle_client=paddle_client,
        glm_client=glm_client,
        qwen_client=qwen_client,
        recognition_prompt="recognition prompt",
        seal_adjudication_prompt="seal prompt",
        flowchart_adjudication_prompt="flow prompt",
        output_dir=tmp_path,
    )

    raw_qwen = json.loads(
        (tmp_path / "raw" / "qwen" / "seal-plain.json").read_text(encoding="utf-8")
    )
    assert summary["qwen_success"] is True
    assert raw_qwen["model_name"] == "qwen-local"
    assert len(qwen_client.calls) == 1
    assert qwen_client.calls[0]["context"]["mode"] == "seal_adjudication"
    assert len(glm_client.calls) == 1


def test_process_image_task_auto_accepts_high_consensus_table_branch(
    tmp_path,
) -> None:
    image_task = ImageTask(
        image_id="table-accept",
        image_path="data/demo.png",
        file_name="table-accept.png",
        file_ext=".png",
    )
    markdown_table = "| 指标 | 值 |\n| --- | --- |\n| 增长率 | 12% |"
    mineru_client = StubClient(
        model_name="mineru-local",
        responses=[{"success": True, "parsed": _single_page_payload([_table_block("m1", markdown_table)])}],
        config={"provider": "minerupro_local", "role": "mineru"},
    )
    paddle_client = StubClient(
        model_name="paddle-local",
        responses=[{"success": True, "parsed": _single_page_payload([_table_block("p1", markdown_table)])}],
        config={"provider": "paddle_local", "role": "paddle"},
    )
    glm_client = StubClient(
        model_name="glm-local",
        responses=[{"success": True, "parsed": _single_page_payload([_table_block("g1", markdown_table)])}],
        config={"provider": "glm_openai_compatible", "role": "glm"},
    )
    qwen_client = StubClient(
        model_name="qwen-local",
        responses=[],
        config={"provider": "qwen_openai_compatible", "role": "judge"},
    )

    process_image_task(
        image_task=image_task,
        args=_build_args(),
        mineru_client=mineru_client,
        paddle_client=paddle_client,
        glm_client=glm_client,
        qwen_client=qwen_client,
        recognition_prompt="recognition prompt",
        seal_adjudication_prompt="seal prompt",
        flowchart_adjudication_prompt="flow prompt",
        output_dir=tmp_path,
        table_adjudication_prompt="table prompt",
    )

    artifact = json.loads(
        (tmp_path / "final" / "table-accept_artifact.json").read_text(encoding="utf-8")
    )
    assert len(qwen_client.calls) == 0
    assert artifact["consensus"]["decision"] == "accepted"
    assert artifact["final_document"]["raw_metadata"]["table_analysis"]["candidate_count"] >= 2
    assert artifact["final_document"]["raw_metadata"]["table_analysis"]["stable_consensus"] is True
    assert artifact["final_document"]["raw_metadata"]["table_analysis"]["artifact_reference_included"] is True


def test_process_image_task_triggers_qwen_for_divergent_table_branch(
    tmp_path,
) -> None:
    image_task = ImageTask(
        image_id="table-review",
        image_path="data/demo.png",
        file_name="table-review.png",
        file_ext=".png",
    )
    mineru_table = "| 指标 | 值 |\n| --- | --- |\n| 增长率 | 12% |"
    paddle_table = "| 地区 | Q1 | Q2 |\n| --- | --- | --- |\n| 华东 | 10 | 20 |"
    glm_table = "| 公式 | 值 |\n| --- | --- |\n| $x^2$ | 5 |"
    mineru_client = StubClient(
        model_name="mineru-local",
        responses=[{"success": True, "parsed": _single_page_payload([_table_block("m1", mineru_table)])}],
        config={"provider": "minerupro_local", "role": "mineru"},
    )
    paddle_client = StubClient(
        model_name="paddle-local",
        responses=[{"success": True, "parsed": _single_page_payload([_table_block("p1", paddle_table)])}],
        config={"provider": "paddle_local", "role": "paddle"},
    )
    glm_client = StubClient(
        model_name="glm-local",
        responses=[{"success": True, "parsed": _single_page_payload([_table_block("g1", glm_table)])}],
        config={"provider": "glm_openai_compatible", "role": "glm"},
    )
    qwen_client = StubClient(
        model_name="qwen-local",
        responses=[
            {
                "success": True,
                "raw_text": json.dumps(
                    {
                        "issue_id": "table-m1",
                        "target_block_id": "m1",
                        "decision": "keep_mineru",
                        "patch": {},
                        "reason": "structures diverge too much",
                    },
                    ensure_ascii=False,
                ),
            }
        ],
        config={"provider": "qwen_openai_compatible", "role": "judge"},
    )

    process_image_task(
        image_task=image_task,
        args=_build_args(),
        mineru_client=mineru_client,
        paddle_client=paddle_client,
        glm_client=glm_client,
        qwen_client=qwen_client,
        recognition_prompt="recognition prompt",
        seal_adjudication_prompt="seal prompt",
        flowchart_adjudication_prompt="flow prompt",
        output_dir=tmp_path,
        table_adjudication_prompt="table prompt",
    )

    assert len(qwen_client.calls) == 1
    assert qwen_client.calls[0]["context"]["mode"] == "table_adjudication"
    assert "pairwise_matrix" in qwen_client.calls[0]["context"]["issue_payload"]
    artifact = json.loads(
        (tmp_path / "final" / "table-review_artifact.json").read_text(encoding="utf-8")
    )
    assert "only one parsable label" not in artifact["consensus"]["reasons"]
    assert "single model result cannot be auto-accepted" not in artifact["consensus"]["reasons"]
    assert (
        "table candidates do not form stable consensus"
        in artifact["consensus"]["reasons"]
    )
    assert artifact["final_document"]["raw_metadata"]["table_analysis"]["candidate_count"] >= 2
    assert artifact["final_document"]["raw_metadata"]["table_analysis"]["requires_qwen"] is True


def test_process_image_task_always_triggers_qwen_for_markdown_chart_table_branch(
    tmp_path,
) -> None:
    image_task = ImageTask(
        image_id="table-fallback",
        image_path="data/demo.png",
        file_name="table-fallback.png",
        file_ext=".png",
    )
    markdown_table = "|指标|值|\n|---|---|\n|增长率|12%|"
    mineru_client = StubClient(
        model_name="mineru-local",
        responses=[{"success": True, "parsed": _single_page_payload([_table_block("m1", markdown_table, block_type="chart")])}],
        config={"provider": "minerupro_local", "role": "mineru"},
    )
    paddle_client = StubClient(
        model_name="paddle-local",
        responses=[{"success": True, "parsed": _single_page_payload([_table_block("p1", markdown_table, block_type="chart")])}],
        config={"provider": "paddle_local", "role": "paddle"},
    )
    glm_client = StubClient(
        model_name="glm-local",
        responses=[{"success": True, "parsed": _single_page_payload([_table_block("g1", markdown_table, block_type="chart")])}],
        config={"provider": "glm_openai_compatible", "role": "glm"},
    )
    qwen_client = StubClient(
        model_name="qwen-local",
        responses=[
            {
                "success": True,
                "raw_text": _qwen_raw_payload(
                    [_table_block("q1", markdown_table, block_type="table")]
                ),
            },
            {
                "success": True,
                "raw_text": json.dumps(
                    {
                        "issue_id": "table-m1",
                        "target_block_id": "m1",
                        "decision": "merge",
                        "patch": {
                            "type": "table",
                            "content": {
                                "table_body": "|指标|值|\n|---|---|\n|增长率|12%|\n|同比|8%|",
                                "table_caption": ["Qwen 终裁表格"],
                            },
                        },
                        "reason": "chart table second-pass adjudication",
                    },
                    ensure_ascii=False,
                ),
            }
        ],
        config={"provider": "qwen_openai_compatible", "role": "judge"},
    )

    process_image_task(
        image_task=image_task,
        args=_build_args(),
        mineru_client=mineru_client,
        paddle_client=paddle_client,
        glm_client=glm_client,
        qwen_client=qwen_client,
        recognition_prompt="recognition prompt",
        seal_adjudication_prompt="seal prompt",
        flowchart_adjudication_prompt="flow prompt",
        output_dir=tmp_path,
        table_adjudication_prompt="table prompt",
    )

    artifact = json.loads(
        (tmp_path / "final" / "table-fallback_artifact.json").read_text(encoding="utf-8")
    )
    final_output = json.loads(
        (tmp_path / "final" / "table-fallback.json").read_text(encoding="utf-8")
    )
    assert len(qwen_client.calls) == 2
    assert qwen_client.calls[1]["context"]["mode"] == "table_adjudication"
    assert qwen_client.calls[1]["context"]["issue_payload"]["review_mode"] == "chart_table_second_pass"
    assert qwen_client.calls[1]["context"]["issue_payload"]["must_output_final_table"] is True
    assert any(
        candidate["candidate_id"] == "qwen"
        for candidate in qwen_client.calls[1]["context"]["issue_payload"]["candidates"]
    )
    assert final_output["model_name"] == "qwen-local"
    assert artifact["final_document"]["blocks"][0]["type"] == "table"
    assert (
        artifact["final_document"]["blocks"][0]["content"]["table_body"]
        == "|指标|值|\n|---|---|\n|增长率|12%|\n|同比|8%|"
    )
    assert artifact["final_document"]["blocks"][0]["content"]["table_caption"] == [
        "Qwen 终裁表格"
    ]
    assert artifact["final_document"]["raw_metadata"]["selected_output_role"] == "qwen"
    assert (
        artifact["final_document"]["raw_metadata"]["selected_by"]
        == "chart_table_second_pass_adjudication"
    )
    assert artifact["final_label"]["caption"] == "Qwen 终裁表格"
    assert artifact["consensus"]["decision"] == "accepted"
    assert (
        "chart table resolved by qwen second-stage adjudication"
        in artifact["consensus"]["reasons"]
    )
    assert "only one parsable label" not in artifact["consensus"]["reasons"]
    assert artifact["final_document"]["raw_metadata"]["table_analysis"]["fallback"] is False
    assert artifact["final_document"]["raw_metadata"]["table_analysis"]["stable_consensus"] is True
    assert (
        artifact["final_document"]["raw_metadata"]["table_analysis"]["forced_second_pass"]
        is True
    )
    assert (
        artifact["final_document"]["raw_metadata"]["table_analysis"]["branch_mode"]
        == "chart_table"
    )
    assert artifact["final_document"]["raw_metadata"]["table_analysis"]["requires_qwen"] is True


def test_process_image_task_always_triggers_qwen_for_plain_chart_branch(
    tmp_path,
) -> None:
    image_task = ImageTask(
        image_id="plain-chart-force-qwen",
        image_path="data/demo.png",
        file_name="plain-chart-force-qwen.png",
        file_ext=".png",
    )
    chart_text = "地区 华东 10 华南 20 华北 15"
    mineru_client = StubClient(
        model_name="mineru-local",
        responses=[{"success": True, "parsed": _single_page_payload([_plain_chart_block("m1", chart_text)])}],
        config={"provider": "minerupro_local", "role": "mineru"},
    )
    paddle_client = StubClient(
        model_name="paddle-local",
        responses=[{"success": True, "parsed": _single_page_payload([_plain_chart_block("p1", chart_text)])}],
        config={"provider": "paddle_local", "role": "paddle"},
    )
    glm_client = StubClient(
        model_name="glm-local",
        responses=[{"success": True, "parsed": _single_page_payload([_plain_chart_block("g1", chart_text)])}],
        config={"provider": "glm_openai_compatible", "role": "glm"},
    )
    qwen_client = StubClient(
        model_name="qwen-local",
        responses=[
            {
                "success": True,
                "raw_text": _qwen_raw_payload(
                    [
                        _table_block(
                            "q1",
                            "| 地区 | 数值 |\n| --- | --- |\n| 华东 | 10 |\n| 华南 | 20 |\n| 华北 | 15 |",
                            block_type="table",
                        )
                    ]
                ),
            },
            {
                "success": True,
                "raw_text": json.dumps(
                    {
                        "issue_id": "table-m1",
                        "target_block_id": "m1",
                        "decision": "merge",
                        "patch": {
                            "type": "table",
                            "content": {
                                "table_body": "| 地区 | 数值 |\n| --- | --- |\n| 华东 | 10 |\n| 华南 | 20 |\n| 华北 | 15 |",
                                "table_caption": ["Qwen 重建表格"],
                            },
                        },
                        "reason": "chart table second-pass adjudication",
                    },
                    ensure_ascii=False,
                ),
            }
        ],
        config={"provider": "qwen_openai_compatible", "role": "judge"},
    )

    process_image_task(
        image_task=image_task,
        args=_build_args(),
        mineru_client=mineru_client,
        paddle_client=paddle_client,
        glm_client=glm_client,
        qwen_client=qwen_client,
        recognition_prompt="recognition prompt",
        seal_adjudication_prompt="seal prompt",
        flowchart_adjudication_prompt="flow prompt",
        output_dir=tmp_path,
        table_adjudication_prompt="table prompt",
    )

    artifact = json.loads(
        (tmp_path / "final" / "plain-chart-force-qwen_artifact.json").read_text(
            encoding="utf-8"
        )
    )
    final_output = json.loads(
        (tmp_path / "final" / "plain-chart-force-qwen.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(qwen_client.calls) == 2
    assert qwen_client.calls[1]["context"]["mode"] == "table_adjudication"
    assert qwen_client.calls[1]["context"]["issue_payload"]["review_mode"] == "chart_table_second_pass"
    assert qwen_client.calls[1]["context"]["issue_payload"]["must_output_final_table"] is True
    assert qwen_client.calls[1]["context"]["issue_payload"]["candidates"][0]["table_format"] == "none"
    assert qwen_client.calls[1]["context"]["issue_payload"]["candidates"][0]["table_content"] == chart_text
    assert any(
        candidate["candidate_id"] == "qwen"
        and candidate["table_format"] == "markdown"
        for candidate in qwen_client.calls[1]["context"]["issue_payload"]["candidates"]
    )
    assert final_output["model_name"] == "qwen-local"
    assert artifact["final_document"]["blocks"][0]["type"] == "table"
    assert (
        artifact["final_document"]["blocks"][0]["content"]["table_body"]
        == "| 地区 | 数值 |\n| --- | --- |\n| 华东 | 10 |\n| 华南 | 20 |\n| 华北 | 15 |"
    )
    assert artifact["final_document"]["blocks"][0]["content"]["table_caption"] == [
        "Qwen 重建表格"
    ]
    assert artifact["final_document"]["raw_metadata"]["selected_output_role"] == "qwen"
    assert (
        artifact["final_document"]["raw_metadata"]["selected_by"]
        == "chart_table_second_pass_adjudication"
    )
    assert artifact["final_label"]["caption"] == "Qwen 重建表格"
    assert artifact["consensus"]["decision"] == "accepted"
    assert (
        "chart table resolved by qwen second-stage adjudication"
        in artifact["consensus"]["reasons"]
    )
    assert (
        artifact["final_document"]["raw_metadata"]["table_analysis"]["forced_second_pass"]
        is True
    )
    assert (
        artifact["final_document"]["raw_metadata"]["table_analysis"]["branch_mode"]
        == "chart_table"
    )
    assert artifact["final_document"]["raw_metadata"]["table_analysis"]["fallback"] is True
    assert (
        artifact["final_document"]["raw_metadata"]["table_analysis"]["fallback_reason"]
        == "table_consensus_unavailable"
    )
    assert artifact["final_document"]["raw_metadata"]["table_analysis"]["requires_qwen"] is True


def test_process_image_task_reviews_plain_chart_when_qwen_patch_is_invalid(
    tmp_path,
) -> None:
    image_task = ImageTask(
        image_id="plain-chart-invalid-qwen",
        image_path="data/demo.png",
        file_name="plain-chart-invalid-qwen.png",
        file_ext=".png",
    )
    chart_text = "地区 华东 10 华南 20 华北 15"
    mineru_client = StubClient(
        model_name="mineru-local",
        responses=[
            {
                "success": True,
                "parsed": _single_page_payload([_plain_chart_block("m1", chart_text)]),
            }
        ],
        config={"provider": "minerupro_local", "role": "mineru"},
    )
    paddle_client = StubClient(
        model_name="paddle-local",
        responses=[
            {
                "success": True,
                "parsed": _single_page_payload([_plain_chart_block("p1", chart_text)]),
            }
        ],
        config={"provider": "paddle_local", "role": "paddle"},
    )
    glm_client = StubClient(
        model_name="glm-local",
        responses=[
            {
                "success": True,
                "parsed": _single_page_payload([_plain_chart_block("g1", chart_text)]),
            }
        ],
        config={"provider": "glm_openai_compatible", "role": "glm"},
    )
    qwen_client = StubClient(
        model_name="qwen-local",
        responses=[
            {
                "success": True,
                "raw_text": _qwen_raw_payload(
                    [
                        _table_block(
                            "q1",
                            "| 地区 | 数值 |\n| --- | --- |\n| 华东 | 10 |\n| 华南 | 20 |\n| 华北 | 15 |",
                            block_type="table",
                        )
                    ]
                ),
            },
            {"success": True, "raw_text": "not-json"},
        ],
        config={"provider": "qwen_openai_compatible", "role": "judge"},
    )

    process_image_task(
        image_task=image_task,
        args=_build_args(),
        mineru_client=mineru_client,
        paddle_client=paddle_client,
        glm_client=glm_client,
        qwen_client=qwen_client,
        recognition_prompt="recognition prompt",
        seal_adjudication_prompt="seal prompt",
        flowchart_adjudication_prompt="flow prompt",
        output_dir=tmp_path,
        table_adjudication_prompt="table prompt",
    )

    artifact = json.loads(
        (tmp_path / "final" / "plain-chart-invalid-qwen_artifact.json").read_text(
            encoding="utf-8"
        )
    )
    final_output = json.loads(
        (tmp_path / "final" / "plain-chart-invalid-qwen.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(qwen_client.calls) == 2
    assert qwen_client.calls[1]["context"]["mode"] == "table_adjudication"
    assert any(
        candidate["candidate_id"] == "qwen"
        for candidate in qwen_client.calls[1]["context"]["issue_payload"]["candidates"]
    )
    assert artifact["consensus"]["decision"] == "review"
    assert artifact["review_required"] is True
    assert (
        "chart table second-stage adjudication did not produce an adoptable final table"
        in artifact["consensus"]["reasons"]
    )
    assert artifact["patch_decisions"][0]["decision"] == "keep_mineru"
    assert artifact["patch_decisions"][0]["reason"] == "llm_patch_invalid_json"
    assert artifact["final_document"]["blocks"][0]["type"] == "chart"
    assert artifact["final_document"]["raw_metadata"].get("selected_by") != (
        "chart_table_second_pass_adjudication"
    )
    assert final_output["model_name"] == "mineru-local"


def test_process_image_task_builds_multiple_chart_table_issues_for_multi_chart_page(
    tmp_path,
) -> None:
    image_task = ImageTask(
        image_id="multi-chart-page",
        image_path="data/demo.png",
        file_name="multi-chart-page.png",
        file_ext=".png",
    )
    left_chart = _plain_chart_block("m1", "left chart raw text")
    left_chart["bbox"] = [0, 0, 480, 840]
    left_chart["content"]["chart_caption"] = ["(a) 5-way 10-shot"]
    right_chart = _plain_chart_block("m2", "right chart raw text")
    right_chart["bbox"] = [520, 0, 1000, 840]
    right_chart["content"]["chart_caption"] = ["(b) 5-way full-shot"]

    qwen_left = _table_block(
        "q1",
        "| Sessions | A |\n| --- | --- |\n| 1 | 62 |\n| 2 | 57 |",
        block_type="table",
    )
    qwen_left["bbox"] = [0, 0, 480, 840]
    qwen_right = _table_block(
        "q2",
        "| Sessions | B |\n| --- | --- |\n| 1 | 61 |\n| 2 | 59 |",
        block_type="table",
    )
    qwen_right["bbox"] = [520, 0, 1000, 840]

    mineru_client = StubClient(
        model_name="mineru-local",
        responses=[
            {"success": True, "parsed": _single_page_payload([left_chart, right_chart])}
        ],
        config={"provider": "minerupro_local", "role": "mineru"},
    )
    paddle_client = StubClient(
        model_name="paddle-local",
        responses=[
            {
                "success": True,
                "parsed": _single_page_payload(
                    [
                        {
                            "block_id": "p1",
                            "type": "paragraph",
                            "bbox": [0, 860, 1000, 920],
                            "text": "legend text",
                            "content": {"paragraph_content": [{"type": "text", "content": "legend text"}]},
                        }
                    ]
                ),
            }
        ],
        config={"provider": "paddle_local", "role": "paddle"},
    )
    glm_client = StubClient(
        model_name="glm-local",
        responses=[
            {
                "success": True,
                "parsed": _single_page_payload(
                    [
                        {
                            "block_id": "g1",
                            "type": "paragraph",
                            "bbox": [0, 930, 1000, 980],
                            "text": "extra note",
                            "content": {"paragraph_content": [{"type": "text", "content": "extra note"}]},
                        }
                    ]
                ),
            }
        ],
        config={"provider": "glm_openai_compatible", "role": "glm"},
    )
    qwen_client = StubClient(
        model_name="qwen-local",
        responses=[
            {"success": True, "raw_text": _qwen_raw_payload([qwen_left, qwen_right])},
            {
                "success": True,
                "raw_text": json.dumps(
                    {
                        "issue_id": "table-m1",
                        "target_block_id": "m1",
                        "decision": "merge",
                        "patch": {
                            "type": "table",
                            "content": {
                                "table_body": "| Sessions | Left |\n| --- | --- |\n| 1 | 62 |\n| 2 | 57 |",
                                "table_caption": ["(a) 5-way 10-shot"],
                            },
                        },
                        "reason": "left chart reconstructed",
                    },
                    ensure_ascii=False,
                ),
            },
            {
                "success": True,
                "raw_text": json.dumps(
                    {
                        "issue_id": "table-m2",
                        "target_block_id": "m2",
                        "decision": "merge",
                        "patch": {
                            "type": "table",
                            "content": {
                                "table_body": "| Sessions | Right |\n| --- | --- |\n| 1 | 61 |\n| 2 | 59 |",
                                "table_caption": ["(b) 5-way full-shot"],
                            },
                        },
                        "reason": "right chart reconstructed",
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        config={"provider": "qwen_openai_compatible", "role": "judge"},
    )

    process_image_task(
        image_task=image_task,
        args=_build_args(),
        mineru_client=mineru_client,
        paddle_client=paddle_client,
        glm_client=glm_client,
        qwen_client=qwen_client,
        recognition_prompt="recognition prompt",
        seal_adjudication_prompt="seal prompt",
        flowchart_adjudication_prompt="flow prompt",
        output_dir=tmp_path,
        table_adjudication_prompt="table prompt",
    )

    artifact = json.loads(
        (tmp_path / "final" / "multi-chart-page_artifact.json").read_text(
            encoding="utf-8"
        )
    )
    stage2_payload = json.loads(
        (tmp_path / "judge_stage2" / "multi-chart-page.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(qwen_client.calls) == 3
    assert qwen_client.calls[1]["context"]["mode"] == "table_adjudication"
    assert qwen_client.calls[2]["context"]["mode"] == "table_adjudication"
    assert {
        qwen_client.calls[1]["context"]["issue_payload"]["target_block_id"],
        qwen_client.calls[2]["context"]["issue_payload"]["target_block_id"],
    } == {"m1", "m2"}
    assert all(
        any(
            candidate["candidate_id"] == "qwen"
            for candidate in call["context"]["issue_payload"]["candidates"]
        )
        for call in qwen_client.calls[1:]
    )
    assert stage2_payload["record_count"] == 2
    assert artifact["final_document"]["raw_metadata"]["table_analysis"]["stage2_issue_count"] == 2
    assert len(artifact["patch_decisions"]) == 2

    blocks = {
        block["block_id"]: block
        for block in artifact["final_document"]["blocks"]
        if block["block_id"] in {"m1", "m2"}
    }
    assert blocks["m1"]["content"]["table_body"] == (
        "| Sessions | Left |\n| --- | --- |\n| 1 | 62 |\n| 2 | 57 |"
    )
    assert blocks["m2"]["content"]["table_body"] == (
        "| Sessions | Right |\n| --- | --- |\n| 1 | 61 |\n| 2 | 59 |"
    )


def test_process_image_task_triggers_qwen_when_mineru_chart_candidate_is_empty(
    tmp_path,
) -> None:
    image_task = ImageTask(
        image_id="blank-chart-force-qwen",
        image_path="data/demo.png",
        file_name="blank-chart-force-qwen.png",
        file_ext=".png",
    )
    paddle_chart_text = "地区 华东 10 华南 20 华北 15"
    mineru_client = StubClient(
        model_name="mineru-local",
        responses=[{"success": True, "parsed": _single_page_payload([_blank_chart_block("m1")])}],
        config={"provider": "minerupro_local", "role": "mineru"},
    )
    paddle_client = StubClient(
        model_name="paddle-local",
        responses=[{"success": True, "parsed": _single_page_payload([_plain_chart_block("p1", paddle_chart_text)])}],
        config={"provider": "paddle_local", "role": "paddle"},
    )
    glm_client = StubClient(
        model_name="glm-local",
        responses=[{"success": True, "parsed": _single_page_payload([_blank_chart_block("g1")])}],
        config={"provider": "glm_openai_compatible", "role": "glm"},
    )
    qwen_client = StubClient(
        model_name="qwen-local",
        responses=[
            {
                "success": True,
                "raw_text": _qwen_raw_payload(
                    [
                        _table_block(
                            "q1",
                            "| 地区 | 数值 |\n| --- | --- |\n| 华东 | 10 |\n| 华南 | 20 |\n| 华北 | 15 |",
                            block_type="table",
                        )
                    ]
                ),
            },
            {
                "success": True,
                "raw_text": json.dumps(
                    {
                        "issue_id": "table-m1",
                        "target_block_id": "m1",
                        "decision": "merge",
                        "patch": {
                            "type": "table",
                            "content": {
                                "table_body": "| 地区 | 数值 |\n| --- | --- |\n| 华东 | 10 |\n| 华南 | 20 |\n| 华北 | 15 |",
                                "table_caption": ["Paddle 重建表格"],
                            },
                        },
                        "reason": "chart table second-pass adjudication",
                    },
                    ensure_ascii=False,
                ),
            }
        ],
        config={"provider": "qwen_openai_compatible", "role": "judge"},
    )

    process_image_task(
        image_task=image_task,
        args=_build_args(),
        mineru_client=mineru_client,
        paddle_client=paddle_client,
        glm_client=glm_client,
        qwen_client=qwen_client,
        recognition_prompt="recognition prompt",
        seal_adjudication_prompt="seal prompt",
        flowchart_adjudication_prompt="flow prompt",
        output_dir=tmp_path,
        table_adjudication_prompt="table prompt",
    )

    artifact = json.loads(
        (tmp_path / "final" / "blank-chart-force-qwen_artifact.json").read_text(
            encoding="utf-8"
        )
    )
    final_output = json.loads(
        (tmp_path / "final" / "blank-chart-force-qwen.json").read_text(
            encoding="utf-8"
        )
    )
    stage2_payload = json.loads(
        (tmp_path / "judge_stage2" / "blank-chart-force-qwen.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(qwen_client.calls) == 2
    assert qwen_client.calls[1]["context"]["mode"] == "table_adjudication"
    assert qwen_client.calls[1]["context"]["issue_payload"]["review_mode"] == "chart_table_second_pass"
    assert any(
        candidate["candidate_id"] == "mineru"
        for candidate in qwen_client.calls[1]["context"]["issue_payload"]["candidates"]
    )
    assert any(
        candidate["candidate_id"] == "qwen"
        for candidate in qwen_client.calls[1]["context"]["issue_payload"]["candidates"]
    )
    assert final_output["model_name"] == "qwen-local"
    assert artifact["issues"]
    assert artifact["issues"][0]["target_block_id"] == "m1"
    assert artifact["patch_decisions"]
    assert artifact["final_document"]["raw_metadata"]["table_analysis"]["stage2_issue_count"] == 1
    assert artifact["final_document"]["raw_metadata"]["selected_output_role"] == "qwen"
    assert stage2_payload["record_count"] == 1


def test_process_image_task_falls_back_to_mineru_when_table_qwen_fails(
    tmp_path,
) -> None:
    image_task = ImageTask(
        image_id="table-qwen-fail",
        image_path="data/demo.png",
        file_name="table-qwen-fail.png",
        file_ext=".png",
    )
    mineru_table = "| 指标 | 值 |\n| --- | --- |\n| 增长率 | 12% |"
    paddle_table = "| 地区 | Q1 | Q2 |\n| --- | --- | --- |\n| 华东 | 10 | 20 |"
    glm_table = "| 公式 | 值 |\n| --- | --- |\n| $x^2$ | 5 |"
    mineru_client = StubClient(
        model_name="mineru-local",
        responses=[{"success": True, "parsed": _single_page_payload([_table_block("m1", mineru_table)])}],
        config={"provider": "minerupro_local", "role": "mineru"},
    )
    paddle_client = StubClient(
        model_name="paddle-local",
        responses=[{"success": True, "parsed": _single_page_payload([_table_block("p1", paddle_table)])}],
        config={"provider": "paddle_local", "role": "paddle"},
    )
    glm_client = StubClient(
        model_name="glm-local",
        responses=[{"success": True, "parsed": _single_page_payload([_table_block("g1", glm_table)])}],
        config={"provider": "glm_openai_compatible", "role": "glm"},
    )
    qwen_client = StubClient(
        model_name="qwen-local",
        responses=[{"success": False, "error": "timeout"}],
        config={"provider": "qwen_openai_compatible", "role": "judge"},
    )

    process_image_task(
        image_task=image_task,
        args=_build_args(),
        mineru_client=mineru_client,
        paddle_client=paddle_client,
        glm_client=glm_client,
        qwen_client=qwen_client,
        recognition_prompt="recognition prompt",
        seal_adjudication_prompt="seal prompt",
        flowchart_adjudication_prompt="flow prompt",
        output_dir=tmp_path,
        table_adjudication_prompt="table prompt",
    )

    artifact = json.loads(
        (tmp_path / "final" / "table-qwen-fail_artifact.json").read_text(encoding="utf-8")
    )
    assert len(qwen_client.calls) == 1
    assert artifact["final_document"]["blocks"][0]["content"]["table_body"] == mineru_table
    assert "only one parsable label" not in artifact["consensus"]["reasons"]
    assert (
        "table second-stage adjudication did not produce an adoptable patch"
        in artifact["consensus"]["reasons"]
    )
    assert artifact["final_document"]["raw_metadata"]["table_analysis"]["artifact_reference_included"] is False
