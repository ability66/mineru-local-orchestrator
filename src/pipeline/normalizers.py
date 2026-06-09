from __future__ import annotations

import json
from typing import Any

from src.normalizer import (
    _extract_first_json_object,
    _strip_code_fences,
    normalize_model_output,
)
from src.pipeline.flowchart_utils import looks_like_mermaid
from src.pipeline.table_utils import is_html_table_like
from src.schema import (
    CanonicalBlock,
    CanonicalDocument,
    CaptionStructured,
    ImageTask,
    ModelOutput,
    OcrRegion,
    ParsedLabel,
    StructuredLabel,
)


def normalize_mineru_payload(
    image_task: ImageTask,
    model_output: ModelOutput,
) -> tuple[ModelOutput, CanonicalDocument, ParsedLabel | None]:
    raw_payload = _extract_raw_payload(model_output)
    document, warnings = _canonical_document_from_payload(
        document_id=image_task.image_id,
        source=model_output.model_name,
        payload=raw_payload,
        default_image_path=image_task.image_path,
    )

    updated = model_output.model_copy(deep=True)
    updated.parsed = raw_payload
    if warnings:
        message = "; ".join(warnings)
        updated.error = _merge_errors(updated.error, message) if not updated.success else updated.error

    label = derive_label_from_document(document)
    return updated, document, label


def normalize_paddle_payload(
    image_task: ImageTask,
    model_output: ModelOutput,
) -> tuple[ModelOutput, CanonicalDocument, ParsedLabel | None]:
    raw_payload = _extract_raw_payload(model_output)
    document, warnings = _canonical_document_from_payload(
        document_id=image_task.image_id,
        source=model_output.model_name,
        payload=raw_payload,
        default_image_path=image_task.image_path,
    )

    updated = model_output.model_copy(deep=True)
    updated.parsed = raw_payload
    if warnings:
        message = "; ".join(warnings)
        updated.error = (
            _merge_errors(updated.error, message) if not updated.success else updated.error
        )

    label = derive_label_from_document(document)
    return updated, document, label


def normalize_qwen_payload(
    image_task: ImageTask,
    model_output: ModelOutput,
) -> tuple[ModelOutput, CanonicalDocument, ParsedLabel | None]:
    normalized_output, parsed_label = normalize_model_output(model_output)
    raw_payload = _parse_json_object(model_output.raw_text)
    content_payload = _extract_document_payload(raw_payload)

    if content_payload is not None:
        document, warnings = _canonical_document_from_payload(
            document_id=image_task.image_id,
            source=model_output.model_name,
            payload=content_payload,
            default_image_path=image_task.image_path,
        )
        if parsed_label is not None:
            document = _apply_label_patch_to_document(document=document, parsed_label=parsed_label)
    elif parsed_label is not None:
        document = _document_from_parsed_label(
            image_task=image_task,
            source=model_output.model_name,
            parsed_label=parsed_label,
        )
        warnings = []
    else:
        document = CanonicalDocument(
            document_id=image_task.image_id,
            source=model_output.model_name,
            backend="qwen",
            page_count=1,
            blocks=[],
        )
        warnings = ["qwen_output_missing_document_blocks"]

    derived_label = derive_label_from_document(document)
    if derived_label is not None:
        parsed_label = derived_label

    if warnings and normalized_output.success and not normalized_output.error:
        normalized_output.error = None

    return normalized_output, document, parsed_label


def derive_label_from_document(document: CanonicalDocument) -> ParsedLabel | None:
    blocks = sorted(document.blocks, key=lambda item: (item.page_idx, item.order_index, item.block_id))
    if not blocks:
        return None

    image_type = "document"
    if any(_is_flowchart_block(block) for block in blocks):
        image_type = "flowchart"
    elif any(block.type == "table" for block in blocks):
        image_type = "table"
    elif any(block.type == "chart" for block in blocks):
        image_type = "chart"
    elif any(_is_seal_block(block) for block in blocks):
        image_type = "seal"

    caption = _pick_caption(blocks)
    visible_text = _deduplicate_texts(
        [text for block in blocks for text in ([block.text] + list(block.visible_text))]
    )
    structured_label = _pick_structured_label(blocks)
    ocr_regions = [region for block in blocks for region in block.ocr_regions]

    return ParsedLabel(
        image_type=image_type,
        caption=caption,
        caption_structured=CaptionStructured(
            brief=caption,
            visual_type=image_type,
            main_subject=caption,
            visible_title=caption if blocks and blocks[0].type == "title" else "",
            key_visible_text=visible_text[:10],
            structure_summary=_build_structure_summary(blocks=blocks, image_type=image_type),
            caption_source="generated",
            confidence="medium" if caption else "low",
        ),
        structured_label=structured_label,
        flowchart_graph=_pick_flowchart_graph(blocks),
        visible_text=visible_text,
        ocr_regions=ocr_regions,
        uncertainty="",
        warnings=_deduplicate_texts([warning for block in blocks for warning in block.warnings]),
    )


def _canonical_document_from_payload(
    document_id: str,
    source: str,
    payload: Any,
    default_image_path: str,
) -> tuple[CanonicalDocument, list[str]]:
    warnings: list[str] = []
    blocks: list[CanonicalBlock] = []

    extracted_payload = _extract_document_payload(payload)
    if extracted_payload is not None:
        payload = extracted_payload

    if isinstance(payload, dict) and "pdf_info" in payload:
        return _document_from_pdf_info(
            document_id=document_id,
            source=source,
            pdf_info=payload.get("pdf_info"),
            default_image_path=default_image_path,
        )

    if _looks_like_nested_pages(payload):
        blocks = _blocks_from_content_list_v2(
            pages=payload,
            source=source,
            default_image_path=default_image_path,
        )
    elif isinstance(payload, list):
        blocks = _blocks_from_content_list_flat(
            items=payload,
            source=source,
            default_image_path=default_image_path,
        )
    else:
        warnings.append("unsupported_payload_shape")

    page_count = max((block.page_idx for block in blocks), default=-1) + 1
    document = CanonicalDocument(
        document_id=document_id,
        source=source,
        backend="mineru_like",
        page_count=max(1, page_count),
        blocks=blocks,
        warnings=warnings,
        raw_metadata={},
    )
    return document, warnings


def _document_from_pdf_info(
    document_id: str,
    source: str,
    pdf_info: Any,
    default_image_path: str,
) -> tuple[CanonicalDocument, list[str]]:
    warnings: list[str] = []
    blocks: list[CanonicalBlock] = []

    if not isinstance(pdf_info, list):
        warnings.append("invalid_pdf_info_payload")
    else:
        for page in pdf_info:
            if not isinstance(page, dict):
                continue
            page_idx = _coerce_non_negative_int(page.get("page_idx"), default=0)
            page_blocks = page.get("para_blocks", [])
            if not isinstance(page_blocks, list):
                continue
            for index, block in enumerate(page_blocks, start=1):
                if not isinstance(block, dict):
                    continue
                canonical = _block_from_generic_payload(
                    block=block,
                    page_idx=page_idx,
                    order_index=index,
                    source=source,
                    default_image_path=default_image_path,
                )
                blocks.append(canonical)

    page_count = max((block.page_idx for block in blocks), default=-1) + 1
    return (
        CanonicalDocument(
            document_id=document_id,
            source=source,
            backend="mineru_middle_json",
            page_count=max(1, page_count),
            blocks=blocks,
            warnings=warnings,
            raw_metadata={},
        ),
        warnings,
    )


def _blocks_from_content_list_v2(
    pages: Any,
    source: str,
    default_image_path: str,
) -> list[CanonicalBlock]:
    blocks: list[CanonicalBlock] = []
    if not isinstance(pages, list):
        return blocks
    for page_idx, page_blocks in enumerate(pages):
        if not isinstance(page_blocks, list):
            continue
        for order_index, block in enumerate(page_blocks, start=1):
            if not isinstance(block, dict):
                continue
            blocks.append(
                _block_from_v2_payload(
                    block=block,
                    page_idx=page_idx,
                    order_index=order_index,
                    source=source,
                    default_image_path=default_image_path,
                )
            )
    return blocks


def _blocks_from_content_list_flat(
    items: Any,
    source: str,
    default_image_path: str,
) -> list[CanonicalBlock]:
    blocks: list[CanonicalBlock] = []
    if not isinstance(items, list):
        return blocks
    page_order: dict[int, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        page_idx = _coerce_non_negative_int(item.get("page_idx"), default=0)
        order_index = page_order.get(page_idx, 0) + 1
        page_order[page_idx] = order_index
        blocks.append(
            _block_from_flat_payload(
                block=item,
                page_idx=page_idx,
                order_index=order_index,
                source=source,
                default_image_path=default_image_path,
            )
        )
    return blocks


def _block_from_v2_payload(
    block: dict[str, Any],
    page_idx: int,
    order_index: int,
    source: str,
    default_image_path: str,
) -> CanonicalBlock:
    block_type = _map_v2_type(str(block.get("type", "paragraph") or "paragraph"))
    normalized_content = _normalize_v2_content(
        block_type=block_type,
        block=block,
        default_image_path=default_image_path,
    )
    text = _extract_block_text(
        block_type=block_type,
        content=normalized_content,
        fallback=block.get("text") if block.get("text") is not None else block.get("content"),
    )
    structured_label = _structured_label_from_block_payload(
        block_type=block_type,
        block=block,
        content=normalized_content,
    )
    ocr_regions = _ocr_regions_from_block(block, normalized_content)
    visible_text = _deduplicate_texts(_collect_visible_texts(normalized_content, text))
    block_id = str(block.get("block_id") or f"p{page_idx:03d}_b{order_index:03d}")
    raw_content = block.get("content")
    payload_format = "content_list_v2" if isinstance(raw_content, dict) else "json_res_flat"

    block_sub_type = _optional_text(block.get("sub_type"))
    if (
        block_type == "chart"
        and str(block_sub_type or "").strip().lower() not in {"flowchart", "html_table"}
        and structured_label.kind == "table"
    ):
        block_sub_type = "html_table"

    return CanonicalBlock(
        block_id=block_id,
        page_idx=page_idx,
        order_index=order_index,
        type=block_type,
        sub_type=block_sub_type,
        bbox=_normalize_bbox(block.get("bbox")),
        text=text,
        text_level=_coerce_non_negative_int(block.get("text_level")),
        content=normalized_content,
        source=source,
        confidence=_coerce_float(block.get("score")),
        structured_label=structured_label,
        caption_structured=_caption_structured_from_block(
            block_type=block_type,
            block_sub_type=block_sub_type,
            text=text,
            content=normalized_content,
            visible_text=visible_text,
        ),
        flowchart_graph=_flowchart_graph_from_block(block),
        visible_text=visible_text,
        ocr_regions=ocr_regions,
        warnings=[],
        provenance={
            "source_block_type": block.get("type"),
            "source_angle": _coerce_float(block.get("angle")),
            "format": payload_format,
        },
    )


def _block_from_flat_payload(
    block: dict[str, Any],
    page_idx: int,
    order_index: int,
    source: str,
    default_image_path: str,
) -> CanonicalBlock:
    mapped = dict(block)
    block_type = _map_flat_type(str(block.get("type", "text") or "text"))
    if block_type in {"title", "paragraph"}:
        content = {
            "title_content" if block_type == "title" else "paragraph_content": [
                {"type": "text", "content": str(block.get("text", "") or "")}
            ]
        }
        if block_type == "title":
            content["level"] = _coerce_non_negative_int(block.get("text_level"), default=1)
    else:
        content = {key: value for key, value in mapped.items() if key not in {"type", "page_idx", "bbox", "text_level"}}

    if block_type in {"chart", "image", "table"} and "img_path" not in content:
        content["img_path"] = default_image_path

    return _block_from_v2_payload(
        block={
            "block_id": block.get("block_id"),
            "type": block_type,
            "sub_type": block.get("sub_type"),
            "bbox": block.get("bbox"),
            "content": content,
            "text_level": block.get("text_level"),
            "score": block.get("score"),
        },
        page_idx=page_idx,
        order_index=order_index,
        source=source,
        default_image_path=default_image_path,
    )


def _block_from_generic_payload(
    block: dict[str, Any],
    page_idx: int,
    order_index: int,
    source: str,
    default_image_path: str,
) -> CanonicalBlock:
    block_type = _infer_generic_block_type(block)
    content = {
        "paragraph_content": [{"type": "text", "content": _extract_text_from_generic_block(block)}]
    }
    if block_type in {"chart", "image", "table"}:
        content["img_path"] = default_image_path
    return _block_from_v2_payload(
        block={
            "block_id": block.get("block_id"),
            "type": block_type,
            "sub_type": block.get("sub_type"),
            "bbox": block.get("bbox"),
            "content": content,
            "score": block.get("score"),
        },
        page_idx=page_idx,
        order_index=order_index,
        source=source,
        default_image_path=default_image_path,
    )


def _document_from_parsed_label(
    image_task: ImageTask,
    source: str,
    parsed_label: ParsedLabel,
) -> CanonicalDocument:
    block_type = "image"
    sub_type: str | None = None
    content: dict[str, Any] = {"img_path": image_task.image_path}

    if parsed_label.image_type == "table":
        block_type = "table"
        content["table_body"] = parsed_label.structured_label.content
        content["table_caption"] = [parsed_label.caption] if parsed_label.caption else []
    elif parsed_label.image_type in {"chart", "flowchart"}:
        block_type = "chart"
        sub_type = "flowchart" if parsed_label.image_type == "flowchart" else None
        if parsed_label.caption:
            content["chart_caption"] = [parsed_label.caption]
        if (
            parsed_label.image_type != "flowchart"
            and parsed_label.structured_label.kind == "table"
            and parsed_label.structured_label.content
        ):
            sub_type = "html_table"
            content["content"] = parsed_label.structured_label.content
        elif parsed_label.image_type != "flowchart" and parsed_label.structured_label.content:
            content["content"] = parsed_label.structured_label.content
        elif looks_like_mermaid(parsed_label.structured_label.content):
            content["content"] = parsed_label.structured_label.content
    elif parsed_label.image_type == "seal" or any(
        region.role == "seal" for region in parsed_label.ocr_regions
    ):
        block_type = "image"
        sub_type = "seal"
        if parsed_label.caption:
            content["image_caption"] = [parsed_label.caption]
    else:
        if parsed_label.caption:
            content["image_caption"] = [parsed_label.caption]

    block = CanonicalBlock(
        block_id="p000_b001",
        page_idx=0,
        order_index=1,
        type=block_type,
        sub_type=sub_type,
        bbox=[0, 0, 1000, 1000],
        text=parsed_label.caption,
        text_level=None,
        content=content,
        source=source,
        confidence=None,
        structured_label=parsed_label.structured_label,
        caption_structured=parsed_label.caption_structured,
        flowchart_graph=parsed_label.flowchart_graph,
        visible_text=parsed_label.visible_text,
        ocr_regions=parsed_label.ocr_regions,
        warnings=parsed_label.warnings,
        provenance={"format": "parsed_label"},
    )
    return CanonicalDocument(
        document_id=image_task.image_id,
        source=source,
        backend="qwen_label_only",
        page_count=1,
        blocks=[block],
        warnings=[],
        raw_metadata={},
    )


def _apply_label_patch_to_document(
    document: CanonicalDocument,
    parsed_label: ParsedLabel,
) -> CanonicalDocument:
    if not document.blocks:
        return document

    patched_document = document.model_copy(deep=True)
    target = _pick_patch_target(blocks=patched_document.blocks, parsed_label=parsed_label)
    if target is None:
        return patched_document

    if parsed_label.image_type == "table" and target.type == "table":
        if parsed_label.structured_label.content.strip() and not str(target.content.get("table_body", "") or "").strip():
            target.content["table_body"] = parsed_label.structured_label.content
            target.structured_label = parsed_label.structured_label
        _append_caption(target.content, "table_caption", parsed_label.caption)
        return patched_document

    if parsed_label.image_type in {"chart", "flowchart"} and target.type == "chart":
        if parsed_label.image_type == "flowchart":
            target.sub_type = "flowchart"
        elif (
            parsed_label.structured_label.kind == "table"
            and parsed_label.structured_label.content.strip()
            and not str(target.content.get("content", "") or "").strip()
        ):
            target.sub_type = target.sub_type or "html_table"
            target.content["content"] = parsed_label.structured_label.content
            target.structured_label = parsed_label.structured_label
        if (
            looks_like_mermaid(parsed_label.structured_label.content)
            and not str(target.content.get("content", "") or "").strip()
        ):
            target.content["content"] = parsed_label.structured_label.content
            target.structured_label = parsed_label.structured_label
        if parsed_label.flowchart_graph and target.flowchart_graph is None:
            target.flowchart_graph = parsed_label.flowchart_graph
        _append_caption(target.content, "chart_caption", parsed_label.caption)
        return patched_document

    if (
        parsed_label.image_type == "seal"
        or any(region.role == "seal" for region in parsed_label.ocr_regions)
    ) and target.type == "image":
        target.sub_type = target.sub_type or "seal"
        target.ocr_regions = _merge_ocr_region_items(target.ocr_regions, parsed_label.ocr_regions)
        _append_caption(target.content, "image_caption", parsed_label.caption)
        return patched_document

    if target.type == "image":
        _append_caption(target.content, "image_caption", parsed_label.caption)
    elif target.type == "chart":
        _append_caption(target.content, "chart_caption", parsed_label.caption)
    elif target.type == "table":
        _append_caption(target.content, "table_caption", parsed_label.caption)
    elif parsed_label.caption.strip() and not target.text.strip():
        target.text = parsed_label.caption.strip()
    return patched_document


def _pick_caption(blocks: list[CanonicalBlock]) -> str:
    for block in blocks:
        if block.type == "title" and block.text.strip():
            return block.text.strip()
    for block in blocks:
        caption = block.caption_structured.brief.strip()
        if caption:
            return caption
    for block in blocks:
        if block.text.strip():
            return block.text.strip()
    return ""


def _pick_structured_label(blocks: list[CanonicalBlock]) -> StructuredLabel:
    for block in blocks:
        if block.structured_label.kind != "none" and block.structured_label.content.strip():
            return block.structured_label
    return StructuredLabel(kind="none", content="", format="none", source="none")


def _pick_flowchart_graph(blocks: list[CanonicalBlock]) -> dict[str, Any] | None:
    for block in blocks:
        if block.flowchart_graph:
            return block.flowchart_graph
    return None


def _build_structure_summary(blocks: list[CanonicalBlock], image_type: str) -> str:
    if image_type == "flowchart":
        return "流程图结构，已保留节点、边和可见关键文字。"
    if image_type == "table":
        return "表格结构，优先保留表体和表格说明。"
    if image_type == "chart":
        return "图表结构，优先保留图表标题、说明和可见关键文字。"
    if image_type == "seal":
        return "图像中包含印章区域，已保留印章 OCR 候选。"
    if any(_is_seal_block(block) for block in blocks):
        return "图像中包含印章区域，已保留印章 OCR 候选。"
    return "基于 MinerU 风格内容块构建的单页结构。"


def _extract_raw_payload(model_output: ModelOutput) -> Any:
    if model_output.parsed is not None:
        return model_output.parsed
    return _parse_json_object(model_output.raw_text)


def _parse_json_object(raw_text: str) -> Any:
    cleaned_text = _strip_code_fences(raw_text)
    json_text, _ = _extract_first_json_object(cleaned_text)
    if json_text is None:
        return None
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        return None


def _extract_document_payload(payload: Any) -> Any | None:
    current = payload
    visited: set[int] = set()
    wrapper_keys = ("parsed", "data", "result", "payload", "output", "response")

    while True:
        if current is None:
            return None
        if isinstance(current, list):
            extraction_list_payload = _extract_page_result_list_payload(current)
            if extraction_list_payload is not None:
                return extraction_list_payload
            return current
        if not isinstance(current, dict):
            return None

        extraction_result = current.get("extraction_result")
        if extraction_result is not None:
            extracted = _extract_extraction_result_payload(extraction_result)
            if extracted is not None:
                return extracted

        extraction_results = current.get("extraction_results")
        if extraction_results is not None:
            extracted = _extract_extraction_result_payload(extraction_results)
            if extracted is not None:
                return extracted

        if "json_res" in current:
            extracted = _extract_document_payload(_coerce_embedded_payload(current.get("json_res")))
            if extracted is not None:
                return extracted
        if "content_list_v2" in current:
            return current["content_list_v2"]
        if "content_list" in current:
            return current["content_list"]
        if "pdf_info" in current:
            return {"pdf_info": current.get("pdf_info")}
        if "parsing_res_list" in current:
            return _extract_paddle_parsing_res_list_payload(current)
        if "blocks" in current:
            return current["blocks"]

        next_payload = None
        for key in wrapper_keys:
            candidate = current.get(key)
            if not isinstance(candidate, (dict, list)):
                continue
            candidate_id = id(candidate)
            if candidate_id in visited:
                continue
            visited.add(candidate_id)
            next_payload = candidate
            break

        if next_payload is None:
            return None
        current = next_payload


def _extract_extraction_result_payload(value: Any) -> Any | None:
    if isinstance(value, dict):
        extracted = _extract_document_payload(_coerce_embedded_payload(value.get("json_res")))
        if extracted is not None:
            return extracted
        return _extract_document_payload(value.get("pages"))
    if isinstance(value, list):
        return _extract_page_result_list_payload(value)
    return None


def _extract_page_result_list_payload(items: list[Any]) -> Any | None:
    if not items:
        return []
    if not all(isinstance(item, dict) for item in items):
        return None
    if not any(
        "json_res" in item
        or "md_res" in item
        or "filename" in item
        or "file_name" in item
        or "page" in item
        for item in items
    ):
        return None

    normalized_page_indexes = _normalize_page_indexes(items)
    ordered_pages: list[tuple[int, list[dict[str, Any]]]] = []
    fallback_payload: Any | None = None
    for index, item in enumerate(items):
        page_index = normalized_page_indexes[index]
        candidate_payload = _coerce_embedded_payload(item.get("json_res"))
        extracted = _extract_document_payload(candidate_payload)
        if extracted is None:
            continue
        if fallback_payload is None:
            fallback_payload = extracted
        page_blocks = _coerce_page_blocks(extracted)
        if page_blocks is None:
            continue
        ordered_pages.append((page_index, page_blocks))

    if ordered_pages:
        ordered_pages.sort(key=lambda item: item[0])
        max_page_index = max(page_index for page_index, _ in ordered_pages)
        pages: list[list[dict[str, Any]]] = [[] for _ in range(max_page_index + 1)]
        for page_index, page_blocks in ordered_pages:
            pages[page_index] = page_blocks
        return pages
    return fallback_payload


def _extract_paddle_parsing_res_list_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    parsing_res_list = payload.get("parsing_res_list")
    if not isinstance(parsing_res_list, list):
        return []
    layout_boxes = _extract_paddle_layout_boxes(payload.get("layout_det_res"))
    transformed_blocks: list[tuple[int, dict[str, Any]]] = []
    for index, block in enumerate(parsing_res_list, start=1):
        if not isinstance(block, dict):
            continue
        order_index = _coerce_non_negative_int(block.get("block_order"), default=index)
        transformed = _paddle_block_to_flat_payload(
            block=block,
            layout_boxes=layout_boxes,
        )
        if transformed is None:
            continue
        transformed_blocks.append((order_index or index, transformed))
    transformed_blocks.sort(key=lambda item: item[0])
    return [item[1] for item in transformed_blocks]


def _extract_paddle_layout_boxes(layout_det_res: Any) -> list[dict[str, Any]]:
    if not isinstance(layout_det_res, dict):
        return []
    boxes = layout_det_res.get("boxes")
    if not isinstance(boxes, list):
        return []
    return [item for item in boxes if isinstance(item, dict)]


def _paddle_block_to_flat_payload(
    block: dict[str, Any],
    layout_boxes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    raw_label = str(block.get("block_label", "") or "").strip().lower()
    block_type, block_sub_type = _map_paddle_block_label(raw_label)
    text = str(block.get("block_content", "") or "").strip()
    matching_box = _match_paddle_layout_box(block=block, layout_boxes=layout_boxes)
    score = block.get("score")
    if score is None and matching_box is not None:
        score = matching_box.get("score")
    block_id = block.get("block_id")
    if block_id in (None, ""):
        block_id = block.get("block_order")
    if block_id in (None, ""):
        block_id = f"{raw_label or 'block'}_{len(layout_boxes)}"
    transformed: dict[str, Any] = {
        "block_id": f"paddle_{block_id}",
        "type": block_type,
        "bbox": block.get("block_bbox") or (
            matching_box.get("coordinate") if matching_box is not None else []
        ),
        "score": score,
    }
    if block_sub_type is not None:
        transformed["sub_type"] = block_sub_type

    content: Any = ""
    if block_type == "title":
        content = text
        transformed["text_level"] = 1
    elif block_type == "paragraph":
        content = text
    elif block_type == "table":
        content = {"table_body": text} if text else {}
    elif block_type == "chart":
        if block_sub_type == "flowchart" and looks_like_mermaid(text):
            content = {"content": text}
        elif text:
            content = {"chart_caption": [text]}
    elif block_type == "image" and text:
        content = {"image_caption": [text]}
    transformed["content"] = content

    if block_sub_type == "seal" and text:
        transformed["ocr_regions"] = [
            {
                "role": "seal",
                "text": text,
                "confidence": score if score is not None else "medium",
                "bbox_hint": block.get("block_bbox"),
            }
        ]
    return transformed


def _map_paddle_block_label(raw_label: str) -> tuple[str, str | None]:
    if raw_label == "seal":
        return "image", "seal"
    if raw_label in {"flowchart"}:
        return "chart", "flowchart"
    if raw_label in {"chart", "graph", "plot"}:
        return "chart", None
    if raw_label in {"table"}:
        return "table", None
    if raw_label in {"image", "figure", "picture", "photo"}:
        return "image", None
    if raw_label in {"title", "doc_title"}:
        return "title", None
    return "paragraph", None


def _match_paddle_layout_box(
    block: dict[str, Any],
    layout_boxes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    block_order = _coerce_non_negative_int(block.get("block_order"))
    block_bbox = _normalize_bbox(block.get("block_bbox"))
    block_label = str(block.get("block_label", "") or "").strip().lower()

    for box in layout_boxes:
        if (
            block_order is not None
            and _coerce_non_negative_int(box.get("order")) == block_order
        ):
            return box

    for box in layout_boxes:
        box_label = str(box.get("label", "") or "").strip().lower()
        box_bbox = _normalize_bbox(box.get("coordinate"))
        if block_label and box_label and block_label != box_label:
            continue
        if block_bbox and box_bbox and block_bbox == box_bbox:
            return box
    return None


def _normalize_page_indexes(items: list[dict[str, Any]]) -> list[int]:
    explicit_pages = [_coerce_non_negative_int(item.get("page")) for item in items]
    normalized_explicit = [page for page in explicit_pages if page is not None]
    subtract_one = bool(normalized_explicit) and 0 not in normalized_explicit and min(normalized_explicit) == 1

    normalized_pages: list[int] = []
    for index, page in enumerate(explicit_pages):
        if page is None:
            normalized_pages.append(index)
            continue
        normalized_pages.append(max(0, page - 1) if subtract_one else page)
    return normalized_pages


def _coerce_embedded_payload(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _coerce_page_blocks(payload: Any) -> list[dict[str, Any]] | None:
    if isinstance(payload, list):
        if not payload:
            return []
        if all(isinstance(item, dict) for item in payload):
            return payload
        if _looks_like_nested_pages(payload):
            first_page = payload[0]
            if isinstance(first_page, list) and all(isinstance(item, dict) for item in first_page):
                return first_page
    return None


def _looks_like_nested_pages(payload: Any) -> bool:
    return isinstance(payload, list) and (
        not payload or isinstance(payload[0], list)
    )


def _map_v2_type(raw_type: str) -> str:
    normalized = raw_type.strip().lower()
    mapping = {
        "text": "paragraph",
        "paragraph": "paragraph",
        "title": "title",
        "image": "image",
        "table": "table",
        "chart": "chart",
        "equation_interline": "equation_interline",
        "equation": "equation_interline",
        "code": "code",
        "algorithm": "algorithm",
        "list": "list",
        "header": "page_header",
        "page_header": "page_header",
        "footer": "page_footer",
        "page_footer": "page_footer",
        "page_number": "page_number",
        "aside_text": "page_aside_text",
        "page_aside_text": "page_aside_text",
        "page_footnote": "page_footnote",
    }
    return mapping.get(normalized, "paragraph")


def _map_flat_type(raw_type: str) -> str:
    normalized = raw_type.strip().lower()
    if normalized == "text":
        return "paragraph"
    return _map_v2_type(normalized)


def _normalize_v2_content(
    block_type: str,
    block: dict[str, Any],
    default_image_path: str,
) -> dict[str, Any]:
    raw_content = block.get("content")
    if isinstance(raw_content, dict):
        normalized_content = dict(raw_content)
    else:
        normalized_content = _content_dict_from_flat_text(
            block_type=block_type,
            raw_text=str(raw_content or "").strip(),
            block=block,
        )

    if block_type in {"chart", "image", "table"} and "img_path" not in normalized_content:
        normalized_content["img_path"] = default_image_path
    return normalized_content


def _content_dict_from_flat_text(
    block_type: str,
    raw_text: str,
    block: dict[str, Any],
) -> dict[str, Any]:
    if block_type == "title":
        return {
            "title_content": [{"type": "text", "content": raw_text}] if raw_text else [],
            "level": _coerce_non_negative_int(block.get("text_level"), default=1) or 1,
        }
    if block_type == "paragraph":
        return {
            "paragraph_content": [{"type": "text", "content": raw_text}] if raw_text else []
        }
    if block_type == "table":
        return {"table_body": raw_text}
    if block_type == "chart":
        if str(block.get("sub_type") or "").strip().lower() == "flowchart":
            if looks_like_mermaid(raw_text):
                return {"content": raw_text}
            return {"chart_caption": [raw_text]} if raw_text else {}
        return {"content": raw_text}
    if block_type == "image":
        return {"image_caption": [raw_text]} if raw_text else {}
    if block_type == "list":
        return {"list_items": [raw_text]} if raw_text else {}
    if block_type == "equation_interline":
        return {"math_content": raw_text, "math_type": "plain_text"} if raw_text else {}
    if raw_text:
        return {"text": raw_text}
    return {}


def _extract_block_text(block_type: str, content: dict[str, Any], fallback: Any) -> str:
    if block_type == "title":
        return _join_span_content(content.get("title_content"))
    if block_type == "paragraph":
        return _join_span_content(content.get("paragraph_content"))
    if block_type == "table":
        captions = content.get("table_caption", [])
        body = str(content.get("table_body", "") or "").strip()
        return " ".join(_normalize_text_list(captions) + ([body] if body else []))
    if block_type == "chart":
        captions = content.get("chart_caption", [])
        body = str(content.get("content", "") or "").strip()
        return " ".join(_normalize_text_list(captions) + ([body] if body else []))
    if block_type == "image":
        captions = content.get("image_caption", [])
        return " ".join(_normalize_text_list(captions))
    if block_type == "list":
        return " ".join(_normalize_text_list(content.get("list_items", [])))
    if block_type == "equation_interline":
        return str(content.get("math_content", "") or "").strip()
    return str(fallback or "").strip()


def _structured_label_from_block_payload(
    block_type: str,
    block: dict[str, Any],
    content: dict[str, Any],
) -> StructuredLabel:
    if block_type == "table":
        table_body = str(content.get("table_body", "") or "")
        structured_format = "html" if is_html_table_like({"type": "table", "content": {"table_body": table_body}}) else "markdown"
        return StructuredLabel(
            kind="table",
            content=table_body,
            format=structured_format,  # type: ignore[arg-type]
            source="mineru",
        )
    if block_type == "chart" and str(block.get("sub_type") or "").strip().lower() == "flowchart":
        mermaid = str(content.get("content", "") or "").strip()
        caption_fallback = " ".join(_normalize_text_list(content.get("chart_caption", []))).strip()
        if looks_like_mermaid(mermaid):
            return StructuredLabel(
                kind="mermaid",
                content=mermaid,
                format="mermaid",
                source="mineru",
            )
        return StructuredLabel(
            kind="text" if caption_fallback else "none",
            content=caption_fallback,
            format="plain_text" if caption_fallback else "none",
            source="mineru" if caption_fallback else "none",
        )
    if (
        block_type == "chart"
        and str(block.get("sub_type") or "").strip().lower() != "flowchart"
        and is_html_table_like(
            {
                "type": "chart",
                "sub_type": block.get("sub_type"),
                "content": {"content": str(content.get("content", "") or "")},
            }
        )
    ):
        return StructuredLabel(
            kind="table",
            content=str(content.get("content", "") or ""),
            format="html",
            source="mineru",
        )
    return StructuredLabel(kind="none", content="", format="none", source="none")


def _caption_structured_from_block(
    block_type: str,
    block_sub_type: str | None,
    text: str,
    content: dict[str, Any],
    visible_text: list[str],
) -> CaptionStructured:
    brief = text.strip()
    if not brief:
        if block_type == "table":
            captions = _normalize_text_list(content.get("table_caption", []))
            brief = captions[0] if captions else ""
        elif block_type == "chart":
            captions = _normalize_text_list(content.get("chart_caption", []))
            brief = captions[0] if captions else ""
        elif block_type == "image":
            captions = _normalize_text_list(content.get("image_caption", []))
            brief = captions[0] if captions else ""
    visual_type = "document"
    if block_type == "table":
        visual_type = "table"
    elif block_type == "chart":
        visual_type = "chart"
    elif block_type == "image":
        visual_type = (
            "seal"
            if str(block_sub_type or "").strip().lower() == "seal"
            else "natural_image"
        )
    return CaptionStructured(
        brief=brief,
        visual_type=visual_type,
        main_subject=brief,
        visible_title=brief if block_type == "title" else "",
        key_visible_text=visible_text[:10],
        structure_summary="",
        caption_source="generated",
        confidence="medium" if brief else "low",
    )


def _flowchart_graph_from_block(block: dict[str, Any]) -> dict[str, Any] | None:
    flowchart_graph = block.get("flowchart_graph")
    return flowchart_graph if isinstance(flowchart_graph, dict) else None


def _is_flowchart_block(block: CanonicalBlock) -> bool:
    if str(block.sub_type or "").strip().lower() == "flowchart":
        return True
    if block.structured_label.kind == "mermaid":
        return True
    return bool(block.flowchart_graph)


def _pick_patch_target(
    blocks: list[CanonicalBlock],
    parsed_label: ParsedLabel,
) -> CanonicalBlock | None:
    if parsed_label.image_type == "table":
        for block in blocks:
            if block.type == "table":
                return block
    elif parsed_label.image_type in {"chart", "flowchart"}:
        for block in blocks:
            if block.type == "chart":
                return block
    elif parsed_label.image_type == "seal" or any(
        str(region.role or "").strip().lower() == "seal"
        for region in parsed_label.ocr_regions
    ):
        for block in blocks:
            if block.type == "image":
                return block
    return _pick_visual_block(blocks) or (blocks[0] if blocks else None)


def _pick_visual_block(blocks: list[CanonicalBlock]) -> CanonicalBlock | None:
    for block in blocks:
        if block.type in {"chart", "image", "table"}:
            return block
    return None


def _ocr_regions_from_block(block: dict[str, Any], content: dict[str, Any]) -> list[OcrRegion]:
    regions_input = block.get("ocr_regions")
    if not isinstance(regions_input, list):
        regions_input = content.get("ocr_regions")
    if not isinstance(regions_input, list):
        regions_input = []

    regions: list[OcrRegion] = []
    for item in regions_input:
        if not isinstance(item, dict):
            continue
        regions.append(
            OcrRegion(
                role=_normalize_ocr_role(item.get("role")),
                text=str(item.get("text", "") or ""),
                bbox_hint=_normalize_bbox_hint(item.get("bbox_hint")),
                confidence=_normalize_confidence(item.get("confidence")),
            )
        )
    if str(block.get("sub_type") or "").strip().lower() == "seal" and not regions:
        text = " ".join(_normalize_text_list(content.get("image_caption", [])))
        if text.strip():
            regions.append(
                OcrRegion(role="seal", text=text.strip(), bbox_hint=None, confidence="medium")
            )
    return regions


def _collect_visible_texts(content: dict[str, Any], text: str) -> list[str]:
    values: list[str] = []
    if text.strip():
        values.append(text.strip())
    for key in (
        "title_content",
        "paragraph_content",
        "table_caption",
        "table_footnote",
        "chart_caption",
        "chart_footnote",
        "image_caption",
        "image_footnote",
        "list_items",
    ):
        value = content.get(key)
        if key.endswith("_content"):
            values.extend(_extract_span_texts(value))
        else:
            values.extend(_normalize_text_list(value))
    return values


def _extract_text_from_generic_block(block: dict[str, Any]) -> str:
    if "text" in block:
        return str(block.get("text", "") or "").strip()
    lines = block.get("lines")
    if not isinstance(lines, list):
        return ""
    texts: list[str] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        spans = line.get("spans", [])
        if not isinstance(spans, list):
            continue
        for span in spans:
            if not isinstance(span, dict):
                continue
            text = str(span.get("content", "") or "").strip()
            if text:
                texts.append(text)
    return " ".join(texts).strip()


def _infer_generic_block_type(block: dict[str, Any]) -> str:
    raw_type = str(block.get("type", "") or "").strip().lower()
    if raw_type in {"table", "chart", "image"}:
        return raw_type
    if raw_type in {"title", "doc_title"}:
        return "title"
    return "paragraph"


def _join_span_content(value: Any) -> str:
    texts = _extract_span_texts(value)
    return " ".join(texts).strip()


def _extract_span_texts(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    texts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        text = str(item.get("content", "") or "").strip()
        if text:
            texts.append(text)
    return texts


def _normalize_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _normalize_bbox(value: Any) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        return []
    numbers: list[float] = []
    for item in value:
        try:
            numbers.append(float(item))
        except (TypeError, ValueError):
            return []
    if all(0.0 <= item <= 1.0 for item in numbers):
        numbers = [item * 1000 for item in numbers]
    return [max(0, min(1000, int(round(item)))) for item in numbers]


def _normalize_bbox_hint(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    numbers: list[float] = []
    for item in value:
        try:
            numbers.append(float(item))
        except (TypeError, ValueError):
            return None
    return numbers


def _coerce_non_negative_int(value: Any, default: int | None = None) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _append_caption(content: dict[str, Any], key: str, caption: str) -> None:
    text = str(caption or "").strip()
    if not text:
        return
    existing = _normalize_text_list(content.get(key, []))
    content[key] = _deduplicate_texts(existing + [text])


def _normalize_ocr_role(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"seal", "stamp", "印章", "公章"}:
        return "seal"
    if normalized in {"watermark", "水印"}:
        return "watermark"
    if normalized in {"footer", "页脚", "底部文字"}:
        return "footer"
    if normalized in {"body", "正文"}:
        return "body"
    if normalized in {"title", "标题"}:
        return "title"
    return "other"


def _normalize_confidence(value: Any) -> str:
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric >= 0.85:
            return "high"
        if numeric >= 0.5:
            return "medium"
        return "low"

    normalized = str(value or "").strip().lower()
    if normalized in {"high", "medium", "low"}:
        return normalized
    if normalized in {"very_high", "strong", "sure"}:
        return "high"
    if normalized in {"mid", "moderate"}:
        return "medium"
    if normalized in {"weak"}:
        return "low"
    try:
        numeric = float(normalized)
    except (TypeError, ValueError):
        return "medium"
    if numeric >= 0.85:
        return "high"
    if numeric >= 0.5:
        return "medium"
    return "low"


def _is_seal_block(block: CanonicalBlock) -> bool:
    if str(block.sub_type or "").strip().lower() == "seal":
        return True
    return any(str(region.role or "").strip() == "seal" for region in block.ocr_regions)


def _merge_ocr_region_items(left: list[OcrRegion], right: list[OcrRegion]) -> list[OcrRegion]:
    merged: list[OcrRegion] = []
    seen: set[tuple[str, str]] = set()
    for region in list(left) + list(right):
        key = (str(region.role or "").strip(), str(region.text or "").strip())
        if key in seen:
            continue
        seen.add(key)
        merged.append(region)
    return merged


def _deduplicate_texts(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        normalized = "".join(text.split()).lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(text)
    return ordered


def _merge_errors(current: str | None, message: str) -> str:
    if not current:
        return message
    return f"{current}; {message}"
