from __future__ import annotations

from typing import Any

from src.schema import (
    CanonicalBlock,
    CanonicalDocument,
    ParsedLabel,
    StructuredLabel,
)

SINGLE_BLOCK_VIEW_IMAGE_TYPES = {
    "seal",
    "natural_image",
    "document",
    "screenshot",
    "diagram",
    "mixed",
    "unknown",
}


def project_document_for_single_block_view(
    document: CanonicalDocument | None,
    label: ParsedLabel | None,
) -> CanonicalDocument | None:
    if not isinstance(document, CanonicalDocument):
        return None
    if not isinstance(label, ParsedLabel):
        return document

    image_type = str(label.image_type or "").strip().lower()
    if image_type not in SINGLE_BLOCK_VIEW_IMAGE_TYPES or len(document.blocks) <= 1:
        return document

    ordered_blocks = sorted(
        document.blocks,
        key=lambda item: (item.page_idx, item.order_index, item.block_id),
    )
    if not ordered_blocks:
        return document

    merged_text_parts: list[str] = []
    merged_ocr_regions = []
    merged_warnings: list[str] = list(document.warnings)
    for block in ordered_blocks:
        merged_text_parts.extend(text_fragments_for_single_block_view(block))
        merged_ocr_regions.extend(block.ocr_regions)
        merged_warnings.extend(block.warnings)

    merged_text = "\n\n".join(part for part in merged_text_parts if part.strip())
    representative = pick_single_block_representative(
        blocks=ordered_blocks, image_type=image_type
    )
    img_path = first_image_path(ordered_blocks)
    bbox = merge_bboxes([block.bbox for block in ordered_blocks]) or representative.bbox
    sub_type = image_type if image_type else representative.sub_type
    caption = str(label.caption or representative.caption_structured.brief or "").strip()

    content: dict[str, Any] = {}
    if img_path:
        content["img_path"] = img_path
    if merged_text:
        content["image_caption"] = [merged_text]

    caption_structured = label.caption_structured.model_copy(deep=True)
    if not caption_structured.visual_type:
        caption_structured.visual_type = image_type
    if not caption_structured.brief:
        caption_structured.brief = caption
    if not caption_structured.main_subject:
        caption_structured.main_subject = caption_structured.brief
    if not caption_structured.key_visible_text:
        caption_structured.key_visible_text = merged_text_parts[:10]

    projected_block = CanonicalBlock(
        block_id=representative.block_id,
        page_idx=representative.page_idx,
        order_index=1,
        type="image",
        sub_type=sub_type,
        bbox=bbox,
        text=merged_text,
        text_level=None,
        content=content,
        source=representative.source,
        confidence=representative.confidence,
        structured_label=StructuredLabel(),
        caption_structured=caption_structured,
        flowchart_graph=None,
        visible_text=merged_text_parts,
        ocr_regions=merged_ocr_regions,
        warnings=ordered_unique_strings(merged_warnings),
        provenance={
            **representative.provenance,
            "normalized_view": "single_block_projection",
            "source_block_count": len(ordered_blocks),
        },
    )
    return CanonicalDocument(
        document_id=document.document_id,
        source=document.source,
        backend=document.backend,
        page_count=max(1, document.page_count),
        blocks=[projected_block],
        warnings=ordered_unique_strings(merged_warnings),
        raw_metadata={
            **document.raw_metadata,
            "normalized_view": "single_block_projection",
            "source_block_count": len(ordered_blocks),
        },
    )


def is_single_block_projection_document(document: CanonicalDocument | None) -> bool:
    if not isinstance(document, CanonicalDocument):
        return False
    return (
        str(document.raw_metadata.get("normalized_view", "") or "").strip()
        == "single_block_projection"
    )


def is_single_block_projection_block(block: CanonicalBlock | None) -> bool:
    if not isinstance(block, CanonicalBlock):
        return False
    return (
        str(block.provenance.get("normalized_view", "") or "").strip()
        == "single_block_projection"
    )


def pick_single_block_representative(
    blocks: list[CanonicalBlock],
    image_type: str,
) -> CanonicalBlock:
    if image_type == "seal":
        for block in blocks:
            if str(block.sub_type or "").strip().lower() == "seal":
                return block
    for block in blocks:
        if block.type == "image":
            return block
    return blocks[0]


def text_fragments_for_single_block_view(block: CanonicalBlock) -> list[str]:
    primary = str(block.text or "").strip()
    if primary:
        return [primary]

    fragments: list[str] = []
    for key in (
        "content",
        "table_body",
    ):
        value = block.content.get(key)
        if isinstance(value, str) and value.strip():
            fragments.append(value.strip())
    for key in (
        "image_caption",
        "chart_caption",
        "table_caption",
        "list_items",
    ):
        value = block.content.get(key)
        if isinstance(value, list):
            fragments.extend(str(item).strip() for item in value if str(item).strip())
    for region in block.ocr_regions:
        text = str(region.text or "").strip()
        if text:
            fragments.append(text)
    return fragments


def first_image_path(blocks: list[CanonicalBlock]) -> str:
    for block in blocks:
        image_path = str(block.content.get("img_path", "") or "").strip()
        if image_path:
            return image_path
    return ""


def merge_bboxes(bboxes: list[list[int]]) -> list[int]:
    valid_bboxes = [bbox for bbox in bboxes if len(bbox) == 4]
    if not valid_bboxes:
        return []
    left = min(int(bbox[0]) for bbox in valid_bboxes)
    top = min(int(bbox[1]) for bbox in valid_bboxes)
    right = max(int(bbox[2]) for bbox in valid_bboxes)
    bottom = max(int(bbox[3]) for bbox in valid_bboxes)
    return [left, top, right, bottom]


def ordered_unique_strings(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(str(value or "").split()).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(str(value).strip())
    return ordered
