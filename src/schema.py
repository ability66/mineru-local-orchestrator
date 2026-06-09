from __future__ import annotations

import copy
from typing import Any, Literal

try:
    from pydantic import BaseModel, Field
except ImportError:
    _MISSING = object()

    class _FieldInfo:
        def __init__(self, default: Any = _MISSING, default_factory: Any = None) -> None:
            self.default = default
            self.default_factory = default_factory

    def Field(default: Any = _MISSING, default_factory: Any = None) -> Any:
        return _FieldInfo(default=default, default_factory=default_factory)

    class BaseModel:
        def __init__(self, **kwargs: Any) -> None:
            annotations = self._collect_annotations()
            for field_name in annotations:
                if field_name in kwargs:
                    value = kwargs[field_name]
                else:
                    value = self._resolve_default(field_name)
                setattr(self, field_name, value)

        @classmethod
        def _collect_annotations(cls) -> dict[str, Any]:
            annotations: dict[str, Any] = {}
            for base in reversed(cls.__mro__):
                annotations.update(getattr(base, "__annotations__", {}))
            return annotations

        @classmethod
        def _resolve_default(cls, field_name: str) -> Any:
            default = getattr(cls, field_name, _MISSING)
            if isinstance(default, _FieldInfo):
                if default.default_factory is not None:
                    return default.default_factory()
                if default.default is not _MISSING:
                    return default.default
            elif default is not _MISSING:
                return copy.deepcopy(default)
            raise TypeError(f"Missing required field: {field_name}")

        def model_dump(self) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for field_name in self._collect_annotations():
                result[field_name] = _dump_value(getattr(self, field_name))
            return result

        def model_copy(self, deep: bool = False) -> "BaseModel":
            return copy.deepcopy(self) if deep else copy.copy(self)


    def _dump_value(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump()
        if isinstance(value, list):
            return [_dump_value(item) for item in value]
        if isinstance(value, dict):
            return {key: _dump_value(item) for key, item in value.items()}
        return value


ImageType = Literal[
    "natural_image",
    "chart",
    "table",
    "flowchart",
    "seal",
    "document",
    "screenshot",
    "diagram",
    "mixed",
    "unknown",
]

StructuredKind = Literal["none", "table", "mermaid", "text"]
StructuredFormat = Literal["markdown", "csv", "html", "mermaid", "plain_text", "none"]
StructuredSource = Literal["model", "fused_graph", "mineru", "none"]
DecisionType = Literal["accepted", "review", "failed"]
CaptionSource = Literal["generated"]
CaptionConfidence = Literal["low", "medium", "high"]
OcrRegionRole = Literal["seal", "watermark", "footer", "body", "title", "other"]
OcrRegionConfidence = Literal["low", "medium", "high"]
IssueType = Literal[
    "seal_missing_ocr",
    "seal_type_disagreement",
    "seal_ocr_conflict",
    "seal_unmatched_qwen_candidate",
    "flowchart_graph_conflict",
    "flowchart_candidate_review",
    "html_table_conflict",
]
PatchDecisionType = Literal[
    "keep_mineru",
    "keep_candidate",
    "use_qwen_fields",
    "merge",
    "add_qwen_block",
    "reject_issue",
]


class ImageTask(BaseModel):
    image_id: str
    image_path: str
    file_name: str
    file_ext: str


class StructuredLabel(BaseModel):
    kind: StructuredKind = "none"
    content: str = ""
    format: StructuredFormat = "none"
    source: StructuredSource = "model"
    graph_confidence: float | None = None


class CaptionStructured(BaseModel):
    brief: str = ""
    visual_type: str = ""
    main_subject: str = ""
    visible_title: str = ""
    key_visible_text: list[str] = Field(default_factory=list)
    structure_summary: str = ""
    caption_source: CaptionSource = "generated"
    confidence: CaptionConfidence = "medium"


class OcrRegion(BaseModel):
    role: OcrRegionRole = "other"
    text: str = ""
    bbox_hint: list[float] | None = None
    confidence: OcrRegionConfidence = "medium"


class ParsedLabel(BaseModel):
    image_type: ImageType = "unknown"
    caption: str = ""
    caption_structured: CaptionStructured = Field(default_factory=CaptionStructured)
    structured_label: StructuredLabel = Field(default_factory=StructuredLabel)
    flowchart_graph: dict[str, Any] | None = None
    visible_text: list[str] = Field(default_factory=list)
    ocr_regions: list[OcrRegion] = Field(default_factory=list)
    uncertainty: str = ""
    warnings: list[str] = Field(default_factory=list)


class ModelOutput(BaseModel):
    image_id: str
    model_name: str
    success: bool
    raw_text: str = ""
    parsed: Any | None = None
    error: str | None = None
    latency_ms: int | None = None
    vendor: str | None = None
    source_type: str | None = None


class ConsensusResult(BaseModel):
    image_id: str
    type_agreement: float
    caption_agreement: float
    structure_agreement: float
    seal_agreement: float = 1.0
    overall_score: float
    evidence_score: float = 0.0
    validator_score: float = 0.0
    hallucination_risk: float = 0.0
    accept_score: float = 0.0
    decision: DecisionType
    reasons: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)
    escalation_reasons: list[str] = Field(default_factory=list)


MinerUBlockType = Literal[
    "title",
    "paragraph",
    "table",
    "chart",
    "image",
    "equation_interline",
    "code",
    "algorithm",
    "list",
    "page_header",
    "page_footer",
    "page_number",
    "page_aside_text",
    "page_footnote",
]


class CanonicalSpan(BaseModel):
    type: str = "text"
    content: str = ""
    url: str | None = None
    children: list[dict[str, Any]] = Field(default_factory=list)


class CanonicalBlock(BaseModel):
    block_id: str
    page_idx: int = 0
    order_index: int = 0
    type: MinerUBlockType = "paragraph"
    sub_type: str | None = None
    bbox: list[int] = Field(default_factory=list)
    text: str = ""
    text_level: int | None = None
    content: dict[str, Any] = Field(default_factory=dict)
    source: str = ""
    confidence: float | None = None
    structured_label: StructuredLabel = Field(default_factory=StructuredLabel)
    caption_structured: CaptionStructured = Field(default_factory=CaptionStructured)
    flowchart_graph: dict[str, Any] | None = None
    visible_text: list[str] = Field(default_factory=list)
    ocr_regions: list[OcrRegion] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


class CanonicalDocument(BaseModel):
    document_id: str
    source: str
    backend: str = "unknown"
    page_count: int = 1
    blocks: list[CanonicalBlock] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


class Issue(BaseModel):
    issue_id: str
    issue_type: IssueType
    page_idx: int = 0
    target_block_id: str | None = None
    mineru_block: dict[str, Any] | None = None
    qwen_block: dict[str, Any] | None = None
    candidate_payload: dict[str, Any] | None = None
    reasons: list[str] = Field(default_factory=list)


class PatchDecision(BaseModel):
    issue_id: str
    target_block_id: str | None = None
    decision: PatchDecisionType = "keep_mineru"
    patch: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class SealSelectionDecision(BaseModel):
    selected_candidate: str = "review"
    reason: str = ""
    confidence: str = "low"


class AdjudicationArtifact(BaseModel):
    image_id: str
    final_document: CanonicalDocument
    consensus: ConsensusResult | None = None
    final_label: ParsedLabel | None = None
    graph_fusion: dict[str, Any] | None = None
    matched_block_count: int = 0
    added_qwen_block_count: int = 0
    review_required: bool = False
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    patch_decisions: list[PatchDecision] = Field(default_factory=list)
    seal_selection: SealSelectionDecision | None = None
