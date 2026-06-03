from __future__ import annotations

import argparse
import base64
import json
import shutil
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from src.pipeline.flowchart_utils import looks_like_mermaid, mermaid_from_flowchart_graph

REPO_ROOT = Path(__file__).resolve().parent.parent
MERMAID_VENDOR_PATH = REPO_ROOT / "vendor" / "mermaid" / "mermaid.min.js"
IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


@dataclass
class MermaidSnapshot:
    title: str
    source_path: str
    code: str
    render_code: str
    origin: str
    status: str
    note: str = ""


@dataclass
class OriginalImageSnapshot:
    title: str
    source_path: str
    data_url: str
    note: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an offline HTML page to compare MinerU/Qwen/Final Mermaid outputs."
    )
    parser.add_argument("--image-id", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--compare-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_dir = args.compare_dir or (args.output_dir / "compare_mermaid")
    output_path = generate_compare_page(
        image_id=args.image_id,
        output_dir=args.output_dir,
        compare_dir=compare_dir,
    )
    print(output_path)


def generate_compare_page(
    image_id: str,
    output_dir: Path,
    compare_dir: Path,
) -> Path:
    snapshots = collect_mermaid_snapshots(image_id=image_id, output_dir=output_dir)
    original_image = collect_original_image_snapshot(image_id=image_id, output_dir=output_dir)
    compare_dir.mkdir(parents=True, exist_ok=True)
    asset_rel_path = ensure_mermaid_asset(compare_dir=compare_dir)
    html_path = compare_dir / f"{image_id}.html"
    html_path.write_text(
        build_compare_html(
            image_id=image_id,
            snapshots=snapshots,
            original_image=original_image,
            mermaid_script_path=asset_rel_path.as_posix(),
        ),
        encoding="utf-8",
    )
    return html_path


def collect_mermaid_snapshots(image_id: str, output_dir: Path) -> list[MermaidSnapshot]:
    normalized_dir = output_dir / "normalized"
    final_dir = output_dir / "final"

    mineru_payload = _load_json(normalized_dir / "mineru" / f"{image_id}.json")
    qwen_payload = _load_json(normalized_dir / "qwen" / f"{image_id}.json")
    final_payload = _load_json(final_dir / f"{image_id}.json")
    legacy_content_list_v2 = _load_json(final_dir / f"{image_id}_content_list_v2.json")
    legacy_content_list = _load_json(final_dir / f"{image_id}_content_list.json")
    artifact_payload = _load_json(final_dir / f"{image_id}_artifact.json")

    mineru_snapshot = _snapshot_from_normalized_payload(
        payload=mineru_payload,
        title="MinerU",
        source_path=f"normalized/mineru/{image_id}.json",
    )
    if mineru_snapshot.status == "missing":
        mineru_snapshot = _snapshot_from_artifact_issue_block(
            payload=artifact_payload,
            title="MinerU",
            source_path=f"final/{image_id}_artifact.json",
            block_key="mineru_block",
            note_prefix="从 artifact.issue.mineru_block 回退",
        )

    qwen_snapshot = _snapshot_from_normalized_payload(
        payload=qwen_payload,
        title="Qwen",
        source_path=f"normalized/qwen/{image_id}.json",
    )
    if qwen_snapshot.status == "missing":
        qwen_snapshot = _snapshot_from_artifact_issue_block(
            payload=artifact_payload,
            title="Qwen",
            source_path=f"final/{image_id}_artifact.json",
            block_key="qwen_block",
            note_prefix="从 artifact.issue.qwen_block 回退",
        )

    return [
        mineru_snapshot,
        qwen_snapshot,
        _snapshot_from_artifact_payload(
            payload=artifact_payload,
            title="Fusion Candidate",
            source_path=f"final/{image_id}_artifact.json",
        ),
        _snapshot_from_final_payload(
            payload=final_payload,
            artifact_payload=artifact_payload,
            legacy_content_list_v2=legacy_content_list_v2,
            legacy_content_list=legacy_content_list,
            title="Final",
            source_path=(
                f"final/{image_id}.json"
                if final_payload is not None
                else (
                    f"final/{image_id}_content_list_v2.json"
                    if legacy_content_list_v2 is not None
                    else f"final/{image_id}_content_list.json"
                )
            ),
        ),
    ]


def collect_original_image_snapshot(image_id: str, output_dir: Path) -> OriginalImageSnapshot | None:
    normalized_dir = output_dir / "normalized"
    final_dir = output_dir / "final"

    mineru_payload = _load_json(normalized_dir / "mineru" / f"{image_id}.json")
    qwen_payload = _load_json(normalized_dir / "qwen" / f"{image_id}.json")
    artifact_payload = _load_json(final_dir / f"{image_id}_artifact.json")
    final_payload = _load_json(final_dir / f"{image_id}.json")

    candidate_paths: list[tuple[str, Path]] = []
    for label, payload in (
        ("normalized/mineru", mineru_payload),
        ("normalized/qwen", qwen_payload),
    ):
        blocks = _safe_blocks_from_document((payload or {}).get("document") if isinstance(payload, dict) else None)
        path = _extract_img_path_from_blocks(blocks)
        if path is not None:
            candidate_paths.append((label, path))

    if isinstance(artifact_payload, dict):
        final_document = artifact_payload.get("final_document")
        path = _extract_img_path_from_blocks(_safe_blocks_from_document(final_document))
        if path is not None:
            candidate_paths.append(("artifact.final_document", path))

    if isinstance(final_payload, dict):
        parsed = final_payload.get("parsed")
        filename = str((parsed or {}).get("filename", "") or "").strip() if isinstance(parsed, dict) else ""
        if filename:
            guessed = _guess_image_path_from_filename(filename)
            if guessed is not None:
                candidate_paths.append(("final.parsed.filename", guessed))

    for source_path, image_path in candidate_paths:
        if image_path.exists() and image_path.is_file():
            data_url = _build_image_data_url(image_path)
            if data_url:
                return OriginalImageSnapshot(
                    title="Original Image",
                    source_path=str(image_path),
                    data_url=data_url,
                    note=f"来源：{source_path}",
                )
    return None


def ensure_mermaid_asset(compare_dir: Path) -> Path:
    if not MERMAID_VENDOR_PATH.exists():
        raise FileNotFoundError(f"Missing local Mermaid asset: {MERMAID_VENDOR_PATH}")

    asset_dir = compare_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    target = asset_dir / "mermaid.min.js"
    shutil.copy2(MERMAID_VENDOR_PATH, target)
    return Path("assets") / "mermaid.min.js"


def build_compare_html(
    image_id: str,
    snapshots: list[MermaidSnapshot],
    original_image: OriginalImageSnapshot | None,
    mermaid_script_path: str,
) -> str:
    original_image_html = _build_original_image_card(original_image)
    cards_html = "\n".join(_build_snapshot_card(index=index, snapshot=snapshot) for index, snapshot in enumerate(snapshots))
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mermaid Compare - {escape(image_id)}</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: #fffdf8;
      --ink: #1f1f1a;
      --muted: #6f6a5f;
      --line: #d5ccb8;
      --accent: #14532d;
      --warn: #9a3412;
      --error: #991b1b;
      --shadow: 0 18px 45px rgba(31, 31, 26, 0.08);
      --code-bg: #f6f0e2;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Noto Serif SC", serif;
      background:
        radial-gradient(circle at top left, rgba(20, 83, 45, 0.08), transparent 28%),
        linear-gradient(180deg, #f7f2e9 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .page {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    .hero {{
      margin-bottom: 24px;
      padding: 24px 28px;
      background: rgba(255, 253, 248, 0.9);
      border: 1px solid rgba(213, 204, 184, 0.8);
      box-shadow: var(--shadow);
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 30px;
      line-height: 1.1;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      font-size: 15px;
    }}
    .image-panel {{
      margin-bottom: 24px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
      align-items: start;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .card-head {{
      padding: 18px 20px 14px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(20, 83, 45, 0.06), rgba(255,255,255,0));
    }}
    .card-head h2 {{
      margin: 0 0 8px;
      font-size: 20px;
    }}
    .meta {{
      display: grid;
      gap: 6px;
      font-size: 13px;
      color: var(--muted);
    }}
    .badge {{
      display: inline-block;
      width: fit-content;
      padding: 4px 8px;
      font-size: 12px;
      border: 1px solid currentColor;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .status-valid {{ color: var(--accent); }}
    .status-derived {{ color: #1d4ed8; }}
    .status-invalid {{ color: var(--warn); }}
    .status-missing {{ color: var(--error); }}
    .diagram-wrap {{
      min-height: 320px;
      padding: 16px;
      background:
        linear-gradient(90deg, rgba(213, 204, 184, 0.24) 1px, transparent 1px) 0 0 / 18px 18px,
        linear-gradient(rgba(213, 204, 184, 0.24) 1px, transparent 1px) 0 0 / 18px 18px,
        #fff;
      border-bottom: 1px solid var(--line);
    }}
    .diagram-frame {{
      min-height: 288px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(255, 253, 248, 0.92);
      border: 1px dashed rgba(213, 204, 184, 0.9);
      padding: 12px;
      overflow: auto;
    }}
    .image-frame {{
      min-height: 240px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(255, 253, 248, 0.92);
      border: 1px dashed rgba(213, 204, 184, 0.9);
      padding: 12px;
      overflow: auto;
    }}
    .image-frame img {{
      max-width: 100%;
      max-height: 720px;
      object-fit: contain;
      box-shadow: 0 10px 28px rgba(31, 31, 26, 0.16);
      background: #fff;
    }}
    .diagram-empty,
    .diagram-error {{
      width: 100%;
      text-align: left;
      white-space: pre-wrap;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.6;
    }}
    .diagram-error {{ color: var(--error); }}
    .code-wrap {{
      padding: 16px;
    }}
    .code-wrap h3 {{
      margin: 0 0 10px;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    pre {{
      margin: 0;
      padding: 14px;
      background: var(--code-bg);
      border: 1px solid rgba(213, 204, 184, 0.9);
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 13px;
      line-height: 1.55;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Flowchart Mermaid 对比页</h1>
      <p>对比 {escape(image_id)} 的 MinerU、Qwen、Fusion Candidate 与 Final 流程图结果。页面完全离线，Mermaid 渲染脚本使用本地资源。</p>
    </section>
    <section class="image-panel">
      {original_image_html}
    </section>
    <section class="grid">
      {cards_html}
    </section>
  </div>
  <script src="{escape(mermaid_script_path)}"></script>
  <script>
    (async function () {{
      if (!window.mermaid) {{
        return;
      }}
      window.mermaid.initialize({{
        startOnLoad: false,
        securityLevel: "loose",
        theme: "neutral",
        flowchart: {{ curve: "basis" }}
      }});

      const containers = document.querySelectorAll("[data-mermaid-b64]");
      for (const container of containers) {{
        const encoded = container.getAttribute("data-mermaid-b64") || "";
        if (!encoded) {{
          continue;
        }}
        const mermaidCode = decodeURIComponent(escape(window.atob(encoded)));
        if (!mermaidCode.trim()) {{
          continue;
        }}
        try {{
          const renderId = "mermaid-" + Math.random().toString(36).slice(2);
          const rendered = await window.mermaid.render(renderId, mermaidCode);
          container.innerHTML = rendered.svg;
          if (rendered.bindFunctions) {{
            rendered.bindFunctions(container);
          }}
        }} catch (error) {{
          container.innerHTML = '<div class="diagram-error">Mermaid 渲染失败\\n' + String(error) + '</div>';
        }}
      }}
    }})();
  </script>
</body>
</html>
"""


def _build_original_image_card(snapshot: OriginalImageSnapshot | None) -> str:
    if snapshot is None:
        return """
      <article class="card">
        <div class="card-head">
          <h2>Original Image</h2>
          <div class="meta">
            <span class="badge status-missing">missing</span>
            <span>未能定位原始图像路径</span>
          </div>
        </div>
        <div class="diagram-wrap">
          <div class="image-frame"><div class="diagram-empty">当前没有可展示的原始图像。</div></div>
        </div>
      </article>
    """
    return f"""
      <article class="card">
        <div class="card-head">
          <h2>{escape(snapshot.title)}</h2>
          <div class="meta">
            <span class="badge status-valid">embedded image</span>
            <span>文件：{escape(snapshot.source_path)}</span>
            <span>说明：{escape(snapshot.note or '原始输入图像')}</span>
          </div>
        </div>
        <div class="diagram-wrap">
          <div class="image-frame">
            <img src="{snapshot.data_url}" alt="{escape(snapshot.title)}" />
          </div>
        </div>
      </article>
    """


def _build_snapshot_card(index: int, snapshot: MermaidSnapshot) -> str:
    render_b64 = base64.b64encode(snapshot.render_code.encode("utf-8")).decode("ascii") if snapshot.render_code else ""
    status_class = {
        "valid": "status-valid",
        "derived": "status-derived",
        "invalid": "status-invalid",
        "missing": "status-missing",
    }.get(snapshot.status, "status-missing")
    badge_text = {
        "valid": "valid mermaid",
        "derived": "derived graph",
        "invalid": "invalid mermaid",
        "missing": "missing",
    }.get(snapshot.status, snapshot.status)
    diagram_html = (
        f'<div class="diagram-frame" data-mermaid-b64="{render_b64}"></div>'
        if snapshot.render_code
        else '<div class="diagram-frame"><div class="diagram-empty">当前没有可渲染的 Mermaid 图。</div></div>'
    )
    code_text = snapshot.code or "(empty)"
    note = escape(snapshot.note or "无补充说明")
    return f"""
      <article class="card">
        <div class="card-head">
          <h2>{escape(snapshot.title)}</h2>
          <div class="meta">
            <span class="badge {status_class}">{escape(badge_text)}</span>
            <span>文件：{escape(snapshot.source_path)}</span>
            <span>提取位置：{escape(snapshot.origin)}</span>
            <span>说明：{note}</span>
          </div>
        </div>
        <div class="diagram-wrap">
          {diagram_html}
        </div>
        <div class="code-wrap">
          <h3>Mermaid / Raw Text</h3>
          <pre id="code-{index}">{escape(code_text)}</pre>
        </div>
      </article>
    """


def _snapshot_from_normalized_payload(payload: Any, title: str, source_path: str) -> MermaidSnapshot:
    if not isinstance(payload, dict):
        return MermaidSnapshot(
            title=title,
            source_path=source_path,
            code="",
            render_code="",
            origin="missing_file",
            status="missing",
            note="未找到标准化输出文件",
        )

    snapshot = _extract_from_document_payload(
        title=title,
        source_path=source_path,
        blocks=_safe_blocks_from_document(payload.get("document")),
    )
    if snapshot.status in {"valid", "derived"}:
        return snapshot

    derived_label = payload.get("derived_label")
    label_snapshot = _extract_from_label_payload(
        title=title,
        source_path=source_path,
        label_payload=derived_label,
    )
    if label_snapshot.status in {"valid", "derived"}:
        return label_snapshot
    if snapshot.status == "invalid":
        return snapshot
    return label_snapshot


def _snapshot_from_final_payload(
    payload: Any,
    artifact_payload: Any,
    legacy_content_list_v2: Any,
    legacy_content_list: Any,
    title: str,
    source_path: str,
) -> MermaidSnapshot:
    if isinstance(payload, dict):
        parsed = payload.get("parsed")
        extraction_results = parsed.get("extraction_results") if isinstance(parsed, dict) else None
        blocks: list[dict[str, Any]] = []
        if isinstance(extraction_results, list):
            for page in extraction_results:
                if not isinstance(page, dict):
                    continue
                page_blocks = page.get("json_res")
                if isinstance(page_blocks, list):
                    blocks.extend(item for item in page_blocks if isinstance(item, dict))
        snapshot = _extract_from_document_payload(title=title, source_path=source_path, blocks=blocks)
        if snapshot.status != "missing":
            return snapshot

    legacy_blocks_v2 = _flatten_content_list_v2(legacy_content_list_v2)
    if legacy_blocks_v2:
        snapshot = _extract_from_document_payload(
            title=title,
            source_path=source_path,
            blocks=legacy_blocks_v2,
        )
        if snapshot.status != "missing":
            snapshot.note = f"{snapshot.note}；来自 legacy content_list_v2".strip("；")
            return snapshot

    if isinstance(legacy_content_list, list):
        legacy_blocks = [item for item in legacy_content_list if isinstance(item, dict)]
        snapshot = _extract_from_document_payload(
            title=title,
            source_path=source_path,
            blocks=legacy_blocks,
        )
        if snapshot.status != "missing":
            snapshot.note = f"{snapshot.note}；来自 legacy content_list".strip("；")
            return snapshot

    if isinstance(artifact_payload, dict):
        final_document = artifact_payload.get("final_document")
        artifact_blocks = _safe_blocks_from_document(final_document)
        snapshot = _extract_from_document_payload(
            title=title,
            source_path=f"{source_path} (artifact fallback)",
            blocks=artifact_blocks,
        )
        if snapshot.status != "missing":
            snapshot.note = f"{snapshot.note}；来自 artifact.final_document 回退".strip("；")
            return snapshot

    return MermaidSnapshot(
        title=title,
        source_path=source_path,
        code="",
        render_code="",
        origin="missing_file",
        status="missing",
        note="未找到 final 输出文件，且 legacy / artifact 回退也没有流程图内容",
    )


def _snapshot_from_artifact_payload(payload: Any, title: str, source_path: str) -> MermaidSnapshot:
    if not isinstance(payload, dict):
        return MermaidSnapshot(
            title=title,
            source_path=source_path,
            code="",
            render_code="",
            origin="missing_file",
            status="missing",
            note="未找到 artifact 输出文件",
        )

    graph_fusion = payload.get("graph_fusion")
    if isinstance(graph_fusion, dict):
        mermaid = str(graph_fusion.get("mermaid", "") or "").strip()
        normalized_mermaid = _normalize_mermaid_text(mermaid)
        if looks_like_mermaid(normalized_mermaid):
            return MermaidSnapshot(
                title=title,
                source_path=source_path,
                code=mermaid,
                render_code=normalized_mermaid,
                origin="graph_fusion.mermaid",
                status="valid",
                note=f"fusion_status={graph_fusion.get('fusion_status', 'unknown')}, fusion_method={graph_fusion.get('fusion_method', 'unknown')}",
            )
        derived = mermaid_from_flowchart_graph(graph_fusion)
        if derived:
            return MermaidSnapshot(
                title=title,
                source_path=source_path,
                code=derived,
                render_code=derived,
                origin="graph_fusion.nodes/edges",
                status="derived",
                note=f"由 graph_fusion 图结构反推，fusion_status={graph_fusion.get('fusion_status', 'unknown')}",
            )

    for issue_index, issue in enumerate(payload.get("issues") or []):
        if not isinstance(issue, dict) or str(issue.get("issue_type", "") or "").strip() != "flowchart_candidate_review":
            continue
        candidate_payload = issue.get("candidate_payload")
        if not isinstance(candidate_payload, dict):
            continue
        candidate_mermaid = str(candidate_payload.get("candidate_mermaid", "") or "").strip()
        normalized_candidate_mermaid = _normalize_mermaid_text(candidate_mermaid)
        if looks_like_mermaid(normalized_candidate_mermaid):
            return MermaidSnapshot(
                title=title,
                source_path=source_path,
                code=candidate_mermaid,
                render_code=normalized_candidate_mermaid,
                origin=f"issues[{issue_index}].candidate_payload.candidate_mermaid",
                status="valid",
                note="来自流程图二阶段 issue 候选 Mermaid",
            )
        candidate_patch = candidate_payload.get("candidate_patch")
        if isinstance(candidate_patch, dict):
            content_payload = candidate_patch.get("content")
            candidate_patch_mermaid = ""
            if isinstance(content_payload, dict):
                candidate_patch_mermaid = str(content_payload.get("content", "") or "").strip()
            normalized_candidate_patch_mermaid = _normalize_mermaid_text(candidate_patch_mermaid)
            if looks_like_mermaid(normalized_candidate_patch_mermaid):
                return MermaidSnapshot(
                    title=title,
                    source_path=source_path,
                    code=candidate_patch_mermaid,
                    render_code=normalized_candidate_patch_mermaid,
                    origin=f"issues[{issue_index}].candidate_payload.candidate_patch.content.content",
                    status="valid",
                    note="来自流程图二阶段 issue 候选 patch",
                )
            flowchart_graph = candidate_patch.get("flowchart_graph")
            derived = mermaid_from_flowchart_graph(flowchart_graph if isinstance(flowchart_graph, dict) else None)
            if derived:
                return MermaidSnapshot(
                    title=title,
                    source_path=source_path,
                    code=derived,
                    render_code=derived,
                    origin=f"issues[{issue_index}].candidate_payload.candidate_patch.flowchart_graph",
                    status="derived",
                    note="由流程图二阶段 issue 候选 graph 反推 Mermaid",
                )
        qwen_mermaid = str(candidate_payload.get("qwen_mermaid", "") or "").strip()
        normalized_qwen_mermaid = _normalize_mermaid_text(qwen_mermaid)
        if looks_like_mermaid(normalized_qwen_mermaid):
            return MermaidSnapshot(
                title=title,
                source_path=source_path,
                code=qwen_mermaid,
                render_code=normalized_qwen_mermaid,
                origin=f"issues[{issue_index}].candidate_payload.qwen_mermaid",
                status="valid",
                note="graph_fusion 缺失，回退使用 qwen_mermaid 作为候选参考",
            )

    return MermaidSnapshot(
        title=title,
        source_path=source_path,
        code="",
        render_code="",
        origin="graph_fusion",
        status="missing",
        note="artifact 中没有可渲染的流程图候选",
    )


def _extract_from_document_payload(
    title: str,
    source_path: str,
    blocks: list[dict[str, Any]],
) -> MermaidSnapshot:
    invalid_snapshot: MermaidSnapshot | None = None

    for index, block in enumerate(blocks):
        if not _is_flowchart_like_block(block):
            continue

        content_text = _extract_block_content_text(block)
        normalized_content_text = _normalize_mermaid_text(content_text)
        if looks_like_mermaid(normalized_content_text):
            return MermaidSnapshot(
                title=title,
                source_path=source_path,
                code=content_text,
                render_code=normalized_content_text,
                origin=f"blocks[{index}].content",
                status="valid",
                note="直接来自块内容",
            )

        structured_content = _extract_structured_content(block)
        normalized_structured_content = _normalize_mermaid_text(structured_content)
        if looks_like_mermaid(normalized_structured_content):
            return MermaidSnapshot(
                title=title,
                source_path=source_path,
                code=structured_content,
                render_code=normalized_structured_content,
                origin=f"blocks[{index}].structured_label.content",
                status="valid",
                note="直接来自 structured_label",
            )

        text_content = str(block.get("text", "") or "").strip()
        normalized_text_content = _normalize_mermaid_text(text_content)
        if looks_like_mermaid(normalized_text_content):
            return MermaidSnapshot(
                title=title,
                source_path=source_path,
                code=text_content,
                render_code=normalized_text_content,
                origin=f"blocks[{index}].text",
                status="valid",
                note="直接来自块文本",
            )

        flowchart_graph = block.get("flowchart_graph")
        derived_mermaid = mermaid_from_flowchart_graph(flowchart_graph if isinstance(flowchart_graph, dict) else None)
        if derived_mermaid:
            return MermaidSnapshot(
                title=title,
                source_path=source_path,
                code=derived_mermaid,
                render_code=derived_mermaid,
                origin=f"blocks[{index}].flowchart_graph",
                status="derived",
                note="由 flowchart_graph 反推 Mermaid",
            )

        raw_text = content_text or structured_content or text_content or _extract_caption_text(block)
        if raw_text and invalid_snapshot is None:
            invalid_snapshot = MermaidSnapshot(
                title=title,
                source_path=source_path,
                code=raw_text,
                render_code="",
                origin=f"blocks[{index}]",
                status="invalid",
                note="识别成流程图，但文本未通过 Mermaid 校验",
            )

    return invalid_snapshot or MermaidSnapshot(
        title=title,
        source_path=source_path,
        code="",
        render_code="",
        origin="blocks",
        status="missing",
        note="未找到流程图相关块",
    )


def _extract_from_label_payload(
    title: str,
    source_path: str,
    label_payload: Any,
) -> MermaidSnapshot:
    if not isinstance(label_payload, dict):
        return MermaidSnapshot(
            title=title,
            source_path=source_path,
            code="",
            render_code="",
            origin="derived_label",
            status="missing",
            note="没有 derived_label 可用",
        )

    structured = label_payload.get("structured_label")
    structured_content = str((structured or {}).get("content", "") or "").strip() if isinstance(structured, dict) else ""
    normalized_structured_content = _normalize_mermaid_text(structured_content)
    if looks_like_mermaid(normalized_structured_content):
        return MermaidSnapshot(
            title=title,
            source_path=source_path,
            code=structured_content,
            render_code=normalized_structured_content,
            origin="derived_label.structured_label.content",
            status="valid",
            note="来自 derived_label",
        )

    flowchart_graph = label_payload.get("flowchart_graph")
    derived_mermaid = mermaid_from_flowchart_graph(flowchart_graph if isinstance(flowchart_graph, dict) else None)
    if derived_mermaid:
        return MermaidSnapshot(
            title=title,
            source_path=source_path,
            code=derived_mermaid,
            render_code=derived_mermaid,
            origin="derived_label.flowchart_graph",
            status="derived",
            note="由 derived_label.flowchart_graph 反推 Mermaid",
        )

    caption = str(label_payload.get("caption", "") or "").strip()
    normalized_caption = _normalize_mermaid_text(caption)
    if looks_like_mermaid(normalized_caption):
        return MermaidSnapshot(
            title=title,
            source_path=source_path,
            code=caption,
            render_code=normalized_caption,
            origin="derived_label.caption",
            status="valid",
            note="来自 derived_label.caption",
        )

    if str(label_payload.get("image_type", "") or "").strip().lower() == "flowchart" and (structured_content or caption):
        return MermaidSnapshot(
            title=title,
            source_path=source_path,
            code=structured_content or caption,
            render_code="",
            origin="derived_label",
            status="invalid",
            note="derived_label 认为是流程图，但可用文本不是合法 Mermaid",
        )

    return MermaidSnapshot(
        title=title,
        source_path=source_path,
        code="",
        render_code="",
        origin="derived_label",
        status="missing",
        note="derived_label 中没有 Mermaid 或 flowchart_graph",
    )


def _safe_blocks_from_document(document_payload: Any) -> list[dict[str, Any]]:
    if not isinstance(document_payload, dict):
        return []
    blocks = document_payload.get("blocks")
    if not isinstance(blocks, list):
        return []
    return [item for item in blocks if isinstance(item, dict)]


def _snapshot_from_artifact_issue_block(
    payload: Any,
    title: str,
    source_path: str,
    block_key: str,
    note_prefix: str,
) -> MermaidSnapshot:
    if not isinstance(payload, dict):
        return MermaidSnapshot(
            title=title,
            source_path=source_path,
            code="",
            render_code="",
            origin="missing_file",
            status="missing",
            note="未找到 artifact 输出文件",
        )

    blocks: list[dict[str, Any]] = []
    for issue in payload.get("issues") or []:
        if not isinstance(issue, dict):
            continue
        block = issue.get(block_key)
        if isinstance(block, dict):
            blocks.append(block)
    snapshot = _extract_from_document_payload(
        title=title,
        source_path=source_path,
        blocks=blocks,
    )
    if snapshot.status != "missing":
        snapshot.note = f"{note_prefix}；{snapshot.note}".strip("；")
    return snapshot


def _flatten_content_list_v2(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        return []
    blocks: list[dict[str, Any]] = []
    for page in payload:
        if not isinstance(page, list):
            continue
        blocks.extend(item for item in page if isinstance(item, dict))
    return blocks


def _is_flowchart_like_block(block: dict[str, Any]) -> bool:
    sub_type = str(block.get("sub_type", "") or "").strip().lower()
    if sub_type == "flowchart":
        return True
    structured_label = block.get("structured_label")
    if isinstance(structured_label, dict) and str(structured_label.get("kind", "") or "").strip().lower() == "mermaid":
        return True
    return isinstance(block.get("flowchart_graph"), dict)


def _extract_block_content_text(block: dict[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, dict):
        direct = str(content.get("content", "") or "").strip()
        if direct:
            return direct
        for key in ("image_caption", "chart_caption"):
            values = content.get(key)
            if isinstance(values, list):
                for item in values:
                    text = str(item or "").strip()
                    if text:
                        return text
    if isinstance(content, str):
        return content.strip()
    return ""


def _extract_structured_content(block: dict[str, Any]) -> str:
    structured = block.get("structured_label")
    if not isinstance(structured, dict):
        return ""
    return str(structured.get("content", "") or "").strip()


def _extract_caption_text(block: dict[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, dict):
        for key in ("chart_caption", "image_caption"):
            captions = content.get(key)
            if isinstance(captions, list):
                text = " ".join(str(item).strip() for item in captions if str(item).strip()).strip()
                if text:
                    return text
    return str(block.get("text", "") or "").strip()


def _extract_img_path_from_blocks(blocks: list[dict[str, Any]]) -> Path | None:
    for block in blocks:
        content = block.get("content")
        if not isinstance(content, dict):
            continue
        img_path = str(content.get("img_path", "") or "").strip()
        if img_path:
            return Path(img_path)
    return None


def _guess_image_path_from_filename(file_name: str) -> Path | None:
    candidate = REPO_ROOT / "data" / file_name
    if candidate.exists():
        return candidate
    matches = sorted((REPO_ROOT / "data").rglob(file_name))
    return matches[0] if matches else None


def _build_image_data_url(image_path: Path) -> str | None:
    suffix = image_path.suffix.lower()
    mime_type = IMAGE_MIME_TYPES.get(suffix)
    if mime_type is None:
        return None
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _normalize_mermaid_text(text: str) -> str:
    value = str(text or "").strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    return value


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
