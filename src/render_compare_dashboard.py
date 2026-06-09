from __future__ import annotations

import argparse
import base64
import json
import re
from dataclasses import dataclass
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from src.pipeline.flowchart_utils import (
    looks_like_mermaid,
    mermaid_from_flowchart_graph,
    normalize_mermaid_text,
)
from src.render_mermaid_compare import (
    collect_mermaid_snapshots,
    collect_original_image_snapshot,
    ensure_mermaid_asset,
)


@dataclass
class ComparePanel:
    title: str
    source_path: str
    image_type: str
    caption: str
    render_kind: str
    render_text: str
    note: str = ""


TYPE_LABELS = {
    "flowchart": "流程图",
    "chart": "图表",
    "table": "表格",
    "seal": "印章",
    "natural_image": "自然图像",
    "document": "文档",
    "unknown": "未知类型",
}

_ALLOWED_TABLE_HTML_TAGS = {
    "table",
    "thead",
    "tbody",
    "tfoot",
    "tr",
    "th",
    "td",
    "caption",
    "colgroup",
    "col",
    "div",
    "span",
    "p",
    "br",
    "sub",
    "sup",
    "strong",
    "em",
    "b",
    "i",
}
_ALLOWED_TABLE_HTML_ATTRS = {
    "table": {"class"},
    "thead": {"class"},
    "tbody": {"class"},
    "tfoot": {"class"},
    "tr": {"class"},
    "th": {"class", "colspan", "rowspan", "scope", "align"},
    "td": {"class", "colspan", "rowspan", "align"},
    "caption": {"class"},
    "colgroup": {"class", "span"},
    "col": {"class", "span"},
    "div": {"class"},
    "span": {"class"},
    "p": {"class"},
}
_SKIPPED_TABLE_HTML_TAGS = {"script", "style", "iframe", "object", "embed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an offline HTML dashboard to compare original image, labels, and outputs."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--dashboard-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dashboard_dir = args.dashboard_dir or (args.output_dir / "compare_dashboard")
    output_path = generate_compare_dashboard(
        output_dir=args.output_dir, dashboard_dir=dashboard_dir
    )
    if output_path is not None:
        print(output_path)


def generate_compare_dashboard(output_dir: Path, dashboard_dir: Path) -> Path | None:
    image_ids = discover_image_ids(output_dir=output_dir)
    if not image_ids:
        return None

    records = [
        collect_compare_record(image_id=image_id, output_dir=output_dir)
        for image_id in image_ids
    ]
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    asset_rel_path = ensure_mermaid_asset(compare_dir=dashboard_dir)
    html_path = dashboard_dir / "index.html"
    html_path.write_text(
        build_dashboard_html(
            records=records, mermaid_script_path=asset_rel_path.as_posix()
        ),
        encoding="utf-8",
    )
    return html_path


def discover_image_ids(output_dir: Path) -> list[str]:
    image_ids: set[str] = set()
    normalized_dir = output_dir / "normalized"
    final_dir = output_dir / "final"

    for provider in ("mineru", "paddle", "glm", "qwen"):
        provider_dir = normalized_dir / provider
        if not provider_dir.exists():
            continue
        for path in provider_dir.glob("*.json"):
            image_ids.add(path.stem)

    if final_dir.exists():
        for path in final_dir.glob("*.json"):
            stem = path.stem
            if stem.endswith("_artifact"):
                image_ids.add(stem[: -len("_artifact")])
            elif not stem.endswith("_content_list_v2") and not stem.endswith(
                "_content_list"
            ):
                image_ids.add(stem)

    return sorted(image_ids)


def collect_compare_record(image_id: str, output_dir: Path) -> dict[str, Any]:
    normalized_dir = output_dir / "normalized"
    final_dir = output_dir / "final"

    mineru_payload = _load_json(normalized_dir / "mineru" / f"{image_id}.json")
    paddle_payload = _load_json(normalized_dir / "paddle" / f"{image_id}.json")
    glm_payload = _load_json(normalized_dir / "glm" / f"{image_id}.json")
    qwen_payload = _load_json(normalized_dir / "qwen" / f"{image_id}.json")
    final_payload = _load_json(final_dir / f"{image_id}.json")
    artifact_payload = _load_json(final_dir / f"{image_id}_artifact.json")

    snapshot_lookup = {
        snapshot.title: snapshot
        for snapshot in collect_mermaid_snapshots(
            image_id=image_id, output_dir=output_dir
        )
    }
    original_image = collect_original_image_snapshot(
        image_id=image_id, output_dir=output_dir
    )
    record_type = _infer_record_type(
        artifact_payload=artifact_payload,
        final_payload=final_payload,
        qwen_payload=qwen_payload,
        mineru_payload=mineru_payload,
        paddle_payload=paddle_payload,
        glm_payload=glm_payload,
    )
    panels = [
        _build_panel_from_normalized_payload(
            payload=mineru_payload,
            artifact_payload=artifact_payload,
            snapshot_lookup=snapshot_lookup,
            title="MinerU",
            source_path=f"normalized/mineru/{image_id}.json",
        ),
        _build_panel_from_normalized_payload(
            payload=paddle_payload,
            artifact_payload=artifact_payload,
            snapshot_lookup=snapshot_lookup,
            title="Paddle",
            source_path=f"normalized/paddle/{image_id}.json",
        ),
        _build_panel_from_normalized_payload(
            payload=glm_payload,
            artifact_payload=artifact_payload,
            snapshot_lookup=snapshot_lookup,
            title="GLM",
            source_path=f"normalized/glm/{image_id}.json",
        ),
    ]
    qwen_panel = _build_qwen_panel(
        record_type=record_type,
        payload=qwen_payload,
        artifact_payload=artifact_payload,
        snapshot_lookup=snapshot_lookup,
        source_path=f"normalized/qwen/{image_id}.json",
        artifact_source_path=f"final/{image_id}_artifact.json",
    )
    if qwen_panel is not None:
        panels.append(qwen_panel)
    panels.append(
        _build_panel_from_final_payload(
            final_payload=final_payload,
            artifact_payload=artifact_payload,
            snapshot_lookup=snapshot_lookup,
            title="Final",
            source_path=f"final/{image_id}.json",
        )
    )
    if str(record_type or "").strip().lower() == "flowchart":
        judge_panel = _build_flowchart_judge_panel(
            artifact_payload=artifact_payload,
            artifact_source_path=f"final/{image_id}_artifact.json",
        )
        if judge_panel is not None:
            panels.append(judge_panel)
    panels = _finalize_panels(record_type=record_type, panels=panels)

    return {
        "image_id": image_id,
        "record_type": record_type,
        "original_image": None
        if original_image is None
        else {
            "title": original_image.title,
            "source_path": original_image.source_path,
            "data_url": original_image.data_url,
            "note": original_image.note,
        },
        "panels": panels,
    }


def build_dashboard_html(
    records: list[dict[str, Any]], mermaid_script_path: str
) -> str:
    type_options = _build_type_options(records)
    type_options_html = "\n".join(
        f'<option value="{escape(option["value"])}">{escape(option["label"])}</option>'
        for option in type_options
    )
    options_html = "\n".join(
        f'<option value="{escape(record["image_id"])}">{escape(record["image_id"])}</option>'
        for record in records
    )
    sections_html = "\n".join(_build_record_section(record) for record in records)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Output Compare Dashboard</title>
  <style>
    :root {{
      --bg: #f5efe5;
      --panel: #fffdf8;
      --ink: #1f1f1a;
      --muted: #6f6a5f;
      --line: #d5ccb8;
      --accent: #14532d;
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
      max-width: 1680px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    .hero {{
      margin-bottom: 24px;
      padding: 24px 28px;
      background: rgba(255, 253, 248, 0.92);
      border: 1px solid rgba(213, 204, 184, 0.8);
      box-shadow: var(--shadow);
      display: grid;
      gap: 16px;
    }}
    .hero h1 {{
      margin: 0;
      font-size: 30px;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      font-size: 15px;
    }}
    .controls {{
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .controls label {{
      font-size: 14px;
      color: var(--muted);
    }}
    select {{
      min-width: 280px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      font-size: 15px;
    }}
    .record {{
      display: none;
      margin-bottom: 28px;
    }}
    .record.active {{
      display: block;
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
      padding: 18px 20px 12px;
      background: linear-gradient(180deg, rgba(20, 83, 45, 0.06), rgba(255,255,255,0));
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
    }}
    .card-head h2 {{
      margin: 0;
      font-size: 20px;
    }}
    .card-body {{
      display: grid;
      gap: 0;
    }}
    .meta {{
      display: grid;
      gap: 6px;
      font-size: 13px;
      color: var(--muted);
    }}
    .card-meta {{
      padding: 14px 16px 16px;
      border-top: 1px solid var(--line);
      background: rgba(246, 240, 226, 0.32);
    }}
    .badge {{
      display: inline-block;
      width: fit-content;
      padding: 4px 8px;
      font-size: 12px;
      border: 1px solid currentColor;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--accent);
    }}
    .diagram-wrap {{
      min-height: 320px;
      padding: 16px;
      background:
        linear-gradient(90deg, rgba(213, 204, 184, 0.24) 1px, transparent 1px) 0 0 / 18px 18px,
        linear-gradient(rgba(213, 204, 184, 0.24) 1px, transparent 1px) 0 0 / 18px 18px,
        #fff;
      border-bottom: 1px solid var(--line);
    }}
    .diagram-frame, .image-frame {{
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
    .markdown-body,
    .html-body {{
      display: grid;
      gap: 12px;
      color: var(--ink);
      font-size: 14px;
      line-height: 1.65;
    }}
    .markdown-body p,
    .html-body p {{
      margin: 0;
    }}
    .markdown-table-wrap,
    .html-table-wrap {{
      overflow: auto;
      border: 1px solid rgba(213, 204, 184, 0.9);
      background: rgba(255, 253, 248, 0.92);
    }}
    .markdown-table,
    .html-body table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    .markdown-table th,
    .markdown-table td,
    .html-body th,
    .html-body td {{
      padding: 10px 12px;
      border: 1px solid rgba(213, 204, 184, 0.9);
      text-align: left;
      vertical-align: top;
    }}
    .markdown-table thead th,
    .html-body thead th {{
      background: var(--code-bg);
      font-weight: 700;
    }}
    .markdown-table tbody tr:nth-child(even),
    .html-body tbody tr:nth-child(even) {{
      background: rgba(246, 240, 226, 0.42);
    }}
    .html-body mjx-container {{
      font-size: 1em !important;
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
      <h1>输出对比总览</h1>
      <p>原图、MinerU、Paddle、GLM、Qwen 与 Final 的统一对比面板。流程图渲染 Mermaid，表格优先展示 Markdown，其次兼容 HTML，其它展示文字。</p>
      <div class="controls">
        <label for="type-select">查看类型</label>
        <select id="type-select">{type_options_html}</select>
        <label for="image-select">切换图片</label>
        <select id="image-select">{options_html}</select>
      </div>
    </section>
    {sections_html}
  </div>
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
      }},
      options: {{
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
      }}
    }};
  </script>
  <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
  <script src="{escape(mermaid_script_path)}"></script>
  <script>
    (function () {{
      const typeSelect = document.getElementById("type-select");
      const select = document.getElementById("image-select");
      const sections = Array.from(document.querySelectorAll(".record"));
      const allRecords = sections.map((section) => ({{
        imageId: section.getAttribute("data-image-id") || "",
        recordType: section.getAttribute("data-record-type") || "unknown"
      }}));

      function syncImageOptions(recordType) {{
        if (!select) {{
          return [];
        }}
        const allowedRecords = allRecords.filter((record) => recordType === "all" || record.recordType === recordType);
        const previousValue = select.value;
        select.innerHTML = "";
        for (const record of allowedRecords) {{
          const option = document.createElement("option");
          option.value = record.imageId;
          option.textContent = record.imageId;
          select.appendChild(option);
        }}
        if (!allowedRecords.length) {{
          return [];
        }}
        const nextValue = allowedRecords.some((record) => record.imageId === previousValue)
          ? previousValue
          : allowedRecords[0].imageId;
        select.value = nextValue;
        return allowedRecords;
      }}

      function showRecord(imageId) {{
        for (const section of sections) {{
          section.classList.toggle("active", section.getAttribute("data-image-id") === imageId);
        }}
      }}

      function moveSelection(targetSelect, direction) {{
        if (!targetSelect || !targetSelect.options.length) {{
          return false;
        }}
        const optionCount = targetSelect.options.length;
        const currentIndex = targetSelect.selectedIndex >= 0 ? targetSelect.selectedIndex : 0;
        const nextIndex = (currentIndex + direction + optionCount) % optionCount;
        if (nextIndex === currentIndex) {{
          return false;
        }}
        targetSelect.selectedIndex = nextIndex;
        targetSelect.dispatchEvent(new Event("change"));
        return true;
      }}

      function shouldIgnoreKeyboardShortcut(target) {{
        if (!(target instanceof Element)) {{
          return false;
        }}
        if (target instanceof HTMLSelectElement || target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement) {{
          return true;
        }}
        return Boolean(target.closest("[contenteditable='true']")) || target.isContentEditable;
      }}

      function syncView() {{
        const currentType = typeSelect ? typeSelect.value : "all";
        const allowedRecords = syncImageOptions(currentType);
        if (allowedRecords.length) {{
          showRecord(select.value);
          return;
        }}
        showRecord("__no_record__");
      }}

      if (typeSelect) {{
        typeSelect.addEventListener("change", syncView);
      }}
      if (select) {{
        select.addEventListener("change", () => showRecord(select.value));
      }}
      window.addEventListener("keydown", (event) => {{
        if (shouldIgnoreKeyboardShortcut(event.target)) {{
          return;
        }}
        if (event.key === "ArrowUp") {{
          if (moveSelection(select, -1)) {{
            event.preventDefault();
          }}
          return;
        }}
        if (event.key === "ArrowDown") {{
          if (moveSelection(select, 1)) {{
            event.preventDefault();
          }}
          return;
        }}
        if (event.key === "ArrowLeft") {{
          if (moveSelection(select, -1)) {{
            event.preventDefault();
          }}
          return;
        }}
        if (event.key === "ArrowRight") {{
          if (moveSelection(select, 1)) {{
            event.preventDefault();
          }}
        }}
      }});
      syncView();
    }})();

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
          const renderId = "dashboard-mermaid-" + Math.random().toString(36).slice(2);
          const rendered = await window.mermaid.render(renderId, mermaidCode);
          container.innerHTML = rendered.svg;
          if (rendered.bindFunctions) {{
            rendered.bindFunctions(container);
          }}
        }} catch (error) {{
          container.innerHTML = '<pre>Mermaid 渲染失败\\n' + String(error) + '</pre>';
        }}
      }}
    }})();
  </script>
</body>
</html>
"""


def _build_record_section(record: dict[str, Any]) -> str:
    original = record.get("original_image")
    original_card = _build_original_image_card(original)
    panel_cards = "\n".join(
        _build_panel_card(panel) for panel in record.get("panels", [])
    )
    return f"""
    <section class="record" data-image-id="{escape(record["image_id"])}" data-record-type="{escape(str(record.get("record_type", "unknown")))}">
      <div class="grid">
        {original_card}
        {panel_cards}
      </div>
    </section>
"""


def _build_original_image_card(snapshot: dict[str, Any] | None) -> str:
    if not snapshot:
        return ""
    return f"""
      <article class="card">
        <div class="card-head">
          <h2>{escape(snapshot.get("title", "Original"))}</h2>
          <span class="badge">image</span>
        </div>
        <div class="card-body">
          <div class="diagram-wrap">
            <div class="image-frame">
              <img src="{snapshot.get("data_url", "")}" alt="{escape(snapshot.get("title", "Original"))}" />
            </div>
          </div>
          <div class="card-meta">
            <div class="meta">
              <span>文件：{escape(str(snapshot.get("source_path", "")))}</span>
              <span>说明：{escape(str(snapshot.get("note", "") or "原始输入图像"))}</span>
            </div>
          </div>
        </div>
      </article>
"""


def _build_panel_card(panel: ComparePanel) -> str:
    content_html = _build_panel_content(panel)
    meta_html = _build_panel_meta(panel)
    return f"""
      <article class="card">
        <div class="card-head">
          <h2>{escape(panel.title)}</h2>
          <span class="badge">{escape(panel.image_type or "unknown")}</span>
        </div>
        <div class="card-body">
          {content_html}
          {meta_html}
        </div>
      </article>
"""


def _build_panel_meta(panel: ComparePanel) -> str:
    lines = [
        f"文件：{escape(panel.source_path)}" if panel.source_path else "",
        f"Caption：{escape(panel.caption)}" if panel.caption else "",
        f"说明：{escape(panel.note)}" if panel.note else "",
    ]
    visible_lines = [line for line in lines if line]
    if not visible_lines:
        return ""
    items_html = "".join(f"<span>{line}</span>" for line in visible_lines)
    return f'<div class="card-meta"><div class="meta">{items_html}</div></div>'


def _build_panel_content(panel: ComparePanel) -> str:
    if panel.render_kind == "mermaid":
        render_b64 = base64.b64encode(panel.render_text.encode("utf-8")).decode("ascii")
        return f"""
        <div class="diagram-wrap">
          <div class="diagram-frame" data-mermaid-b64="{render_b64}"></div>
        </div>
        <div class="code-wrap">
          <h3>Mermaid / Raw Text</h3>
          <pre>{escape(panel.render_text)}</pre>
        </div>
"""
    if panel.render_kind == "html":
        rendered_html = _sanitize_table_html(panel.render_text)
        return f"""
        <div class="code-wrap">
          <h3>Rendered HTML</h3>
          <div class="html-body"><div class="html-table-wrap">{rendered_html}</div></div>
        </div>
        <div class="code-wrap">
          <h3>HTML Source</h3>
          <pre>{escape(panel.render_text or "(empty)")}</pre>
        </div>
"""
    if panel.render_kind == "markdown":
        rendered_html = _render_markdown_content(panel.render_text)
        return f"""
        <div class="code-wrap">
          <h3>Rendered Markdown</h3>
          <div class="markdown-body">{rendered_html}</div>
        </div>
        <div class="code-wrap">
          <h3>Markdown Source</h3>
          <pre>{escape(panel.render_text or "(empty)")}</pre>
        </div>
"""
    if panel.render_kind == "text":
        return f"""
        <div class="code-wrap">
          <h3>Text</h3>
          <pre>{escape(panel.render_text or "(empty)")}</pre>
        </div>
"""
    return """
        <div class="code-wrap">
          <h3>Content</h3>
          <pre>(empty)</pre>
        </div>
"""


def _build_panel_from_normalized_payload(
    payload: Any,
    artifact_payload: Any,
    snapshot_lookup: dict[str, Any],
    title: str,
    source_path: str,
) -> ComparePanel:
    blocks = _safe_blocks_from_document(
        (payload or {}).get("document") if isinstance(payload, dict) else None
    )
    label_payload = (
        (payload or {}).get("derived_label") if isinstance(payload, dict) else None
    )
    note_suffix = ""
    fallback_key = {"MinerU": "mineru_block", "Qwen": "qwen_block"}.get(title)
    if not blocks and fallback_key and isinstance(artifact_payload, dict):
        blocks = _extract_issue_fallback_blocks(
            artifact_payload, fallback_key=fallback_key
        )
        if blocks:
            note_suffix = f"从 artifact.issue.{fallback_key} 回退"
    return _build_panel(
        title=title,
        source_path=source_path,
        blocks=blocks,
        label_payload=label_payload,
        mermaid_snapshot=snapshot_lookup.get(title),
        extra_note=note_suffix,
    )


def _build_qwen_panel(
    record_type: str,
    payload: Any,
    artifact_payload: Any,
    snapshot_lookup: dict[str, Any],
    source_path: str,
    artifact_source_path: str,
) -> ComparePanel | None:
    if str(record_type or "").strip().lower() == "flowchart":
        return _build_panel_from_normalized_payload(
            payload=payload,
            artifact_payload=artifact_payload,
            snapshot_lookup=snapshot_lookup,
            title="Qwen",
            source_path=source_path,
        )

    adjudication_text = _extract_qwen_adjudication_text(artifact_payload)
    if not adjudication_text:
        return None

    blocks = _safe_blocks_from_document(
        (payload or {}).get("document") if isinstance(payload, dict) else None
    )
    artifact_label = (
        artifact_payload.get("final_label")
        if isinstance(artifact_payload, dict)
        else None
    )
    label_payload = (
        artifact_label
        if isinstance(artifact_label, dict)
        else (payload or {}).get("derived_label") if isinstance(payload, dict) else None
    )
    image_type = _infer_image_type(
        label_payload=label_payload,
        blocks=blocks,
        prefer_label_semantics=True,
    )
    caption = _infer_caption(label_payload=label_payload, blocks=blocks)
    return ComparePanel(
        title="Qwen",
        source_path=artifact_source_path,
        image_type=image_type or record_type or "unknown",
        caption=caption,
        render_kind="text",
        render_text=adjudication_text,
        note="展示裁决结果与原因",
    )


def _build_flowchart_judge_panel(
    artifact_payload: Any,
    artifact_source_path: str,
) -> ComparePanel | None:
    render_text = _extract_qwen_adjudication_text(artifact_payload)
    if not render_text:
        return None
    return ComparePanel(
        title="Judge Reason",
        source_path=artifact_source_path,
        image_type="flowchart",
        caption="",
        render_kind="text",
        render_text=render_text,
        note="二阶段裁决原因",
    )


def _build_panel_from_final_payload(
    final_payload: Any,
    artifact_payload: Any,
    snapshot_lookup: dict[str, Any],
    title: str,
    source_path: str,
) -> ComparePanel:
    if isinstance(artifact_payload, dict):
        blocks = _safe_blocks_from_document(artifact_payload.get("final_document"))
        label_payload = artifact_payload.get("final_label")
        if blocks or isinstance(label_payload, dict):
            return _build_panel(
                title=title,
                source_path=f"{source_path} (artifact fallback)",
                blocks=blocks,
                label_payload=label_payload,
                mermaid_snapshot=snapshot_lookup.get(title),
                prefer_label_semantics=True,
            )

    blocks: list[dict[str, Any]] = []
    if isinstance(final_payload, dict):
        blocks = _extract_blocks_from_final_payload(final_payload)

    return _build_panel(
        title=title,
        source_path=source_path,
        blocks=blocks,
        label_payload=None,
        mermaid_snapshot=snapshot_lookup.get(title),
    )


def _build_panel(
    title: str,
    source_path: str,
    blocks: list[dict[str, Any]],
    label_payload: Any,
    mermaid_snapshot: Any,
    extra_note: str = "",
    prefer_label_semantics: bool = False,
) -> ComparePanel:
    image_type = _infer_image_type(
        label_payload=label_payload,
        blocks=blocks,
        prefer_label_semantics=prefer_label_semantics,
    )
    caption = _infer_caption(label_payload=label_payload, blocks=blocks)
    flowchart_signal = _has_flowchart_render_signal(
        blocks=blocks,
        label_payload=label_payload,
        prefer_label_semantics=prefer_label_semantics,
    )
    mermaid_text = ""

    if flowchart_signal and mermaid_snapshot is not None and str(
        getattr(mermaid_snapshot, "status", "") or ""
    ) in {"valid", "derived"}:
        mermaid_text = str(getattr(mermaid_snapshot, "render_code", "") or "")
        return ComparePanel(
            title=title,
            source_path=source_path,
            image_type=image_type or "flowchart",
            caption=caption,
            render_kind="mermaid",
            render_text=mermaid_text,
            note="；".join(
                item
                for item in [
                    str(getattr(mermaid_snapshot, "note", "") or ""),
                    extra_note,
                ]
                if item
            ),
        )

    if flowchart_signal:
        mermaid_text = _extract_mermaid_text(
            blocks=blocks,
            label_payload=label_payload,
            prefer_label_semantics=prefer_label_semantics,
        )
        if mermaid_text:
            return ComparePanel(
                title=title,
                source_path=source_path,
                image_type=image_type or "flowchart",
                caption=caption,
                render_kind="mermaid",
                render_text=mermaid_text,
                note="；".join(
                    item
                    for item in ["直接从结构化内容提取 Mermaid", extra_note]
                    if item
                ),
            )

    table_render = _extract_table_render_payload(
        blocks=blocks,
        label_payload=label_payload,
        prefer_label_semantics=prefer_label_semantics,
    )
    if table_render is not None:
        render_kind, render_text = table_render
        return ComparePanel(
            title=title,
            source_path=source_path,
            image_type=image_type or "table",
            caption=caption,
            render_kind=render_kind,
            render_text=render_text,
            note="；".join(
                item
                for item in [
                    "表格内容以 Markdown 展示"
                    if render_kind == "markdown"
                    else "表格内容以 HTML 展示",
                    extra_note,
                ]
                if item
            ),
        )

    text_content = _extract_textual_content(
        blocks=blocks,
        label_payload=label_payload,
        prefer_label_semantics=prefer_label_semantics,
    )
    return ComparePanel(
        title=title,
        source_path=source_path,
        image_type=image_type or "unknown",
        caption=caption,
        render_kind="text" if text_content else "empty",
        render_text=text_content,
        note="；".join(item for item in ["标签与文本摘要", extra_note] if item),
    )


def _safe_blocks_from_document(document_payload: Any) -> list[dict[str, Any]]:
    if not isinstance(document_payload, dict):
        return []
    raw_blocks = document_payload.get("blocks")
    if not isinstance(raw_blocks, list):
        return []
    return [item for item in raw_blocks if isinstance(item, dict)]


def _label_image_type(label_payload: Any) -> str:
    if not isinstance(label_payload, dict):
        return ""
    return str(label_payload.get("image_type", "") or "").strip()


def _infer_image_type(
    label_payload: Any,
    blocks: list[dict[str, Any]],
    prefer_label_semantics: bool = False,
) -> str:
    label_image_type = _label_image_type(label_payload)
    if prefer_label_semantics and label_image_type:
        return label_image_type
    inferred_from_blocks = _infer_record_type_from_blocks(blocks)
    if inferred_from_blocks != "unknown":
        return inferred_from_blocks
    if label_image_type:
        return label_image_type
    return "unknown"


def _infer_record_type(
    artifact_payload: Any,
    final_payload: Any,
    qwen_payload: Any,
    mineru_payload: Any,
    paddle_payload: Any,
    glm_payload: Any,
) -> str:
    if isinstance(artifact_payload, dict):
        artifact_label_type = _label_image_type(artifact_payload.get("final_label"))
        if artifact_label_type:
            return artifact_label_type

    candidate_blocks: list[list[dict[str, Any]]] = []
    if isinstance(artifact_payload, dict):
        candidate_blocks.append(
            _safe_blocks_from_document(artifact_payload.get("final_document"))
        )
    if isinstance(final_payload, dict):
        candidate_blocks.append(_extract_blocks_from_final_payload(final_payload))

    candidate_documents = []
    if isinstance(glm_payload, dict):
        candidate_documents.append(glm_payload.get("document"))
    if isinstance(paddle_payload, dict):
        candidate_documents.append(paddle_payload.get("document"))
    if isinstance(qwen_payload, dict):
        candidate_documents.append(qwen_payload.get("document"))
    if isinstance(mineru_payload, dict):
        candidate_documents.append(mineru_payload.get("document"))

    for blocks in candidate_blocks:
        inferred = _infer_record_type_from_blocks(blocks)
        if inferred != "unknown":
            return inferred

    for document_payload in candidate_documents:
        blocks = _safe_blocks_from_document(document_payload)
        inferred = _infer_record_type_from_blocks(blocks)
        if inferred != "unknown":
            return inferred
    return "unknown"


def _infer_record_type_from_blocks(blocks: list[dict[str, Any]]) -> str:
    for block in blocks:
        sub_type = str(block.get("sub_type", "") or "").strip().lower()
        if sub_type:
            return sub_type

    for block in blocks:
        block_type = str(block.get("type", "") or "").strip().lower()
        if block_type:
            return block_type
    return "unknown"


def _extract_blocks_from_final_payload(final_payload: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if not isinstance(final_payload, dict):
        return blocks
    parsed = final_payload.get("parsed")
    extraction_results = (
        parsed.get("extraction_results") if isinstance(parsed, dict) else None
    )
    if not isinstance(extraction_results, list):
        return blocks
    for page in extraction_results:
        if not isinstance(page, dict):
            continue
        json_res = page.get("json_res")
        if isinstance(json_res, list):
            blocks.extend(item for item in json_res if isinstance(item, dict))
    return blocks


def _infer_caption(label_payload: Any, blocks: list[dict[str, Any]]) -> str:
    if isinstance(label_payload, dict):
        caption = str(label_payload.get("caption", "") or "").strip()
        if caption:
            return caption
    for block in blocks:
        caption = _extract_caption_from_block(block)
        if caption:
            return caption
    return ""


def _extract_table_render_payload(
    blocks: list[dict[str, Any]],
    label_payload: Any,
    prefer_label_semantics: bool = False,
) -> tuple[str, str] | None:
    markdown_table = _extract_table_markdown(
        blocks=blocks,
        label_payload=label_payload,
        prefer_label_semantics=prefer_label_semantics,
    )
    if markdown_table:
        return "markdown", markdown_table

    html_table = _extract_table_html(
        blocks=blocks,
        label_payload=label_payload,
        prefer_label_semantics=prefer_label_semantics,
    )
    if html_table:
        return "html", html_table
    return None


def _extract_table_html(
    blocks: list[dict[str, Any]],
    label_payload: Any,
    prefer_label_semantics: bool = False,
) -> str:
    label_image_type = _label_image_type(label_payload).strip().lower()
    if isinstance(label_payload, dict):
        structured = label_payload.get("structured_label")
        if isinstance(structured, dict):
            structured_format = str(
                structured.get("format", "") or ""
            ).strip().lower()
            content = str(structured.get("content", "") or "").strip()
            if content and (
                structured_format == "html"
                or _looks_like_html_table(content)
                or (label_image_type == "table" and _looks_like_html_table(content))
            ):
                return content
        if prefer_label_semantics and label_image_type not in {"", "table"}:
            return ""

    for block in blocks:
        structured = block.get("structured_label")
        structured_content = ""
        structured_format = ""
        if isinstance(structured, dict):
            structured_content = str(structured.get("content", "") or "").strip()
            structured_format = str(structured.get("format", "") or "").strip().lower()

        content_payload = block.get("content")
        table_body = ""
        html_content = ""
        if isinstance(content_payload, dict):
            table_body = str(content_payload.get("table_body", "") or "").strip()
            html_content = str(content_payload.get("content", "") or "").strip()

        for candidate in (table_body, structured_content, html_content):
            if candidate and (
                structured_format == "html"
                or _looks_like_html_table(candidate)
                or (label_image_type == "table" and _looks_like_html_table(candidate))
            ):
                return candidate
    return ""


def _extract_table_markdown(
    blocks: list[dict[str, Any]],
    label_payload: Any,
    prefer_label_semantics: bool = False,
) -> str:
    label_image_type = _label_image_type(label_payload).strip().lower()
    if isinstance(label_payload, dict):
        structured = label_payload.get("structured_label")
        if isinstance(structured, dict):
            structured_kind = str(structured.get("kind", "") or "").strip().lower()
            structured_format = str(
                structured.get("format", "") or ""
            ).strip().lower()
            content = str(structured.get("content", "") or "").strip()
            if content and (
                structured_format == "markdown"
                or _looks_like_markdown_table(content)
            ):
                return content
        if prefer_label_semantics and label_image_type not in {"", "table"}:
            return ""
    for block in blocks:
        block_type = str(block.get("type", "") or "").strip().lower()
        block_sub_type = str(block.get("sub_type", "") or "").strip().lower()
        structured = block.get("structured_label")
        structured_content = ""
        structured_kind = ""
        structured_format = ""
        if isinstance(structured, dict):
            structured_content = str(structured.get("content", "") or "").strip()
            structured_kind = str(structured.get("kind", "") or "").strip().lower()
            structured_format = str(structured.get("format", "") or "").strip().lower()
        content_payload = block.get("content")
        table_body = ""
        if isinstance(content_payload, dict):
            table_body = str(content_payload.get("table_body", "") or "").strip()

        if table_body and (
            structured_format == "markdown"
            or _looks_like_markdown_table(table_body)
        ):
            return table_body
        if structured_content and (
            structured_format == "markdown"
            or _looks_like_markdown_table(structured_content)
        ):
            return structured_content

        block_text = str(block.get("text", "") or "").strip()
        if block_text and (
            block_type == "table"
            or block_sub_type == "table"
            or label_image_type == "table"
        ) and _looks_like_markdown_table(block_text):
            return block_text
    return ""


def _extract_qwen_adjudication_text(artifact_payload: Any) -> str:
    if not isinstance(artifact_payload, dict):
        return ""

    lines: list[str] = []
    seen_lines: set[str] = set()

    def add_line(text: str) -> None:
        line = str(text or "").strip()
        if not line:
            return
        normalized = " ".join(line.split()).lower()
        if normalized in seen_lines:
            return
        seen_lines.add(normalized)
        lines.append(line)

    consensus = artifact_payload.get("consensus")
    if isinstance(consensus, dict):
        decision = str(consensus.get("decision", "") or "").strip()
        if decision:
            add_line(f"裁决结果：{decision}")
        reasons = consensus.get("reasons")
        if isinstance(reasons, list):
            reason_texts = [str(item).strip() for item in reasons if str(item).strip()]
            if reason_texts:
                add_line("裁决原因：")
                for item in reason_texts:
                    add_line(f"- {item}")

    top_level_reasons = artifact_payload.get("reasons")
    if isinstance(top_level_reasons, list):
        reason_texts = [
            str(item).strip() for item in top_level_reasons if str(item).strip()
        ]
        if reason_texts:
            if not any(line == "裁决原因：" for line in lines):
                add_line("裁决原因：")
            for item in reason_texts:
                add_line(f"- {item}")

    seal_selection = artifact_payload.get("seal_selection")
    if isinstance(seal_selection, dict):
        selected_candidate = str(
            seal_selection.get("selected_candidate", "") or ""
        ).strip()
        if selected_candidate:
            add_line(f"印章候选：{selected_candidate}")
        selection_reason = str(seal_selection.get("reason", "") or "").strip()
        if selection_reason:
            add_line(f"选择原因：{selection_reason}")

    patch_decisions = artifact_payload.get("patch_decisions")
    if isinstance(patch_decisions, list):
        entries: list[str] = []
        for item in patch_decisions:
            if not isinstance(item, dict):
                continue
            issue_id = str(item.get("issue_id", "") or "").strip()
            decision = str(item.get("decision", "") or "").strip()
            reason = str(item.get("reason", "") or "").strip()
            if not decision and not reason:
                continue
            detail = decision or "unknown"
            if issue_id:
                detail = f"{issue_id}: {detail}"
            if reason:
                detail = f"{detail} ({reason})"
            entries.append(detail)
        if entries:
            add_line("补丁决策：")
            for entry in entries:
                add_line(f"- {entry}")

    return "\n".join(lines)


def _finalize_panels(record_type: str, panels: list[ComparePanel]) -> list[ComparePanel]:
    filtered = [
        panel
        for panel in panels
        if panel is not None and panel.render_kind != "empty"
    ]
    if str(record_type or "").strip().lower() != "flowchart":
        return filtered

    priority = {
        "Qwen": 0,
        "MinerU": 1,
        "Final": 2,
        "Judge Reason": 3,
    }
    return sorted(
        filtered,
        key=lambda panel: (priority.get(panel.title, 99), panel.title),
    )


def _render_markdown_content(markdown_text: str) -> str:
    text = str(markdown_text or "").strip()
    if not text:
        return "<p>(empty)</p>"

    lines = [line.rstrip() for line in text.splitlines()]
    fragments: list[str] = []
    index = 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        if _is_markdown_table_start(lines, index):
            end_index = index + 2
            while end_index < len(lines):
                row = _split_markdown_table_row(lines[end_index])
                if not lines[end_index].strip() or row is None:
                    break
                end_index += 1
            fragments.append(_render_markdown_table(lines[index:end_index]))
            index = end_index
            continue

        paragraph_lines: list[str] = []
        while index < len(lines):
            if not lines[index].strip():
                break
            if _is_markdown_table_start(lines, index):
                break
            paragraph_lines.append(lines[index].strip())
            index += 1
        if paragraph_lines:
            paragraph_html = "<br />".join(
                escape(line) for line in paragraph_lines if line
            )
            fragments.append(f"<p>{paragraph_html}</p>")
        else:
            index += 1
    return "".join(fragments) or "<p>(empty)</p>"


def _looks_like_markdown_table(text: str) -> bool:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    return _is_markdown_table_start(lines, 0)


def _looks_like_html_table(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    return "<table" in lowered and (
        "<tr" in lowered
        or "<td" in lowered
        or "<th" in lowered
        or "<thead" in lowered
        or "<tbody" in lowered
    )


def _sanitize_table_html(html_text: str) -> str:
    parser = _SafeTableHTMLParser()
    parser.feed(str(html_text or ""))
    parser.close()
    return parser.rendered_html()


class _SafeTableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._parts: list[str] = []
        self._open_tags: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._emit_start(tag, attrs, closed=False)

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self._emit_start(tag, attrs, closed=True)

    def handle_endtag(self, tag: str) -> None:
        lowered = str(tag or "").strip().lower()
        if lowered in _SKIPPED_TABLE_HTML_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if self._skip_depth or lowered not in _ALLOWED_TABLE_HTML_TAGS:
            return
        if self._open_tags and self._open_tags[-1] == lowered:
            self._open_tags.pop()
            self._parts.append(f"</{lowered}>")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self._parts.append(escape(data))

    def handle_entityref(self, name: str) -> None:
        if self._skip_depth:
            return
        self._parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self._skip_depth:
            return
        self._parts.append(f"&#{name};")

    def rendered_html(self) -> str:
        while self._open_tags:
            self._parts.append(f"</{self._open_tags.pop()}>")
        return "".join(self._parts)

    def _emit_start(
        self, tag: str, attrs: list[tuple[str, str | None]], closed: bool
    ) -> None:
        lowered = str(tag or "").strip().lower()
        if lowered in _SKIPPED_TABLE_HTML_TAGS:
            if not closed:
                self._skip_depth += 1
            return
        if self._skip_depth or lowered not in _ALLOWED_TABLE_HTML_TAGS:
            return
        attr_text = _format_safe_html_attrs(lowered, attrs)
        if closed or lowered in {"br", "col"}:
            self._parts.append(f"<{lowered}{attr_text} />")
            return
        self._parts.append(f"<{lowered}{attr_text}>")
        self._open_tags.append(lowered)


def _format_safe_html_attrs(
    tag: str, attrs: list[tuple[str, str | None]]
) -> str:
    allowed = _ALLOWED_TABLE_HTML_ATTRS.get(tag, set())
    parts: list[str] = []
    for name, value in attrs:
        attr_name = str(name or "").strip().lower()
        if attr_name not in allowed:
            continue
        normalized_value = str(value or "").strip()
        if not normalized_value:
            continue
        parts.append(f' {attr_name}="{escape(normalized_value, quote=True)}"')
    return "".join(parts)


def _is_markdown_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    header_cells = _split_markdown_table_row(lines[index])
    if header_cells is None:
        return False
    return _is_markdown_table_separator(
        lines[index + 1], expected_columns=len(header_cells)
    )


def _split_markdown_table_row(line: str) -> list[str] | None:
    stripped = str(line or "").strip()
    if "|" not in stripped:
        return None
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = [cell.strip() for cell in stripped.split("|")]
    if not cells or all(not cell for cell in cells):
        return None
    return cells


def _is_markdown_table_separator(line: str, expected_columns: int) -> bool:
    cells = _split_markdown_table_row(line)
    if cells is None or len(cells) != expected_columns:
        return False
    separator_pattern = re.compile(r"^:?-{1,}:?$")
    return all(separator_pattern.fullmatch(cell) for cell in cells)


def _render_markdown_table(lines: list[str]) -> str:
    if len(lines) < 2:
        return f"<p>{escape(chr(10).join(lines))}</p>"
    header_cells = _split_markdown_table_row(lines[0]) or []
    body_rows = [
        _split_markdown_table_row(line) or []
        for line in lines[2:]
        if str(line).strip()
    ]
    column_count = len(header_cells)

    thead_html = "".join(f"<th>{escape(cell)}</th>" for cell in header_cells)
    tbody_html_rows: list[str] = []
    for row in body_rows:
        normalized_row = row[:column_count] + [""] * max(0, column_count - len(row))
        cells_html = "".join(f"<td>{escape(cell)}</td>" for cell in normalized_row)
        tbody_html_rows.append(f"<tr>{cells_html}</tr>")
    tbody_html = "".join(tbody_html_rows)
    return (
        '<div class="markdown-table-wrap"><table class="markdown-table">'
        f"<thead><tr>{thead_html}</tr></thead>"
        f"<tbody>{tbody_html}</tbody>"
        "</table></div>"
    )


def _extract_mermaid_text(
    blocks: list[dict[str, Any]],
    label_payload: Any,
    prefer_label_semantics: bool = False,
) -> str:
    if isinstance(label_payload, dict):
        structured = label_payload.get("structured_label")
        if (
            isinstance(structured, dict)
            and str(structured.get("kind", "") or "").strip() == "mermaid"
        ):
            content = normalize_mermaid_text(
                str(structured.get("content", "") or "").strip()
            )
            if looks_like_mermaid(content):
                return content
        if prefer_label_semantics:
            return ""
    for block in blocks:
        sub_type = str(block.get("sub_type", "") or "").strip().lower()
        structured = block.get("structured_label")
        flowchart_graph = block.get("flowchart_graph")
        candidates: list[str] = []
        if isinstance(structured, dict):
            content = normalize_mermaid_text(
                str(structured.get("content", "") or "").strip()
            )
            if looks_like_mermaid(content):
                return content
        content_payload = block.get("content")
        if isinstance(content_payload, dict):
            for key in ("content", "text"):
                value = str(content_payload.get(key, "") or "").strip()
                if value:
                    candidates.append(value)
        text = str(block.get("text", "") or "").strip()
        if text:
            candidates.append(text)
        for candidate in candidates:
            normalized = normalize_mermaid_text(candidate)
            if looks_like_mermaid(normalized):
                return normalized
        if sub_type == "flowchart" and isinstance(flowchart_graph, dict):
            derived_mermaid = mermaid_from_flowchart_graph(flowchart_graph)
            normalized = normalize_mermaid_text(derived_mermaid)
            if looks_like_mermaid(normalized):
                return normalized
    return ""


def _has_flowchart_render_signal(
    blocks: list[dict[str, Any]],
    label_payload: Any,
    prefer_label_semantics: bool = False,
) -> bool:
    if isinstance(label_payload, dict):
        image_type = str(label_payload.get("image_type", "") or "").strip().lower()
        if image_type == "flowchart":
            return True
        structured = label_payload.get("structured_label")
        if (
            isinstance(structured, dict)
            and str(structured.get("kind", "") or "").strip().lower() == "mermaid"
        ):
            return True
        if isinstance(label_payload.get("flowchart_graph"), dict):
            return True
        if prefer_label_semantics:
            return False

    for block in blocks:
        if str(block.get("sub_type", "") or "").strip().lower() == "flowchart":
            return True
        structured = block.get("structured_label")
        if (
            isinstance(structured, dict)
            and str(structured.get("kind", "") or "").strip().lower() == "mermaid"
        ):
            return True
        if isinstance(block.get("flowchart_graph"), dict):
            return True
    return False


def _extract_textual_content(
    blocks: list[dict[str, Any]],
    label_payload: Any,
    prefer_label_semantics: bool = False,
) -> str:
    texts: list[str] = []
    if isinstance(label_payload, dict) and not (prefer_label_semantics and blocks):
        caption = str(label_payload.get("caption", "") or "").strip()
        if caption:
            texts.append(caption)
        structured = label_payload.get("structured_label")
        if isinstance(structured, dict):
            content = str(structured.get("content", "") or "").strip()
            if content:
                texts.append(content)
    for block in blocks:
        text = str(block.get("text", "") or "").strip()
        if text:
            texts.append(text)
        content_payload = block.get("content")
        if isinstance(content_payload, dict):
            for key in ("image_caption", "chart_caption", "table_caption"):
                value = content_payload.get(key)
                if isinstance(value, list):
                    texts.extend(
                        str(item).strip() for item in value if str(item).strip()
                    )
            if isinstance(content_payload.get("content"), str):
                content_text = str(content_payload.get("content") or "").strip()
                if content_text:
                    texts.append(content_text)
            if isinstance(content_payload.get("table_body"), str):
                body_text = str(content_payload.get("table_body") or "").strip()
                if body_text:
                    texts.append(body_text)
    return "\n\n".join(_deduplicate_texts(texts))


def _extract_caption_from_block(block: dict[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, dict):
        for key in ("chart_caption", "image_caption", "table_caption"):
            values = content.get(key)
            if isinstance(values, list):
                text = " ".join(
                    str(item).strip() for item in values if str(item).strip()
                ).strip()
                if text:
                    return text
    return str(block.get("text", "") or "").strip()


def _extract_issue_fallback_blocks(
    artifact_payload: dict[str, Any], fallback_key: str
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for issue in artifact_payload.get("issues") or []:
        if not isinstance(issue, dict):
            continue
        block = issue.get(fallback_key)
        if isinstance(block, dict):
            blocks.append(block)
    return blocks


def _deduplicate_texts(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        normalized = "".join(text.split()).lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(text)
    return ordered


def _build_type_options(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    ordered_types: list[str] = []
    seen_types: set[str] = set()
    for record in records:
        record_type = str(record.get("record_type", "unknown") or "unknown").strip()
        if not record_type or record_type in seen_types:
            continue
        seen_types.add(record_type)
        ordered_types.append(record_type)

    options = [{"value": "all", "label": "全部类型"}]
    for record_type in ordered_types:
        options.append(
            {
                "value": record_type,
                "label": TYPE_LABELS.get(record_type, record_type),
            }
        )
    return options


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
