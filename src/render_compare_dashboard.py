from __future__ import annotations

import argparse
import base64
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

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

    for provider in ("mineru", "qwen"):
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

    return {
        "image_id": image_id,
        "original_image": None
        if original_image is None
        else {
            "title": original_image.title,
            "source_path": original_image.source_path,
            "data_url": original_image.data_url,
            "note": original_image.note,
        },
        "panels": [
            _build_panel_from_normalized_payload(
                payload=mineru_payload,
                artifact_payload=artifact_payload,
                snapshot_lookup=snapshot_lookup,
                title="MinerU",
                source_path=f"normalized/mineru/{image_id}.json",
            ),
            _build_panel_from_normalized_payload(
                payload=qwen_payload,
                artifact_payload=artifact_payload,
                snapshot_lookup=snapshot_lookup,
                title="Qwen",
                source_path=f"normalized/qwen/{image_id}.json",
            ),
            _build_panel_from_final_payload(
                final_payload=final_payload,
                artifact_payload=artifact_payload,
                snapshot_lookup=snapshot_lookup,
                title="Final",
                source_path=f"final/{image_id}.json",
            ),
        ],
    }


def build_dashboard_html(
    records: list[dict[str, Any]], mermaid_script_path: str
) -> str:
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
      <p>原图、MinerU、Qwen 与 Final 的统一对比面板。流程图渲染 Mermaid，表格展示 Markdown，其它展示文字。</p>
      <div class="controls">
        <label for="image-select">切换图片</label>
        <select id="image-select">{options_html}</select>
      </div>
    </section>
    {sections_html}
  </div>
  <script src="{escape(mermaid_script_path)}"></script>
  <script>
    (function () {{
      const select = document.getElementById("image-select");
      const sections = Array.from(document.querySelectorAll(".record"));
      function showRecord(imageId) {{
        for (const section of sections) {{
          section.classList.toggle("active", section.getAttribute("data-image-id") === imageId);
        }}
      }}
      if (select) {{
        showRecord(select.value);
        select.addEventListener("change", () => showRecord(select.value));
      }}
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
    <section class="record" data-image-id="{escape(record["image_id"])}">
      <div class="grid">
        {original_card}
        {panel_cards}
      </div>
    </section>
"""


def _build_original_image_card(snapshot: dict[str, Any] | None) -> str:
    if not snapshot:
        return """
      <article class="card">
        <div class="card-head">
          <h2>Original</h2>
          <div class="meta">
            <span class="badge">missing</span>
            <span>未找到原始图像</span>
          </div>
        </div>
        <div class="diagram-wrap"><div class="image-frame"><pre>当前没有可展示的原始图像。</pre></div></div>
      </article>
"""
    return f"""
      <article class="card">
        <div class="card-head">
          <h2>{escape(snapshot.get("title", "Original"))}</h2>
          <div class="meta">
            <span class="badge">image</span>
            <span>文件：{escape(str(snapshot.get("source_path", "")))}</span>
            <span>说明：{escape(str(snapshot.get("note", "") or "原始输入图像"))}</span>
          </div>
        </div>
        <div class="diagram-wrap">
          <div class="image-frame">
            <img src="{snapshot.get("data_url", "")}" alt="{escape(snapshot.get("title", "Original"))}" />
          </div>
        </div>
      </article>
"""


def _build_panel_card(panel: ComparePanel) -> str:
    content_html = _build_panel_content(panel)
    return f"""
      <article class="card">
        <div class="card-head">
          <h2>{escape(panel.title)}</h2>
          <div class="meta">
            <span class="badge">{escape(panel.image_type or "unknown")}</span>
            <span>文件：{escape(panel.source_path)}</span>
            <span>Label：{escape(panel.image_type or "unknown")}</span>
            <span>Caption：{escape(panel.caption or "(empty)")}</span>
            <span>说明：{escape(panel.note or "无补充说明")}</span>
          </div>
        </div>
        {content_html}
      </article>
"""


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
    if panel.render_kind in {"markdown", "text"}:
        label = "Markdown" if panel.render_kind == "markdown" else "Text"
        return f"""
        <div class="code-wrap">
          <h3>{escape(label)}</h3>
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
    if not blocks and isinstance(artifact_payload, dict):
        fallback_key = "mineru_block" if title == "MinerU" else "qwen_block"
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
            )

    blocks: list[dict[str, Any]] = []
    if isinstance(final_payload, dict):
        parsed = final_payload.get("parsed")
        extraction_results = (
            parsed.get("extraction_results") if isinstance(parsed, dict) else None
        )
        if isinstance(extraction_results, list):
            for page in extraction_results:
                if not isinstance(page, dict):
                    continue
                json_res = page.get("json_res")
                if isinstance(json_res, list):
                    blocks.extend(item for item in json_res if isinstance(item, dict))

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
) -> ComparePanel:
    image_type = _infer_image_type(label_payload=label_payload, blocks=blocks)
    caption = _infer_caption(label_payload=label_payload, blocks=blocks)

    if mermaid_snapshot is not None and str(
        getattr(mermaid_snapshot, "status", "") or ""
    ) in {"valid", "derived"}:
        return ComparePanel(
            title=title,
            source_path=source_path,
            image_type=image_type or "flowchart",
            caption=caption,
            render_kind="mermaid",
            render_text=str(getattr(mermaid_snapshot, "render_code", "") or ""),
            note="；".join(
                item
                for item in [
                    str(getattr(mermaid_snapshot, "note", "") or ""),
                    extra_note,
                ]
                if item
            ),
        )

    table_markdown = _extract_table_markdown(blocks=blocks, label_payload=label_payload)
    if table_markdown:
        return ComparePanel(
            title=title,
            source_path=source_path,
            image_type=image_type or "table",
            caption=caption,
            render_kind="markdown",
            render_text=table_markdown,
            note="；".join(
                item for item in ["表格内容以 Markdown 展示", extra_note] if item
            ),
        )

    text_content = _extract_textual_content(blocks=blocks, label_payload=label_payload)
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


def _infer_image_type(label_payload: Any, blocks: list[dict[str, Any]]) -> str:
    if isinstance(label_payload, dict):
        image_type = str(label_payload.get("image_type", "") or "").strip()
        if image_type:
            return image_type
    for block in blocks:
        sub_type = str(block.get("sub_type", "") or "").strip().lower()
        block_type = str(block.get("type", "") or "").strip().lower()
        if sub_type == "flowchart":
            return "flowchart"
        if block_type == "table":
            return "table"
        if block_type == "chart":
            return "chart"
        if block_type == "image":
            return "natural_image"
    return "unknown"


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


def _extract_table_markdown(blocks: list[dict[str, Any]], label_payload: Any) -> str:
    if isinstance(label_payload, dict):
        structured = label_payload.get("structured_label")
        if (
            isinstance(structured, dict)
            and str(structured.get("kind", "") or "").strip() == "table"
        ):
            content = str(structured.get("content", "") or "").strip()
            if content:
                return content
    for block in blocks:
        if str(block.get("type", "") or "").strip().lower() != "table":
            continue
        structured = block.get("structured_label")
        if isinstance(structured, dict):
            content = str(structured.get("content", "") or "").strip()
            if content:
                return content
        content_payload = block.get("content")
        if isinstance(content_payload, dict):
            body = str(content_payload.get("table_body", "") or "").strip()
            if body:
                return body
    return ""


def _extract_textual_content(blocks: list[dict[str, Any]], label_payload: Any) -> str:
    texts: list[str] = []
    if isinstance(label_payload, dict):
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


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
