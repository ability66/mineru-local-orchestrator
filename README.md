# mineru-local-orchestrator

面向本地部署服务的图表 / 流程图 / 印章识别编排仓库。

当前目标：

- 上游接入本地 `MinerUPro 2.5`
- 上游接入本地 `Paddle` 视觉服务
- 上游接入本地 `GLM` 视觉服务
- 上游接入本地 `qwen 122b` judge 服务
- 保留类似“投票”的思路，但在两路场景下改成 `MinerU 主锚点 + Qwen 仲裁/补充`
- 最终输出统一到 MinerU 风格
- 保留图表、流程图、印章识别相关能力

## 设计原则

- `MinerU` 负责页面结构、块级 bbox、阅读顺序、基础内容块切分
- `Paddle / GLM` 负责一阶段辅助视觉识别
- `Qwen` 只在发现分歧时作为二阶段 judge 介入
- 由于当前只有两路上游，不做伪“多数投票”，而是做字段级双源仲裁
- 最终交付以 MinerU 风格 JSON 为主，同时额外保存 debug 产物

## 当前输出约定

当前仓库默认写出：

- `outputs/raw/mineru/{image_id}.json`
- `outputs/raw/paddle/{image_id}.json`
- `outputs/raw/glm/{image_id}.json`
- `outputs/raw/qwen/{image_id}.json`
- `outputs/normalized/mineru/{image_id}.json`
- `outputs/normalized/paddle/{image_id}.json`
- `outputs/normalized/glm/{image_id}.json`
- `outputs/normalized/qwen/{image_id}.json`
- `outputs/final/{image_id}.json`
- `outputs/final/{image_id}_artifact.json`
- `outputs/summary.jsonl`

说明：

- `final/{image_id}.json` 是主产物，外层对齐 `tmp.json` 风格，核心结果在 `parsed.extraction_results[].json_res`
- `Paddle / GLM` 只作为一阶段辅助来源，不作为并列最终结果格式暴露
- `Qwen` 只作为分歧 judge，不再作为并列最终结果格式暴露
- `artifact.json` 保存双源仲裁、graph fusion、review 原因等 debug 信息

## 关于 MinerU 输出兼容

当前实现参考了 MinerU 官方输出文件说明中的 `content_list.json` 与 `content_list_v2.json` 字段命名和块类型约定，但你还没有提供本地 `MinerUPro 2.5` 的真实响应样例。

因此本仓库现在的状态是：

- 字段命名和块类型尽量对齐官方约定
- 接口层和 writer 已经预留可调位置
- 等你提供本地接口返回样例后，再把兼容性收敛到 1:1

官方参考：

- https://opendatalab.github.io/MinerU/zh/reference/output_files/

## 运行

初始化依赖：

```bash
uv sync --extra dev
```

查看帮助：

```bash
uv run python -m src.main --help
```

服务器更新环境：

```bash
git pull --rebase
uv sync
```

预处理整页图片并裁出视觉块：

离线读取已存在的 layout json：

```bash
uv run python -m src.preprocess.main \
  --data-dir data/pages \
  --layout-source json \
  --layout-dir data/layouts \
  --output-dir data/preprocess \
  --workers 8 \
  --overwrite
```

在线调用 `mineru_vl_utils` 的 layout client：

```bash
uv run python -m src.preprocess.main \
  --data-dir data/pages \
  --layout-source mineru_vl \
  --server-url http://80.11.138.9:30000 \
  --output-dir data/preprocess \
  --workers 8 \
  --overwrite
```

输出目录结构示例：

```text
data/preprocess/<page_stem>/
  layout.json
  <page_stem>_001_chart_bar_line.png
  manifest.json
```

生成可视化对比页：

```bash
uv run python -m src.render_compare_dashboard --output-dir outputs
```

在服务器上直接查看已生成页面：

```bash
uv run python -m src.serve_dashboard --root-dir outputs --host 0.0.0.0 --port 18743
```

访问地址示例：

```text
http://<server-ip>:18743/compare_dashboard/index.html
http://<server-ip>:18743/compare_mermaid/figure1.html
```

说明：

- `--port` 必填，不提供默认端口，避免误占用
- 默认 `--host 127.0.0.1` 仅本机可访问；部署到服务器对外查看时可显式传 `--host 0.0.0.0`
- 该服务只负责静态查看 `outputs` 下的已生成 HTML 与资源，不会触发识别流程

基础运行：

```bash
uv run python -m src.main \
  --data-dir data \
  --output-dir outputs \
  --models-config configs/models.local.yaml \
  --prompts-config configs/prompts.yaml
```

如果前面已经做了 preprocess，后续主流程可以直接吃裁好的图：

```bash
uv run python -m src.main \
  --data-dir data/preprocess \
  --output-dir outputs \
  --models-config configs/models.local.yaml \
  --prompts-config configs/prompts.yaml
```

## 待补信息

你后续需要提供：

- `MinerUPro 2.5` 本地端口、路径、请求示例、返回示例
- `qwen 122b` 本地端口、协议、模型名、是否视觉输入
- 一份你认可的 MinerU 标准输出样例
