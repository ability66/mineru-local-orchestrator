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

## FlowVQA 用法

如果你已经把 `https://github.com/flowvqa/flowvqa` 完整克隆到本地，可以直接把主图片目录接进当前流程图识别工作流。

注意：

- `--data-dir` 请指向 `Data/A. Main Set Flowchart Images`
- 不要直接把整个 `Data` 目录传给 `--data-dir`
- `B. Directional Bias Bottom Top Set Images` 不是主识别集，且文件名 stem 会与主集冲突

推荐命令：

```bash
uv run python -m src.main \
  --data-dir "/path/to/flowvqa/Data/A. Main Set Flowchart Images" \
  --flowvqa-root "/path/to/flowvqa" \
  --output-dir outputs/flowvqa_main \
  --models-config configs/models.local.yaml \
  --prompts-config configs/prompts.yaml \
  --manual-compare-mode \
  --workers 8 \
  --limit 100
```

说明：

- `--flowvqa-root` 指向你本地克隆的 `flowvqa` 仓库根目录
- 设置后，系统会自动按图片文件名（例如 `code00294.png`）去 `train_full.json` / `test_full.json` 匹配对应的 gold Mermaid
- `--manual-compare-mode` 会在每张图完成后更新 compare 页面，适合边跑边看
- `--workers 8` 可用于提高吞吐；如果远端服务吃不消，再降到 `4`
- `--limit 100` 只是示例；想全量跑就去掉这个参数

运行后再开静态服务：

```bash
uv run python -m src.serve_dashboard \
  --root-dir outputs/flowvqa_main \
  --host 0.0.0.0 \
  --port 18743
```

访问：

```text
http://<server-ip>:18743/compare_dashboard/index.html
http://<server-ip>:18743/compare_mermaid/code00294.html
```

当 `--flowvqa-root` 生效后，前端页面会额外展示：

- `Ground Truth` 面板
- `MinerU Raw` 面板
- `Ours` 面板
- Ground Truth、MinerU Raw、Ours 三者的 Mermaid 源码和渲染结果
- 如果仓库中提供兼容的 FlowVQA 评测器，也会额外展示相对 Ground Truth 的评测指标

输出目录中与 FlowVQA 相关的关键文件：

- `outputs/flowvqa_main/final/<image_id>_artifact.json`
  - 包含 `final_document.raw_metadata.flowvqa_eval`
- `outputs/flowvqa_main/compare_dashboard/index.html`
  - 总览页，会在运行过程中持续刷新
- `outputs/flowvqa_main/compare_mermaid/<image_id>.html`
  - 单图 Mermaid 对比页，适合实时查看

如果你只想先抽样验证，可保留 `--limit`；如果想看某张图的细节，优先打开单图页，因为它比总览页更接近实时。
