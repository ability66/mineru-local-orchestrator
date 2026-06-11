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

## 评测接口

数值型图表 / chart-to-table 评测接口位于 `eval_dataset/chart_td_f1`。

直接比较两个表格字符串：

```python
from eval_dataset.chart_td_f1 import evaluate_chart_table

prediction = """| year | revenue | profit |
| --- | ---: | ---: |
| 2023 | 100 | 20 |
| 2024 | 120 | 30 |"""

ground_truth = """| metric | 2023 | 2024 |
| --- | ---: | ---: |
| revenue | 100 | 120 |
| profit | 20 | 30 |"""

result = evaluate_chart_table(
    prediction=prediction,
    ground_truth=ground_truth,
    tolerance="slight",
    allow_transpose=True,
)

print(result["triple_f1"])
print(result["map_slight"])
```

直接读取 `outputs/final/0.json` 的真实结构：

```python
import json
from pathlib import Path

from eval_dataset.chart_td_f1 import evaluate_from_record

record = json.loads(Path("outputs/final/0.json").read_text(encoding="utf-8"))

result = evaluate_from_record(
    record,
    chart_index=0,
    tolerance="slight",
)

print(result["prediction_field_path"])
print(result["groundtruth_field_path"])
print(result["triple_f1"])
```

当前 `evaluate_from_record()` 会从 record 中自动提取 `type == "chart"` 的 block，并读取其中的 `content` 作为输入。
