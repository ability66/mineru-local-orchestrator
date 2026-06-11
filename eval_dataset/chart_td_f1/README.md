# Chart TD-F1 Evaluation

该目录提供一个最小可运行的数值型图表 / chart-to-table 评测模块。当前版本不再是“按思路重写”，而是直接复制并 vendoring 了 ChartArena 的核心实现到 `eval_dataset/chart_td_f1/chartarena_vendor/`：

- `metrics/SCRM.py`
- `metrics/mermaid_eval.py`
- `metrics/tree_eval.py`
- `methods/context.py`
- `methods/normalize.py`
- `methods/parsers/parse_json.py`
- `methods/parsers/parse_markdown.py`
- `methods/parsers/parse_pie.py`
- `methods/parsers/parse_python_code.py`
- `methods/parsers/parse_svg.py`

也就是说，底层三元组生成、tolerance matching 和 `mAP` 逻辑直接来自 ChartArena 原实现；当前目录主要补的是：

- 对 `outputs/final/0.json` 的真实字段路径适配
- 一个稳定的本仓库接口
- README
- 测试

`chart_td_f1` 现在不再运行时依赖 `eval_dataset/ChartArena`。删除那个目录后，只要 `.venv` 里的依赖还在，新模块仍可独立运行。

## 这个指标评什么

目标不是比较 Markdown 字符串长得像不像，而是比较模型是否把图表里的语义数据恢复出来。

例如下面两张表：

```markdown
| year | revenue | profit |
| --- | ---: | ---: |
| 2023 | 100 | 20 |
| 2024 | 120 | 30 |
```

```markdown
| metric | 2023 | 2024 |
| --- | ---: | ---: |
| revenue | 100 | 120 |
| profit | 20 | 30 |
```

虽然字符串不同、布局也转置了，但语义上都表示同一组数据。该评测器会先把它们转为 canonical triples，再比较 triple 集合。

## Canonical Triple

概念上的 triple 形式为：

```python
{
    "entity": "...",
    "header": "...",
    "value": "...",
}
```

内部比较时会把 `entity/header` 归一化，并对这两个 key 做无序对称化，以支持转置表匹配。

例如：

```python
("2023", "revenue", "100")
("2023", "profit", "20")
("2024", "revenue", "120")
("2024", "profit", "30")
```

## 为什么不直接比 Markdown 字符串

- 行顺序交换不应影响分数。
- 列顺序交换不应影响分数。
- 宽表和转置表语义相同，不应因为布局不同而记错。
- 数值轻微误差需要 tolerance，而不是纯字符串不等就 0 分。

## 支持的输入格式

当前实现支持：

- Markdown table
- 标准 CSV
- ChartArena 风格内部 CSV（`" \\t "` / `" \\n "` 分隔）
- JSON string / `dict` / `list`
  - `{"headers": [...], "rows": [...]}`
  - `{"values": {...}}`
  - `list[dict]`
  - `list[list]`

本仓库实际使用场景里，`outputs/final/0.json` 的图表内容是 Markdown table。

## 依赖

该模块运行时需要 `.venv` 中安装：

- `numpy`
- `python-Levenshtein`

建议命令：

```bash
python -m venv .venv
./.venv/bin/pip install -e '.[dev,chart_eval]'
```

如果你只想补齐最小依赖，也可以直接：

```bash
./.venv/bin/pip install numpy python-Levenshtein pytest
```

不需要保留 `eval_dataset/ChartArena` 目录。

## Tolerance 定义

- `strict`
  - ChartArena 原实现参数：`tol_word=0`, `tol_num=0.05`
- `slight`
  - ChartArena 原实现参数：`tol_word=1`, `tol_num=0.10`
- `high`
  - ChartArena 原实现参数：`tol_word=5`, `tol_num=0.20`

这里的 `tol_word` 和 `tol_num` 直接沿用 ChartArena `SCRM.csv_eval()` 的定义。

## 指标

- `parse_success`
  - pred / gt 是否都成功解析为表格
- `triple_precision`
  - 匹配到的 triples / 预测 triples
- `triple_recall`
  - 匹配到的 triples / GT triples
- `triple_f1`
  - triple 级 F1
- `triple_iou`
  - triple 集合 IoU
- `exact_match`
  - canonical triples 是否完全一致
- `avg_numeric_error`
  - 已匹配与未匹配 numeric triples 的平均误差

说明：

- `map_strict / map_slight / map_high` 直接来自 vendored ChartArena 原始 `csv_eval()`
- `triple_* / avg_numeric_error` 是本目录在外层补充的辅助指标，方便按 precision / recall / F1 / IoU 查看结果
- 两类指标共享同一套 ChartArena CSV 归一化与 tolerance 设定，但数值不要求完全相同

返回中还会包含：

- `matched`
- `pred_count`
- `gt_count`
- `pred_variant`
- `gt_variant`
- `map_strict`
- `map_slight`
- `map_high`
- `ap_50_* / ap_75_* / ap_90_*`

## Python 调用

```python
from eval_dataset.chart_td_f1 import evaluate_chart_table, evaluate_from_record

result = evaluate_chart_table(
    prediction=pred_table,
    ground_truth=gold_table,
    tolerance="slight",
    allow_transpose=True,
)

print(result["triple_f1"])
print(result["triple_iou"])
```

## 直接兼容 `outputs/final/0.json`

### 实际字段路径

当前仓库里的 `outputs/final/0.json` 真实图表字段是：

- `parsed.extraction_results[0].json_res[0].content`
- `parsed.extraction_results[0].json_res[2].content`

并且对应 block 满足：

- `parsed.extraction_results[0].json_res[0].type == "chart"`
- `parsed.extraction_results[0].json_res[2].type == "chart"`

这些 `content` 字段里存放的是 Markdown table。

### 注意

`outputs/final/0.json` 本身没有独立的 `gt` / `groundtruth` / `prediction` 字段，也没有同文件内的 gold table。因此当前模块里的 `build_dataset.py` 和默认 `evaluate_from_record()` 用的是 self-gold smoke test：

- `prediction = chart.content`
- `groundtruth = chart.content`

这是为了先验证评测 pipeline 与真实字段结构兼容，不代表正式 benchmark。

### `evaluate_from_record()`

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
```

如果你后续有另一份同结构 record 作为 gold，也可以传：

```python
result = evaluate_from_record(
    pred_record,
    ground_truth_record=gold_record,
    chart_index=0,
    ground_truth_chart_index=0,
    tolerance="slight",
)
```

## 构建 smoke dataset

```bash
python eval_dataset/chart_td_f1/build_dataset.py
```

它会从 `outputs/final/0.json` 读取所有 `type == "chart"` 的 block，并生成：

- `eval_dataset/chart_td_f1/dataset.json`

每个样例都会显式记录实际字段路径：

- `prediction_field_path`
- `groundtruth_field_path`

## 运行评测

```bash
python eval_dataset/chart_td_f1/run_eval.py
```

## 外部文件评测

```bash
python eval_dataset/chart_td_f1/evaluate_files.py \
  --pred path/to/pred_table.md \
  --gold path/to/gold_table.md \
  --output result.json
```

可选：

- `--tolerance strict|slight|high`
- `--disable-transpose`

## 运行测试

```bash
./.venv/bin/python -m pytest -q tests/test_chart_td_f1.py
```

测试覆盖：

- 完全一致
- 行顺序交换
- 列顺序交换
- 转置表
- strict / slight / high tolerance
- 绑定错误
- 缺失行 / 多余行
- 非法 Markdown / 空输出
- 百分号 / 千分位 / 美元符号归一化
- 中文表头 / 中文类别
- 基于 `outputs/final/0.json` 实际结构的 record 样例
- `evaluate_from_record()`

## 与 ChartArena 的关系

当前版本直接复用了本地 ChartArena clone 的核心实现，而不是再做等价重写：

- 归一化路径：ChartArena `methods/normalize.py`
- 三元组评分：ChartArena `metrics/SCRM.py`

本目录相对于原始 ChartArena 额外增加的是：

- `evaluate_from_record()`，直接兼容 `outputs/final/0.json`
- `parse_success / triple_precision / triple_recall / triple_f1 / triple_iou / avg_numeric_error`
  这些是包在 ChartArena `mAP` 之外的辅助指标
- smoke dataset 构建脚本
- 面向本仓库的 README 和测试

ChartArena 仓库 README 中的 license 说明是：

> This benchmark is released for research purposes only.

因此当前实现的定位是：直接承接 ChartArena 原始评测逻辑，再在外层补本仓库接口。
