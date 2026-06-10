# Mermaid TD-F1 Evaluation

该目录提供一个最小可运行的 Mermaid flowchart 评测数据集与 TD-F1 评测实现。

## 数据来源

当前数据集来自 `outputs/final/figure2.json` 中的 Mermaid `content` 字段。

## 当前假设

当前版本中，`prediction` 和 `groundtruth` 暂时使用同一份 Mermaid，用于验证评测 pipeline 是否可运行。

## 构建数据集

```bash
python eval_dataset/mermaid_td_f1/build_dataset.py
```

## 运行评测

```bash
python eval_dataset/mermaid_td_f1/run_eval.py
```

该脚本除了评测 `dataset.json` 外，还会运行内置 demo tests，其中包含：

- `Test 6: virtual routing node normalization`
- `Test 7: chained virtual node normalization`

用于验证显式空节点归一化后不会产生额外扣分。

## 运行新增测试

如果需要单独运行 virtual routing node 相关自动化测试，可执行：

```bash
pytest -q tests/test_mermaid_td_f1.py
```

该测试文件当前覆盖：

- `test_root_ignores_virtual_node`
- `test_virtual_routing_nodes_are_contracted_without_penalty`
- `test_chained_virtual_nodes_are_contracted_iteratively`

## Python 调用

```python
from eval_dataset.mermaid_td_f1.evaluator import evaluate_flowchart

result = evaluate_flowchart(
    ground_truth_mermaid=gold_mermaid,
    predicted_mermaid=pred_mermaid,
)

print(result["structure_f1"])
print(result["semantic_f1"])
```

## 外部文件评测

```bash
python eval_dataset/mermaid_td_f1/evaluate_files.py \
    --pred path/to/pred.mmd \
    --gold path/to/gold.mmd \
    --output result.json
```

说明：

- `pred.mmd` 为预测 Mermaid 文件
- `gold.mmd` 为标准 Mermaid 文件
- `result.json` 为输出评测结果

用户可以直接替换任意两个 Mermaid 文件进行评测，无需修改代码。

## Qwen 同构流程图压力测试

该脚本不会使用固定 Mermaid。

它会让本地 Qwen 每次随机生成：

- `gold_mermaid`
- `pred_mermaid`

两张业务语义完全一致但 Mermaid 表达方式差异极大的流程图。

然后使用 TD-F1 评测器验证以下能力是否正确工作：

- sibling-order-invariant
- virtual node normalization
- merge handling
- loop handling
- binding handling

运行：

```bash
python eval_dataset/mermaid_td_f1/qwen_equivalence_stress_test.py \
  --num-tests 100 \
  --model qwen \
  --base-url http://localhost:8000/v1/chat/completions \
  --temperature 0.9 \
  --output-dir eval_dataset/mermaid_td_f1/qwen_debug_outputs
```

可选：

- `--api-key`：不传时默认读取 `LOCAL_QWEN_API_KEY`，并兼容回退到 `QWEN_LOCAL_API_KEY`
- `--timeout`：默认 `120`
- `--fail-threshold`：默认 `0.999999`
- `--self-check`：要求 Qwen 额外解释为什么两张图等价，便于人工区分是评测器问题还是模型生成了不等价图

如果任意一次得分低于阈值，或 Qwen 输出无法解析，脚本会立即停止并保存失败样本，包括：

- `failure_iter_{i}.json`
- `failure_iter_{i}_gold.mmd`
- `failure_iter_{i}_pred.mmd`
- `failure_iter_{i}_prompt.txt`
- `failure_iter_{i}_raw_output.txt`

该脚本的目标不是测试模型，而是测试评测器：

- Qwen 的职责：生成大量本质同构但表达差异极大的 Mermaid 图对
- TD-F1 的职责：判断这些图对是否应该得到 1 分

只要出现 “Qwen 明确生成的是等价图，但 TD-F1 < 1”，就应视为发现了评测器 bug，并使用保存下来的失败样本继续 debug。

## TD-F1 简述

TD-F1 是一个 top-down、sibling-order-invariant、soft-recursive-alignment 的 Mermaid flowchart 评测指标。
它不会因为局部结构错误阻断整棵子树，而是通过 precision / recall 自然扣分。

## Virtual Routing Node Normalization

评测前会消除 virtual routing nodes。空节点、布局节点、只用于汇聚或分发的中转节点不表达业务语义，因此不会影响最终 TD-F1 分数。系统会将 `P -> V -> Q` 形式的虚拟节点路径收缩为 `P -> Q`。
如果虚拟节点两侧边都带有非空文字标签，当前版本会保守地跳过收缩，以避免改变原始流程语义。

## 说明与限制

- 当前实现是轻量 Mermaid 解析器，只覆盖常见 flowchart 节点与边写法。
- 不调用外部 Mermaid CLI，不依赖网络。
- 复杂 `subgraph`、样式混排、多重高级语法暂不保证完整支持。
