# mineru-local-orchestrator

面向本地部署服务的图表 / 流程图 / 印章识别编排仓库。

当前目标：

- 上游接入本地 `MinerUPro 2.5`
- 上游接入本地 `qwen 122b` 服务
- 保留类似“投票”的思路，但在两路场景下改成 `MinerU 主锚点 + Qwen 仲裁/补充`
- 最终输出统一到 MinerU 风格
- 保留图表、流程图、印章识别相关能力

## 设计原则

- `MinerU` 负责页面结构、块级 bbox、阅读顺序、基础内容块切分
- `Qwen` 负责图表/流程图/印章语义补充、文字修正、caption 补全、冲突仲裁
- 由于当前只有两路上游，不做伪“多数投票”，而是做字段级双源仲裁
- 最终交付以 MinerU 风格 JSON 为主，同时额外保存 debug 产物

## 当前输出约定

当前仓库默认写出：

- `outputs/raw/mineru/{image_id}.json`
- `outputs/raw/qwen/{image_id}.json`
- `outputs/normalized/mineru/{image_id}.json`
- `outputs/normalized/qwen/{image_id}.json`
- `outputs/final/{image_id}.json`
- `outputs/final/{image_id}_artifact.json`
- `outputs/summary.jsonl`

说明：

- `final/{image_id}.json` 是主产物，外层对齐 `tmp.json` 风格，核心结果在 `parsed.extraction_results[].json_res`
- `Qwen` 只作为补充和仲裁来源，不再作为并列最终结果格式暴露
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

基础运行：

```bash
uv run python -m src.main \
  --data-dir data \
  --output-dir outputs \
  --models-config configs/models.local.yaml \
  --prompts-config configs/prompts.yaml
```

## 待补信息

你后续需要提供：

- `MinerUPro 2.5` 本地端口、路径、请求示例、返回示例
- `qwen 122b` 本地端口、协议、模型名、是否视觉输入
- 一份你认可的 MinerU 标准输出样例
