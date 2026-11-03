# 实验记录：给继续整理和修改项目的 Agent

这里保存真实实验的参考证据，不需要模型、数据集或 GPU 即可阅读。它不是新的推理实验，也不表示历史效果已经恢复。

先读 [交接说明](AGENT_HANDOFF.md)，再读 [2026-11-03 完整实验检查](records/reports/latest_experiment_review_20261103/最新实验检查.md)。逐题范围、来源、哈希及压缩规则见 [manifest.json](records/manifest.json)。

## 包含哪些记录

| 范围 | 内容 |
|---|---|
| 最新数组 154750 / 154751，136 题 | 完整保存的 4B、9B 调用日志、节点报告及程序输入、中央候选视图和绑定、各轮判断、补查请求与工具返回、冻结答案和独立评价、checkpoint 统计；不含原始 inbox 数据库 |
| 上一完整数组 154057 / 154058，136 题 | 逐题冻结预测、最终回复、独立评价、公开问题及完成信息，供同题比较；没有打包这轮全部模型调用和中间过程 |
| 历史修复与回退 | 已有结果报告、差异审计、对照结果、修复 patch 和关键源码快照；不是所有历史运行的完整备份 |
| 前端复盘 | 第 1、20、128 题已导出的真实页面数据，以及 IG01 交接问题的三张截图；可以重新生成离线 HTML |

最新完整实验的独立计分：

| 时间口径 | 完整正确 | 至少一个字段正确 | 平均字段分 |
|---|---:|---:|---:|
| 官方 ±60 秒 | 17/136 | 41/136 | 0.200368 |
| 历史 ±300 秒 | 28/136 | 60/136 | 0.308015 |

最新推理源码指纹为 `fc3983009443a0026fcc4efec4cd01237add4cb2ec314599b2238a03d21f3f4c`。两种时间口径必须分开；旧报告中的阶段性进度、测试数和历史候选状态也不能当作当前状态。

## 怎样查一题

从项目根目录执行，工具只使用 Python 标准库：

```bash
# 列出第1题保存了哪些模型调用日志
python3 -m tools.experiment_records list --row 1 --contains calls

# 查看第1题9B第一轮的实际决策
python3 -m tools.experiment_records show --row 1 --path coordinator_1800/decision_0.json

# 将第1题的记录解包到一个新的本地目录，便于用普通文件工具分析
python3 -m tools.experiment_records materialize --row 1 --output outputs/review_row_1

# 上一完整实验的同题评价
python3 -m tools.experiment_records show --run previous_full --row 1 --path evaluation/result.json
```

`materialize` 拒绝覆盖已有目录。它会恢复原来的文件名及可用的冻结推理源码，不会补造原始数据库，也不运行模型。

重点文件的区别：

- `nodes_1800/calls.jsonl`：4B 实际收到的消息和实际回复，包含重试。目录输入不等于全部原始数据。
- `nodes_1800/device_*/step000/input.json` 和 `frames/c1800/device_*.json`：程序落盘的节点事实、目录和来源索引。
- `nodes_1800/device_*/report.json`：整理后的报告，其中部分解释字段由程序生成，不能全当成 4B 填写。
- `coordinator_1800/input_*.json`：程序保存的候选库、绑定和装配状态，不等于当轮模型看到了全部候选内容。
- `coordinator_1800/calls.jsonl`：9B 的真实输入 `messages` 和原始输出。判断“模型是否看到”时，以这里的实际消息为准；候选片段、目录编号和证据列表需要分别检查。
- `coordinator_1800/queries/`、`node_followup_*` 和 `node_calls_*.jsonl`：补查工具结果、节点答复和相关实际调用。
- `coordinator_1800/prediction.json`：被冻结的最终预测。
- `evaluation/result.json`：答案冻结后的独立评价，含标签。**不要把评价标签追加到重放模型的输入。**

上述文件中的服务器绝对路径是历史记录。其他机器可以阅读这些证据，但不能直接假定那些本地路径存在。

## 压缩和校验

模型调用 `.jsonl` 保持原始纯文本；较大的 JSON 使用 gzip 无损压缩，以减少仓库体积。每题 `case_manifest.json` 记录原文件名、存储位置、原始字节数、SHA256 和压缩方式。读取工具自动解压，不会截断或重写历史内容。

```bash
# 核对一题的原始字节哈希
python3 -m tools.experiment_records verify --row 1

# 核对全部归档；只读取和计算哈希，不加载模型
python3 -m tools.experiment_records verify
```

历史报告中的 gzip JSON 也可直接用标准库读取，例如：

```python
import gzip
import json
from pathlib import Path

path = Path('experiments/records/reports/latest_experiment_review_20261103/audit.json.gz')
with gzip.open(path, 'rt', encoding='utf-8') as stream:
    audit = json.load(stream)
print(audit['score_summary'])
```

## 恢复三个前端案例

在新克隆、还没有 `web/trace/data/` 的仓库中：

```bash
python3 -m tools.experiment_records frontend
python3 -m tools.build_standalone_trace
```

然后打开 `web/trace/EdgeRCA_trace.html`。若本地数据目录已存在，工具会拒绝覆盖；可以先用 `--output` 指定新的目录保存副本。

## 没有包含的内容

完整 Bank 原始遥测、`inbox.sqlite`、模型权重、虚拟环境和重复的大型备份包仍在服务器或原始来源。没有原始遥测时，不能用本包重算全部指标或完整调用链；没有模型时，不能重新进行真实推理。本包适合核对已保存的处理行为、比较预测和评价、定位信息流损失。

导出工具为 `python3 -m tools.export_experiment_records`，默认写入一个新的 `experiments/records` 目录。它不修改原始实验，不重新申请资源，也不覆盖已有归档。
