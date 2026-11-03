# 相同证据、按题固定随机种子的最终裁决对照

这是最终裁决阶段的开发验证，复用已生成的节点与补查证据，不代表完整Bank新准确率。
全部预测冻结后，独立评分进程才读取标准答案。

| 分组 | 版本 | 题数 | 合法回复 | ±60秒完全正确 | ±60秒平均分 | ±300秒完全正确 |
|---|---|---:|---:|---:|---:|---:|
| development | baseline | 8 | 3 | 3 | 0.375 | 3 |
| development | candidate | 8 | 5 | 3 | 0.416 | 3 |
| development | reasoning | 8 | 6 | 3 | 0.416 | 3 |
| holdout | baseline | 8 | 7 | 1 | 0.219 | 1 |
| holdout | candidate | 8 | 7 | 1 | 0.219 | 1 |
| holdout | reasoning | 8 | 8 | 1 | 0.219 | 1 |
| all | baseline | 16 | 10 | 4 | 0.297 | 4 |
| all | candidate | 16 | 12 | 4 | 0.318 | 4 |
| all | reasoning | 16 | 14 | 4 | 0.318 | 4 |
