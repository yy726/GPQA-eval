# 实验记录 / Experiment log（分支 `submission-yy726`）

这是 yy726 的参考提交分支：**Qwen3-0.6B-Base + OLMo 3 Dolci 数据**，在 Colab T4 上迭代。
比赛规则和锁定评测在 `main` 分支；这里额外加了训练代码：

| 文件 | 作用 |
|---|---|
| `notebooks/10_iterate_0p6b.ipynb` | **一键实验**：Dolci 数据 → 去污染 → QLoRA SFT → 合并 → 锁定评测 → 诊断 JSON |
| `notebooks/02_sft_dolci_lora.ipynb` | 单独的 SFT notebook（推到 Hub，适合配合 `01` 分开跑） |
| `notebooks/03_grpo_mcq_lora.ipynb` | 进阶：选择题 GRPO |
| `scripts/prepare_dolci_sft.py` | Dolci 流式读取、按来源和长度过滤、可选去掉 `<think>` |
| `scripts/analyze_samples.py` | 从本地逐题日志算聚合诊断（准确率、可提取率、截断率、长度、字母分布），不输出题目 |

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yy726/GPQA-eval/blob/submission-yy726/notebooks/10_iterate_0p6b.ipynb)

## 第一轮实验矩阵

| ID | 配置 | 要回答的问题 |
|---|---|---|
| `E0_base` | Qwen3-0.6B-Base，不训练 | 起点分数；Base 会不会按 `(A)`–`(D)` 格式作答 |
| `E1_think_sft` | Dolci-Think-SFT，来源含 math/science/stem，5k 条，≤4k token，QLoRA r=32，LR 2e-4，1 epoch | 长思维链 SFT 在 4096 token 生成预算下有没有用 |
| `E2_instruct_sft` | Dolci-Instruct-SFT，同样的过滤和数量 | 短回答风格 vs 长思维链 |
| `E3_think_strip` | 与 E1 相同的数据，但去掉 `<think>` | 同一批题目，只改答案风格 |

建议顺序：先 `SMOKE=True` 跑一次 E1 确认流程，再 E0 → E1 → E2 → E3（都用 `SEEDS="0"`）。

## 怎么读诊断

- `scores.gpqa_diamond.mean`：主指标（1 个 seed 时噪声约 ±3 分；0.6B 接近 25% 的随机水平是正常的）。
- `diagnostics.*.flexible-extract.extract_rate`：能提取出答案字母的比例。低说明是**格式问题**，SFT/GRPO 很容易改善。
- `diagnostics.*.length.truncation_rate`：打满 4096 token 的比例。高说明**思维链太长被截断**，可以考虑去掉 `<think>`（E3）或筛选更短的样本。
- `pred_letter_dist`：如果几乎全是同一个字母，说明模型在"猜"，分数没有意义。

每次运行后，把 notebook 最后打印的 JSON 贴给 Claude，结果汇总到 [results.md](results.md)。
