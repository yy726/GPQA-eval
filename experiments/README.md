# 实验记录 / Experiment log（分支 `submission-yy726`）

这是 yy726 的参考提交分支：**Qwen3-0.6B-Base + OLMo 3 Dolci 数据**，在 Colab T4 上迭代。
比赛规则和锁定评测在 `main` 分支；这里额外加了训练代码：

| 文件 | 作用 |
|---|---|
| `notebooks/11_sft_grpo_unsloth.ipynb` | **推荐先跑**：照 Unsloth 官方 Qwen3 Instruct / GRPO notebook 的写法，SFT 预热（Dolci + 选择题格式）→ GRPO（SciQ 选择题，可验证奖励），在同一个 vLLM 引擎里对 base / SFT / GRPO 各做一次 GPQA 快速评测。`MODE="quick"` 约 15–25 分钟（待实测） |
| `notebooks/10_iterate_0p6b.ipynb` | 一键实验（只 SFT）：Dolci 数据 → 去污染 → QLoRA SFT → 合并 → **锁定配置**评测 → 诊断 JSON |
| `notebooks/02_sft_dolci_lora.ipynb` | 单独的 SFT notebook（推到 Hub，适合配合 `01` 分开跑） |
| `notebooks/03_grpo_mcq_lora.ipynb` | 进阶：选择题 GRPO |
| `scripts/prepare_dolci_sft.py` | Dolci 流式读取、按来源和长度过滤、可选去掉 `<think>` |
| `scripts/analyze_samples.py` | 从本地逐题日志算聚合诊断（准确率、可提取率、截断率、长度、字母分布），不输出题目 |

notebook 11（SFT + GRPO，快速）：[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yy726/GPQA-eval/blob/submission-yy726/notebooks/11_sft_grpo_unsloth.ipynb)
notebook 10（SFT + 锁定评测）：[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yy726/GPQA-eval/blob/submission-yy726/notebooks/10_iterate_0p6b.ipynb)

## 用多大的模型？

Qwen 官方后训练版的 GPQA Diamond（Qwen3 技术报告；随机猜 = 25%）：

| 模型 | non-thinking | thinking | 结论 |
|---|---|---|---|
| Qwen3-0.6B | 22.9 | 27.9 | 官方完整后训练都在随机水平附近 → **只适合验证流程**，能观察到的主要是格式指标（可提取率、截断率） |
| Qwen3-1.7B | 28.6 | 40.1 | **能看到 GPQA 变化的最小尺寸**；提升主要靠长推理，4096 token 预算会限制它 |
| Qwen3-4B | 41.7 | 55.9 | 变化最清楚；Unsloth 官方 T4 GRPO 示例用的就是 4B（16-bit LoRA） |

建议：0.6B + `MODE="quick"` 先确认流程 → 1.7B 跑 `full` 看 SFT / GRPO 的变化 → 4B 做最终提交。

## 第一轮实验矩阵

| ID | 配置 | 要回答的问题 |
|---|---|---|
| `E0_base` | Qwen3-0.6B-Base，不训练 | 起点分数；Base 会不会按 `(A)`–`(D)` 格式作答 |
| `E1_think_sft` | Dolci-Think-SFT，来源含 math/science/stem，5k 条，≤4k token，QLoRA r=32，LR 2e-4，1 epoch | 长思维链 SFT 在 4096 token 生成预算下有没有用 |
| `E2_instruct_sft` | Dolci-Instruct-SFT，同样的过滤和数量 | 短回答风格 vs 长思维链 |
| `E3_think_strip` | 与 E1 相同的数据，但去掉 `<think>` | 同一批题目，只改答案风格 |

建议顺序：先 `SMOKE=True` 跑一次 E1 确认流程，再 E0 → E1 → E2 → E3（都用 `SEEDS="0"`）。

**数据都是抽样的子集**：`prepare_dolci_sft.py` 流式读取 Dolci（不会下载全部 ~200 万条），用 shuffle buffer 随机打乱后只保留前 N 条满足来源和长度过滤的样本。notebook 10 默认 N=5000（`SMOKE=True` 时 200 条、训练 30 步、评测只跑 20 题）；notebook 11 的 `quick` 模式是 200 + 200 条、SFT 40 步、GRPO 30 步、GPQA 60 题。

## 怎么读诊断

- `scores.gpqa_diamond.mean`：主指标（1 个 seed 时噪声约 ±3 分；0.6B 接近 25% 的随机水平是正常的）。
- `diagnostics.*.flexible-extract.extract_rate`：能提取出答案字母的比例。低说明是**格式问题**，SFT/GRPO 很容易改善。
- `diagnostics.*.length.truncation_rate`：打满 4096 token 的比例。高说明**思维链太长被截断**，可以考虑去掉 `<think>`（E3）或筛选更短的样本。
- `pred_letter_dist`：如果几乎全是同一个字母，说明模型在"猜"，分数没有意义。

每次运行后，把 notebook 最后打印的 JSON 贴给 Claude，结果汇总到 [results.md](results.md)。

## 已知问题 / Known issues（Colab T4）

| 报错 | 原因 | 解决 |
|---|---|---|
| `RuntimeError: FlashInfer top-p/top-k sampling unavailable: unsupported compute capability 7.5. Unset VLLM_USE_FLASHINFER_SAMPLER=1.` | Unsloth（2026.11）在 T4 上给 vLLM 设置 `VLLM_USE_FLASHINFER_SAMPLER=1`；新版 vLLM（如 0.33）在计算能力 7.5 上不支持这个采样器，被强制打开时直接报错 | 在 `from unsloth import ...` / 加载模型之前设置 `os.environ["UNSLOTH_VLLM_NO_FLASHINFER"] = "1"` 和 `os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"`（notebook 10 / 11 已自动处理）。在已经报错的 runtime 里，设置后重跑加载那一格即可，不用重装 |
| 装出来的 vLLM 不是 notebook 里写的版本 | 安装输出被隐藏时，失败看不到 | notebook 已去掉 `%%capture`；装完会打印实际版本，贴回来时请一起附上 |
| SFT / GRPO 之后模型答完不停，截断率 100%，GRPO 每个回答都是满长度、格式奖励≈0 | `unsloth/Qwen3-*-Base` 的 eos 是 `<|endoftext|>`，ChatML 每轮以 `<|im_end|>` 结束（run1） | 加载后设置 `tokenizer.eos_token = "<|im_end|>"`、`model.generation_config.eos_token_id = [<|im_end|>, <|endoftext|>]`，采样时传 `stop_token_ids`（notebook 02/03/10/11 已处理）。这样保存的模型在 lm-eval 里也会停 |
| `prepare_dolci_sft.py` 报 chat template 相关错误 | `unsloth/Qwen3-*-Base` 的 tokenizer 没有 chat_template | 脚本已自动用 ChatML 兜底；notebook 里用 Unsloth 的 `qwen3-instruct` 模板 |
| `prepare_dolci_sft.py` 打印完结果后卡住 | 流式读取的后台线程不退出 | 脚本结尾已强制退出 |
