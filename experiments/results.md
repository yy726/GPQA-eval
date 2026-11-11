# 结果汇总 / Results

> 由 Colab T4 上的 `notebooks/11_sft_grpo_unsloth.ipynb`（快速评测）和 `notebooks/10_iterate_0p6b.ipynb`（锁定评测）产出，数字只包含聚合指标。

## notebook 11（SFT + GRPO；快速评测，只在同一次运行内比较）

快速评测：GPQA Diamond 子集，题面同 lm-eval 模板，temperature 0.6，1 个样本；**不是**排行榜分数。

### run1 · 2026-11-10 · Qwen3-0.6B-Base · `quick`（[带输出的 notebook](runs/run1_quick_qwen3-0.6b.ipynb)）

环境：T4，unsloth 2026.11.3，vllm 0.33.0（TRITON_ATTN，fp16），trl 1.15.0，transformers 5.17.0，torch 2.14.0。
数据：Dolci-Instruct-SFT 200 条（≤1024 token）+ SciQ 格式样本 200 条，去污染删除 0 条；GRPO 用另外 180 道 SciQ。

| 阶段 | acc_flex | acc_strict | 可提取率 | 截断率 | 平均 token | 评测用时 |
|---|---|---|---|---|---|---|
| 0_base | 16.7 | 0.0 | 60.0 | 56.7 | 699 | 1:47 |
| 1_sft（40 步，1 分钟） | 18.3 | 18.3 | 88.3 | **100** | 1024 | 3:51 |
| 2_grpo（30 步，22 分钟） | 18.3 | 18.3 | 81.7 | **100** | 1024 | 3:50 |

GRPO 日志：总奖励 1.02 → 1.06；格式奖励 0.04 → 0.02；回答长度始终 1024 token。SFT loss 1.35 → 1.37。

**观察**
1. 流程在 T4 上端到端跑通（安装 → 加载 → 评测 → SFT → GRPO → 合并保存）。
2. **模型答完不停**：Base tokenizer 的 eos 是 `<|endoftext|>`，ChatML 每轮以 `<|im_end|>` 结束，没被当成结束符。SFT 后每个回答都打满 1024 token 并继续编造新轮次；GRPO 因此每步都生成满长度（慢），而且"恰好一次 `The answer is (X)`"的格式奖励被重复内容破坏，几乎为 0，GRPO 没有有效信号。→ 已修复（见下）。
3. SFT 确实教会了格式：可提取率 60% → 88%，strict 从 0 → 18%。
4. 60 题的标准误约 ±5 分，三个阶段的准确率差异不显著；0.6B 本身也接近随机水平（25%），符合预期。GRPO 后预测字母偏向 D（25/60），是没停住 + 无效奖励的副作用。

**修复（commit 见 git log）**：加载后把 `tokenizer.eos_token` 设为 `<|im_end|>`、`generation_config.eos_token_id = [<|im_end|>, <|endoftext|>]`；快速评测和 GRPO 采样都传 `stop_token_ids`；Base tokenizer 缺 chat_template 时自动补；`prepare_dolci_sft.py` 跑完强制退出（run1 里卡住只能 ^C）。

### run2 · 2026-11-11 · Qwen3-1.7B-Base · `medium` · **A100 40GB**（[带输出的 notebook](runs/run2_medium_qwen3-1.7b_a100.ipynb)）

环境：A100-SXM4-40GB，unsloth 2026.11.3，vllm 0.31.0，trl 1.15.0，transformers 5.17.0，torch 2.13.0。`MAX_SEQ_LEN` 手动改成 4096（2048 时 GPQA 提示 + 2048 生成会超出上下文）。
数据：Dolci-Instruct 1000 条（不按来源过滤）+ SciQ 格式样本 500 条；GRPO 用 1081 道 SciQ。快速评测：GPQA Diamond 全部 198 题（标准误约 ±3 分）。

| 阶段 | acc_flex | acc_strict | 可提取率 | 截断率 | 平均 token | 评测用时 |
|---|---|---|---|---|---|---|
| 0_base | 16.2 | 0.5 | 46.5 | 39.4 | 1028 | 1.0 分钟 |
| 1_sft（150 步，1.6 分钟） | 23.2 | 24.2 | 86.9 | **95.5** | 1958 | 1.4 分钟 |
| 2_grpo（150 步，17.2 分钟） | 24.2 | 22.2 | 85.4 | **94.9** | 1949 | 1.4 分钟 |

GRPO 日志：总奖励 2.49 → 2.83；格式奖励 0.84 → 0.99；正确率奖励 1.65 → 1.84（约 92% 答对）；回答长度 762 → 767（上限 768）。SFT loss 1.06 → 0.83。

**观察**
1. A100 速度：1.7B 的 SFT 约 6.6k tok/s（150 步约 63 万 token），GRPO 约 7 秒/步（16 个回答 × ≤768 token），198 题评测约 1 分钟。
2. **还是不停**：eos 修复后截断率仍约 95%；GRPO 回答几乎全是满长度，而格式奖励却到了 0.99——模型写出一次 `The answer is (X)` 之后继续生成别的内容。lm-eval 的 flex 取最后一个字母，这会直接伤害正式分数。→ 本次加了 `<|im_start|>` 和字符串形式的停止条件、**截断惩罚奖励**，以及诊断格（训练标签末尾、SciQ 原始输出）。
3. base → SFT 的提升（16 → 23）主要来自"能提取出答案"（47% → 87%）；只看提取出来的，准确率约 27%，接近随机。
4. **SciQ 对 GRPO 来说太简单**（约 92% 答对，组内差异小），需要更难的题库 → 新增 `GRPO_DATA="mmlu_pro_stem"`。
5. Dolci-Instruct 的来源里只有约 7% 是科学（OpenThoughts3+ Science），代码、多语言、指令遵循占了很大比例 → 新增按 `source_dataset` / `domain` 过滤。

### 下一步：A100 上 5 次实验（Qwen3-4B-Base，`MODE="a100"`，每次约 3 小时）

| # | 配置 | 要回答的问题 |
|---|---|---|
| X1 | Instruct（科学/数学/逻辑）2 万 + SciQ 2000 → GRPO SciQ 150 步（含截断惩罚）；`PUSH_SFT_REPO` 保存 SFT | 答完会不会停；4B 基线 |
| X2 | `THINK_N=8000`（≤3072 token 的短思维链）+ Instruct 8000 + SciQ 2000 → 同样的 GRPO | 短思维链有没有帮助 |
| X3 | `RUN_SFT=False`、`INIT_MODEL`=X1/X2 中更好的 SFT → `GRPO_DATA="mmlu_pro_stem"` | 更难的 RL 数据能不能提分 |
| X4 | 按 X3 的 W&B 曲线调一项：`GRPO_LR` 5e-6 → 2e-5，或步数 / 组大小 | GRPO 调参 |
| X5 | 最佳配置 → 合并 → 推 HF → `main` 的 `01` 锁定评测（4 seeds） | 正式分数（注明 A100，不参与排名） |

## notebook 10（锁定评测）

| ID | seeds | GPQA Diamond | GPQA Main | 可提取率（flex，Diamond） | 截断率（Diamond） | 训练时长 (h) | 备注 |
|---|---|---|---|---|---|---|---|
| E0_base | | 待跑 | | | | — | |
| E1_think_sft | | 待跑 | | | | | |
| E2_instruct_sft | | 待跑 | | | | | |
| E3_think_strip | | 待跑 | | | | | |
