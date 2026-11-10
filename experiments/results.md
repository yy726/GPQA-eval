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

### 下一步

| 运行 | 配置 | 看什么 |
|---|---|---|
| run2 | 0.6B，`quick`，修复后 | 截断率应远低于 100%；GRPO 格式奖励明显上升、回答长度下降；GRPO 用时从 22 分钟降到几分钟 |
| run3 | `unsloth/Qwen3-1.7B-Base`，`medium` | 198 题上 base → SFT → GRPO 的真实变化 |
| run4 | run3 最好的阶段 → 合并 → 推到 HF → main 的 `01` 锁定评测 4 seeds | 可上榜的正式分数 |

## notebook 10（锁定评测）

| ID | seeds | GPQA Diamond | GPQA Main | 可提取率（flex，Diamond） | 截断率（Diamond） | 训练时长 (h) | 备注 |
|---|---|---|---|---|---|---|---|
| E0_base | | 待跑 | | | | — | |
| E1_think_sft | | 待跑 | | | | | |
| E2_instruct_sft | | 待跑 | | | | | |
| E3_think_strip | | 待跑 | | | | | |
