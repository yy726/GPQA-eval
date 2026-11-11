# 结果汇总 / Results

> 由 Colab（T4 / A100）上的 `notebooks/11_sft_grpo_unsloth.ipynb`（快速评测）和 `notebooks/10_iterate_0p6b.ipynb`（锁定评测）产出，数字只包含聚合指标。

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

### run3 = X1 · 2026-11-11 · Qwen3-4B-Base · `a100` · **A100 40GB**（[带输出的 notebook](runs/run3_x1_a100_qwen3-4b.ipynb)）

配置：Dolci-Instruct（科学/数学/逻辑）1 万条 + SciQ 格式 2000 条 → SFT 750 步（20.6 分钟）→ GRPO SciQ 150 步（38 分钟，每步 2 题 × 8，回答 ≤1536）。快速评测生成上限被截到 3374（`MAX_SEQ_LEN=6144`）。

| 阶段 | Diamond flex | Main flex | strict（D / M） | 可提取率（D / M） | 截断率（D / M） | 平均 token | 评测时间 |
|---|---|---|---|---|---|---|---|
| base | 25.8 | 24.6 | 0.5 / 0.2 | 78 / 81 | 41 / 37 | ~1600 | 7 分钟 |
| SFT | **34.8** | 28.1 | 30.8 / 30.1 | 91 / 92 | **97 / 97** | ~3300 | 16.5 分钟 |
| GRPO | 29.8 | **29.9** | 27.3 / 27.9 | 88 / 88 | **19 / 15** | ~780 | 4.6 分钟 |

（单样本；Diamond 标准误约 ±3.4，Main 约 ±2.1。）

观察：
1. **SFT 有效**：两个集合合起来（646 题）base 约 25.0 → SFT 约 30.2。
2. **GRPO 只学会了停**：截断率 97% → 15–19%，回答短了 4 倍，但合计准确率约 29.9，和 SFT 持平（Diamond −5、Main +1.8 都在噪声内）。日志：截断比例 0.71 → 0.02；SciQ 正确率奖励 1.75 → 1.90；结束时 **82% 的组内 8 个回答奖励完全相同**（`frac_reward_zero_std` 0.82）——SciQ 太简单，几乎没有学习信号。另外每步只有 2 道题，150 步一共才 300 道题。
3. **"答完不停"的根因找到了**：诊断输出里模型答完写的是 `\tNdrFc`（token 70564，C# 代码 `NdrFcShort` 的碎片，Qwen 词表里的欠训练 glitch token），然后用普通文本接着编 "What is the correct answer to this question:…"。训练标签里确实有 `<|im_end|>`，但 Qwen3-4B-Base 预训练几乎没输出过 `<|im_end|>`，**输出层这一行欠训练**，而 4B 的输入/输出嵌入是共享的、LoRA 改不了——SFT 学到的"该结束了"方向落到了邻近的 glitch token 上。`<|im_start|>` 同理，所以"新一轮"也是普通文本，stop 条件抓不到。GRPO 的截断惩罚逼模型改用预训练里见得多的 `<|endoftext|>` 结束（`finish=stop`、`stop_reason=None`）。
4. **修法（照 OLMo 3 / open-instruct）**：最后一个 assistant 回合以 eos=`<|endoftext|>` 结束（open-instruct 的 `olmo` 模板就是这样，`<|im_end|>` 只用在中间回合 / 工具调用），`generation_config.eos_token_id` 同时包含两个结束符；open-instruct 给新 token 的嵌入行"播种"，并且明确不支持和 LoRA 一起用——和我们遇到的问题一致。→ `recipe_utils.setup_chat(style="olmo")`。
5. 实测速度：SFT 6.6k token/秒（4B，810 万 token 20.6 分钟），GRPO 15 秒/步（16 个回答，≤1536），整轮约 1.6 小时。
6. 数据：`include=["logic"]` 抽到了 2900 条 Logic Puzzles（29%），科学只有约 19%。→ 去掉 logic。

### 之后的修改（notebook 11 新版，以及 12/13）

- 模板 `CHAT_STYLE="olmo"`（见上）；评测加 `repeat_rate` / `answer_then_more` / `glitch_rate`；SFT 后 teacher forcing 检查 eos 概率。
- **锁定评测生成预算 4096 → 8192**（main 规则，2026-11-11）。`a100` 预设：上下文 12288、快速评测 8192、SFT 样本 ≤6144、GRPO 回答 ≤6144，截断惩罚改成 DAPO 的软超长惩罚（最后 1024 token 线性扣到 −1）。
- SFT：去掉 logic；`SFT_SELECT="coverage"`（Dr.SCI 的 EESFT：按 4-gram 覆盖度从 3 倍候选里挑）。
- GRPO（照 OLMo 3 / DAPO / Dr.SCI）：MMLU-Pro STEM 选择题 + Dr.SCI 数值题；每段开始前按当前模型通过率预筛（留 1–3/4）；两段课程（Dr.SCI 通过率 0.5–0.875 → 0.25–0.375）；每步 4 道题 × 8；`epsilon_high=0.28`；常数学习率；格式分 0.5。
- 流程照 OLMo 3：**SFT → DPO → GRPO**（OLMo 3：RL 从 DPO 检查点开始更好）。DPO 用 OLMo 3 公开的 `allenai/Dolci-Instruct-DPO` 里的 delta learning 对（chosen = Qwen3-32B，rejected = Qwen3-0.6B），合规，可以上榜。

### 下一步（300 CU；时间为估计）

| 顺序 | notebook | 内容 | 时间（估） |
|---|---|---|---|
| X1b | 11 | SFT（新模板、无 logic、coverage 选择）；`RUN_GRPO=False`，**设 `PUSH_SFT_REPO`** | 约 1.2 小时 |
| D1 | 12 | `INIT_MODEL`=X1b SFT，`DPO_DATA="dolci"`（约 8000 对科学/数学 delta 对），**设 `PUSH_REPO`** | 约 40 分钟 |
| R1 | 11 | `RUN_SFT=False`、`INIT_MODEL`=D1，`GRPO_DATA="mix"`，两段课程 | 约 2–3 小时 |
| P1 | 14 | 知识探针：base / X1b SFT 各一次 | 约 2 × 40 分钟 |
| K1 | 13 | Qwen3-32B → 4B 蒸馏（只作参考） | 约 2 小时 |
| 终跑 | — | 最好的模型 → 推 HF（公开）→ `main` 的 `01` 锁定评测（8192） | 约 1.5 小时 |

成功标准：X1b 的 SFT 后截断率 < 30%、`glitch_rate` ≈ 0、teacher forcing 的 eos 概率接近 1；R1 的合计准确率高于 X1b。

## notebook 10（锁定评测）

| ID | seeds | GPQA Diamond | GPQA Main | 可提取率（flex，Diamond） | 截断率（Diamond） | 训练时长 (h) | 备注 |
|---|---|---|---|---|---|---|---|
| E0_base | | 待跑 | | | | — | |
| E1_think_sft | | 待跑 | | | | | |
| E2_instruct_sft | | 待跑 | | | | | |
| E3_think_strip | | 待跑 | | | | | |
