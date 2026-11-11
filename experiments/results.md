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

### X1 之后的修改（X1b 用的版本）

- 模板 `CHAT_STYLE="olmo"`（见上）；评测加 `repeat_rate` / `answer_then_more` / `glitch_rate`；SFT 后 teacher forcing 检查 eos 概率。
- **锁定评测生成预算 4096 → 8192**（main 规则，2026-11-11）。`a100` 预设：上下文 12288、快速评测 8192、SFT 样本 ≤6144、GRPO 回答 ≤6144，截断惩罚改成 DAPO 的软超长惩罚。
- SFT：去掉 logic；`SFT_SELECT="coverage"`（Dr.SCI 的 EESFT：按 4-gram 覆盖度从 3 倍候选里挑）。
- GRPO（照 OLMo 3 / DAPO / Dr.SCI）：MMLU-Pro STEM 选择题 + Dr.SCI 数值题；每段开始前按当前模型通过率预筛（留 1–3/4）；两段课程；每步 4 道题 × 8；`epsilon_high=0.28`；常数学习率；格式分 0.5。
- 流程照 OLMo 3：**SFT → DPO → GRPO**。DPO 用 OLMo 3 公开的 `allenai/Dolci-Instruct-DPO` 里的 delta learning 对（chosen = Qwen3-32B，rejected = Qwen3-0.6B），合规，可以上榜。

### run4 = X1b · 2026-11-11 · Qwen3-4B-Base · `a100` · **A100 40GB**（[带输出的 notebook](runs/run4_x1b_a100_qwen3-4b.ipynb)）

配置：Dolci-Instruct（科学/数学）6 万候选 → 覆盖度选 2 万条（4-gram 覆盖 880 万 vs 随机 603 万）+ SciQ 格式 2000 条，2170 万 token → SFT 1375 步（58 分钟，loss 0.80 → 0.56）→ GRPO `mix` 两段各 50 步（每步 4 题 × 8，学习率 5e-6，回答 ≤6144；每段预筛 4 分钟 + 训练 18 分钟）。没设 `PUSH_SFT_REPO`，SFT 模型没有保存到 HF。

| 阶段 | Diamond flex | Main flex | strict（D / M） | 可提取率（D / M） | 截断率（D / M） | 平均 token（D / M） | 评测时间 |
|---|---|---|---|---|---|---|---|
| base | 26.3 | 23.7 | 0.5 / 0.2 | 78 / 81 | 41 / 37 | 3619 / 3354 | 29 分钟 |
| SFT | **37.4** | 29.7 | 36.9 / 29.2 | 97 / 94 | **3.0 / 4.7** | 392 / 521 | 6 分钟 |
| GRPO | 30.8 | **30.8** | 30.8 / 30.6 | 97 / 96 | 3.5 / 3.6 | 440 / 438 | 5.5 分钟 |

（单样本，温度 0.6。Main 的 448 题包含全部 198 道 Diamond 题。）

**结束符修好了**：SFT 后截断率 97% → 3–5%，`glitch_rate` 0，teacher forcing 下答案结尾预测 `<|endoftext|>` 的概率中位数 0.9999；strict 和 flex 基本一致。

**为什么 GRPO 看起来比 SFT 差——其实是没有变化**：
1. **评测噪声**。单样本时 Diamond（198 题）一次评测的标准误约 3.4 分，两个阶段之差约 4.8 分；Diamond 的 −6.6 只有 1.4 个标准误。包含全部 Diamond 题的 Main 上 GRPO 是 +1.1（X1 也是同样的图案：Diamond −5.0、Main +1.8）。SFT 自己的两个数字也对不上：如果 Main 里那 198 道 Diamond 题也是 37.4%，剩下 250 题只有 23.6%，低于随机。→ 只能说"GRPO ≈ SFT"。
2. **GRPO 的训练奖励是平的**：正确性奖励第一段 0.93 → 1.08（10 步平均的标准误约 0.08），第二段 1.05 → 1.05；格式分一直在 0.45–0.48（满分 0.5），超长惩罚约 0。也就是说连训练分布上都没有学到东西。
3. **原因一：训练量太小**。100 次更新 × 4 道题 = 400 道题、每题只见一次、共 3200 个回答。OLMo 3 的 RL 每步 64 题 × 8，Dr.SCI 是几十万道题 × 多个 epoch，差 2–3 个数量级。学习率 5e-6 × 100 步，每个 LoRA 参数最多移动 5e-4。
4. **原因二：容易学的东西 SFT 已经学完了**。X1 里 GRPO"有用"是因为有一个所有题目都一致的信号（截断 71% → 2%）；X1b 的 SFT 已经会收尾、会写答案格式，剩下的只有"这道题选 B"这种每题不同的信号——一道选择题 1 bit、25% 能猜中，功劳还要摊到约 350 个 token 上，每步 4 道题时梯度几乎全是噪声。
5. **原因三：SFT 后的回答没有推理可以强化**。回答平均 300–400 token（最短 7–50 token），是"背一段相关知识 + The answer is (X)"的风格（来自 SciQ 格式样本和 Dolci-Instruct 的短答案）。RL 只能在模型已经会采样到的行为里做选择；诊断里 GRPO 后贪心回答反而更短（同一道 SciQ 题 226 → 55 token），它在往"更快作答"漂。
6. **SFT 的提分主要来自"答完"**。Main 上：可提取率 81% → 94%，而**有答案的题里的准确率**只是 29.3% → 31.4%（GRPO 后 32.2%）。总分 +6.0 里约 4 分来自收尾 / 格式，约 2 分来自答得更对。要再往上，需要真正提高"答对率"的手段：思维链 SFT（8192 预算下 `THINK_N` 现在可用）、DPO 的 32B vs 0.6B 偏好对、蒸馏。
7. 排除掉的嫌疑：字母偏置（按各阶段的选项分布算，随机猜的期望都是 24–25%）；Unsloth 的 GRPO 实现（`loss_type=dapo`、`beta=0`、组内归一化，vLLM 重要性采样修正默认关闭，权重确实变了）。
8. 实测速度：SFT 6.2k token/秒；GRPO 每步 21 秒（32 个回答，生成约 6 秒 + 训练约 4 秒，其余是偶尔写满 6144 token 的回答在拖）；预筛 2400 个回答 4 分钟。

### X1b 之后的修改（notebook 11 现在的版本，12 / 13 同步）

- **快速评测**（`recipe_utils.QuickEval`，11 / 12 / 13 / `quick_eval.py` 共用）：每题采样 `eval_samples` 次（a100 = 4），`acc_flex` 是 avg@k，附标准误；汇总里给**阶段之间的配对差值 ± 标准误**；Main 另报其中 Diamond 题和其余题的准确率。base 默认不再评测（`RUN_BASE_EVAL=False`，已知 26 / 24，省 30 分钟）。
- **GRPO 规模**：每步 16 题 × 8（原 4 × 8），120 步 = 1920 道题（原 400）；学习率 1e-5；预筛候选 2 倍。
- **留出验证集** `rl_val`：300 道不参与训练的同分布题（MMLU-Pro STEM + Dr.SCI 数值题），每题采样 4 次；训练前、每段之后各测一次，给出通过率和配对差值。它不涨 = GRPO 没学到能泛化的东西；它涨了但 GPQA 不涨 = 迁移问题。
- **回答上限**按模型自己的长度定（`GRPO_MAX_COMPLETION="auto"`：2 × 收尾长度的 99 分位）：上限 6144 而回答只有 400 token 时，偶尔一个停不下来的回答会让整步多等一分钟。
- 每段记录 `adapter_change`（LoRA 权重相对上一段变了多少）；`FINAL_ADAPTER` 可以选择合并 SFT 而不是 GRPO。
- 旧 notebook 02 / 03 / 10 也改成 `setup_chat(style="olmo")`（之前还是 `<|im_end|>` 结尾，会重现 X1 的问题）。

### X2 的计划：共享 SFT 检查点 S1 + 下游各做一次（2026-11-11 重新设计）

X1b 之后没有再跑 GRPO，而是先把数据配方和流程改了（依据和抽样统计见 [DATASETS.md](DATASETS.md)）：

- **notebook 15（新）**：共享 SFT 检查点 **S1**，只训练一次，推到 HF 私有仓库 `<用户名>/qwen3-4b-s1-sft`；11 / 12 / 13 的 `INIT_MODEL="auto"` 自动从它开始。
  配方约 10.9 万条、7600 万 token：MegaScience 教材题（物理 / 化学 / 生物为主）6.8 万 + GPQA 题面的 4 选 1 科学题 9000 + Dolci 科学 1.6 万 + NaturalReasoning 8000 + Dolci 数学 8000；
  每个部分按 4-gram 覆盖度从 2 倍候选里挑。LoRA r=128、学习率 1e-4（cosine）、有效 batch 32、1 epoch。检查点存 Google Drive，可续训；`MAX_TRAIN_HOURS` 限时。
- **notebook 11（GRPO）**：默认只做 GRPO。题库 `science` = 50% 留出的 4 选 1 科学题（和 GPQA 同题面，S1 没见过解答）+ 20% MMLU-Pro STEM + 30% Dr.SCI 数值题；
  120 步 × 16 题 × 8，学习率 1e-5；留出验证集；GPQA 每题采样 8 次。
- **notebook 12（DPO）**：Dolci-Instruct-DPO 的科学 + 数学偏好对约 6000 对（两种类型都用）+ 留出科学题上的在线偏好对约 1500 对；长度归一化 DPO（`sigmoid_norm`，β=5），学习率 1e-5。
- **notebook 13（蒸馏，参考）**：学生从 S1 开始；教材题 4000 + GPQA 题面选择题 2000；结束符位置不算损失。

为什么这样能让 GRPO 的差别看得出来：(1) S1 的回答是逐个分析选项的分步推理，采样出来的 8 个回答之间有"推理路径对不对"的差别可以选；
(2) 题库和 GPQA 同一种题面、同一个难度层次；(3) 用留出验证集（同分布，约 400 题 × 4 次采样）和 GPQA 的配对差值来量。
论文里同等规模的参照是 +1 到 +3 分（Dr.SCI：SFT 后 RL +2–3），不是大幅提升。

### 额度安排（剩余约 240 CU；A100 按前两次的消耗估算约 12 CU/小时 ≈ 20 小时；时间都是估计）

| 顺序 | notebook | 内容 | A100 小时 | CU |
|---|---|---|---|---|
| 0 | 15 `quick` | 0.6B 跑通流程（T4 也行） | 0.3 | 约 4 |
| 1 | 15 | **S1**：数据 0.5 + 训练 3.5 + 推送和评测 0.5 | 4.5 | 约 54 |
| 2 | 11 | **R1**：GRPO，从 S1 开始 | 3.3 | 约 40 |
| 3 | 12 | **D1**：DPO，从 S1 开始，推到 HF | 1.8 | 约 22 |
| 4 | 11 | **R2**：GRPO，从 D1 开始（OLMo 3 的顺序；只在 D1 ≥ S1 时做） | 3.3 | 约 40 |
| 5 | `main` 的 01 | 最好的模型做锁定评测（4 个 seed；可以用 T4） | 0.7 | 约 8 |
| — | 13 | K1：蒸馏（只作参考，不能上榜；额度有剩再做） | 2.2 | 约 26 |

1–5 合计约 168 CU，留约 70 CU 给失败重跑；做 K1 的话余量只剩约 45 CU。

判断标准：
- **S1**：截断率 < 10%，`mean_tokens` 在 600–1300，Main 上"有答案的题里的准确率"高于 X1b 的 31%。达不到就先别跑下游。
- **GRPO / DPO 有没有用**：看汇总里的配对差值（|z| ≥ 2 才算数）；GRPO 另看 `rl_val`。`rl_val` 不动 = 没学到能泛化的东西；`rl_val` 涨而 GPQA 不涨 = 迁移问题。
- 如果 R1 的 `rl_val` 不动，不要再加 GRPO 步数，把额度留给 DPO 和 S1 的配方消融（DATASETS.md 里的 S1-nomcq / S1-think）。

## notebook 10（锁定评测）

| ID | seeds | GPQA Diamond | GPQA Main | 可提取率（flex，Diamond） | 截断率（Diamond） | 训练时长 (h) | 备注 |
|---|---|---|---|---|---|---|---|
| E0_base | | 待跑 | | | | — | |
| E1_think_sft | | 待跑 | | | | | |
| E2_instruct_sft | | 待跑 | | | | | |
| E3_think_strip | | 待跑 | | | | | |
