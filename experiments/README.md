# 实验记录 / Experiment log（分支 `submission-yy726`）

这是 yy726 的参考提交分支：**Qwen3-0.6B-Base + OLMo 3 Dolci 数据**，在 Colab T4 上迭代。
比赛规则和锁定评测在 `main` 分支；这里额外加了训练代码：

| 文件 | 作用 |
|---|---|
| `notebooks/11_sft_grpo_unsloth.ipynb` | **主力 notebook**：SFT（Dolci 科学/数学 + 可选思维链 / 额外数据 + 4-gram 覆盖度选样 + 选择题格式）→ GRPO（MMLU-Pro STEM + Dr.SCI 数值题，通过率预筛 + 两段课程 + 软超长惩罚，每步 16 题 × 8，留出验证集），OLMo 风格结束符，GPQA 快速评测（≤8192，每题采样 4 次 + 阶段间配对比较）+ 诊断，W&B。模式 `quick` / `medium` / `a100` |
| `notebooks/12_dpo_unsloth.ipynb` | **DPO**：从 11 的 SFT 模型出发，默认用 OLMo 3 公开的 Dolci-Instruct-DPO（delta learning：Qwen3-32B vs 0.6B），也可用自己采样的"答对且收尾 vs 答错或没停"；可迭代 |
| `notebooks/13_distill_gkd.ipynb` | **知识蒸馏（GKD）**：Qwen3-32B（4-bit）→ Qwen3-4B-Base，离线对齐老师的逐 token 分布。**只作参考，不参与排名** |
| `notebooks/14_decision_probe.ipynb` | **Decision 知识探针**：加打分头直接选选项（不生成文字），测 4B 不写思维链"知道"多少。只做分析，不能提交 |
| `scripts/recipe_utils.py` / `scripts/quick_eval.py` / `scripts/select_coverage.py` | 11–14 共用：聊天模板与结束符、题面、答案提取与诊断指标、多次采样的快速评测与配对比较（`QuickEval`）、GRPO 奖励（选择题 / 数值题 / 软超长惩罚）、Dr.SCI 与 Dolci DPO 转换、去污染、多数据源抽样、4-gram 覆盖度选样 |
| `experiments/DATASETS.md` | 可能对 GPQA 有用的数据集（MegaScience、Dr.SCI、OpenThoughts3…）和用法 |
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
| Qwen3-1.7B | 28.6 | 40.1 | **能看到 GPQA 变化的最小尺寸**；提升主要靠长推理，生成预算（现为 8192）会限制它 |
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

- `scores.gpqa_diamond.mean`：主指标（1 个 seed 时噪声约 ±3 分，两次评测之差约 ±5 分；0.6B 接近 25% 的随机水平是正常的）。
- notebook 11–13 的快速评测：`acc_flex` 是每题采样 k 次的平均，`acc_flex_se` 是标准误；**比较两个阶段只看汇总里的配对差值**（`compare`：差 ± 标准误，|z| < 2 = 看不出差别）。X1b 里单样本的 "SFT 37.4 → GRPO 30.8（Diamond）" 就是噪声（见 `results.md`）。
- `diagnostics.*.flexible-extract.extract_rate`：能提取出答案字母的比例。低说明是**格式问题**，SFT/GRPO 很容易改善。
- `diagnostics.*.length.truncation_rate`：打满生成预算（现为 8192 token；第一轮实验时是 4096）的比例。高说明**思维链太长被截断**，可以考虑去掉 `<think>`（E3）或筛选更短的样本。
- `pred_letter_dist`：如果几乎全是同一个字母，说明模型在"猜"，分数没有意义。

每次运行后，把 notebook 最后打印的 JSON 贴给 Claude，结果汇总到 [results.md](results.md)。

## 已知问题 / Known issues（Colab T4）

| 报错 | 原因 | 解决 |
|---|---|---|
| `RuntimeError: FlashInfer top-p/top-k sampling unavailable: unsupported compute capability 7.5. Unset VLLM_USE_FLASHINFER_SAMPLER=1.` | Unsloth（2026.11）在 T4 上给 vLLM 设置 `VLLM_USE_FLASHINFER_SAMPLER=1`；新版 vLLM（如 0.33）在计算能力 7.5 上不支持这个采样器，被强制打开时直接报错 | 在 `from unsloth import ...` / 加载模型之前设置 `os.environ["UNSLOTH_VLLM_NO_FLASHINFER"] = "1"` 和 `os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"`（notebook 10 / 11 已自动处理）。在已经报错的 runtime 里，设置后重跑加载那一格即可，不用重装 |
| 装出来的 vLLM 不是 notebook 里写的版本 | 安装输出被隐藏时，失败看不到 | notebook 已去掉 `%%capture`；装完会打印实际版本，贴回来时请一起附上 |
| SFT 之后模型答完不停（截断率 95–97%），答案后面出现 `\tNdrFc` 然后接着编下一题 | Qwen3 Base 预训练几乎没输出过 `<|im_end|>`，输出层这一行欠训练，4B 嵌入共享、LoRA 改不了；模型学成了输出邻近的 glitch token（run1–X1，见 `results.md`） | **照 OLMo 3：回答以 `<|endoftext|>` 结束**（`recipe_utils.setup_chat(style="olmo")`，notebook 02 / 03 / 10–13 都已改）。`generation_config` 里两个结束符都停，保存的模型在 lm-eval 里也会停 |
| `prepare_dolci_sft.py` 报 chat template 相关错误 | `unsloth/Qwen3-*-Base` 的 tokenizer 没有 chat_template | 脚本已自动用 ChatML 兜底；notebook 里用 Unsloth 的 `qwen3-instruct` 模板 |
| `prepare_dolci_sft.py` 打印完结果后卡住 | 流式读取的后台线程不退出 | 脚本结尾已强制退出 |

## A100（仅本分支，不参与排名）

比赛规则仍然是 T4；下面是 yy726 在 Colab A100 40GB 上探索上限用的预算（4B 的数字是 X1 / X1b 实测）。

**算力**：A100 每小时约 5.4–13 个 compute unit（以 Colab"资源"面板为准）。300 CU / 5 次实验 ≈ 每次 60 CU ≈ 4.6–11 小时；按**每次约 3 小时**规划，留余量给失败重跑和最后的锁定评测。

| 模型（A100 40GB） | 训练方式 | SFT 吞吐（估） | GRPO（估） |
|---|---|---|---|
| Qwen3-1.7B | 16-bit LoRA | ~6.6k tok/s（run2 实测） | ~7 秒/步（run2 实测，16 × ≤768 token） |
| **Qwen3-4B**（推荐） | 16-bit LoRA r=64 | **~6.6k tok/s（X1 实测）**，≈24M token/小时 | **~21 秒/步（X1b 实测，32 个回答，平均约 350 token）**；新版每步 128 个回答，估 30–45 秒/步 |
| Qwen3-8B | 16-bit LoRA | ~1.5–2k tok/s | 更慢；不在比赛允许的起点里 |

**每次实验（4B，`MODE="a100"`）**：SFT 2 万条（2170 万 token）约 1 小时（X1b 实测）；GRPO 120 步（每步 16 题 × 8）含预筛和验证约 1.5–2 小时（估）；快速评测（Diamond 198 + Main 448，≤8192，每题 4 次采样）回答正常结束时每个阶段约 10 分钟；base 不收尾，单次采样就要 30 分钟（默认跳过）。
**全量 Dolci 不现实**：Instruct 约 215 万条（≈1–1.5B token，4B 要 100+ 小时）；Think 约 190 万条（>10B token）。

**流程**（照 OLMo 3，详见 `results.md`）：notebook 11 SFT（`PUSH_SFT_REPO`）→ notebook 12 DPO（Dolci-Instruct-DPO 的 32B vs 0.6B delta 对，`PUSH_REPO`）→ notebook 11 GRPO（`RUN_SFT=False`、`INIT_MODEL`=DPO 模型）。
借鉴来源：OLMo 3 / open-instruct（结束符、SFT→DPO→RL、clip-higher、去掉无信号的组）、Dr.SCI（4-gram 覆盖度选 SFT 数据、按难度的课程、数值题）、DAPO（软超长惩罚）、Delta Learning（弱对弱的偏好对）。

## Weights & Biases

notebook 11 会自动记录（写法同 [rlhf-book `policy_gradients/train.py`](https://github.com/natolambert/rlhf-book/tree/main/code/policy_gradients)：环境变量优先，没有 key 就 `mode="disabled"`）：
1. 在 https://wandb.ai/authorize 拿 API key，放进 Colab 左侧 🔑 Secrets，名字叫 `WANDB_API_KEY`。
2. 可选：在配置格改 `WANDB_PROJECT`（默认 `gpqa-hackathon`）/ `WANDB_ENTITY`；`EXP_NAME` 决定 run 名（`X1-Qwen3-4B-Base-a100`）。
3. 记录内容：SFT 和 GRPO 的训练曲线（loss、各项奖励、回答长度、`frac_reward_zero_std`…）、每个阶段的 GPQA 快速评测（`gpqa/<set>/<stage>/*` 在 run summary 里，外加一张汇总表）、报告 JSON（artifact）、全部配置。
4. **不会**上传 GPQA 题目或模型生成内容（`log_completions=False`）。

## 更多配方（notebook 12–14）

都照 Unsloth 官方 notebook 的写法，配置格有 `quick`（0.6B / T4 验证流程）和 `a100` 两档；W&B、快速评测和报告 JSON 的格式同 notebook 11。
12–14 都还没在 GPU 上跑过：**先用 `MODE="quick"` 跑一遍**确认流程，再上 A100。

| notebook | 参考的 Unsloth 示例 | 做什么 | A100 时间（估） | 能上榜吗 |
|---|---|---|---|---|
| 12 DPO | [Zephyr (7B) DPO](https://github.com/unslothai/notebooks/blob/main/nb/Zephyr_(7B)-DPO.ipynb) | `INIT_MODEL` = 11 的 SFT；默认 `DPO_DATA="dolci"`：OLMo 3 的 Dolci-Instruct-DPO 里 delta learning 对（Qwen3-32B vs 0.6B），按来源留科学/数学；`"onpolicy"`：自己采样的答对且收尾 vs 答错或没停；`ROUNDS=2` 迭代 | `dolci` 约 40 分钟 | 能（公开数据集；T4 上重跑） |
| 13 蒸馏 | [Qwen3 (0.6B) Knowledge Distillation](https://github.com/unslothai/notebooks/blob/main/python_scripts/Qwen3_(0.6B)-Knowledge_Distillation.py) | TRL `GKDTrainer` + Unsloth 分块损失；老师 Qwen3-32B 4-bit（官方 GPQA-D 68.4 / 54.6，OOM 换 14B），学生 4B 16-bit LoRA；默认离线（`lmbda=0`），短答案数据 | 约 2 小时 | **不能**（老师是官方后训练模型） |
| 14 探针 | [Qwen3.5 (4B) Decision](https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_5_(4B)-Decision.ipynb) | `FastDecisionModel` 在 MMLU-Pro STEM + SciQ 上训练打分头，对 GPQA 只算准确率 / 分学科准确率；换 `BACKBONE` 对比 base 和 SFT | 约 40 分钟 | 不能（不生成文字） |

注意：
- 12 依赖 11 推到 HF 的 SFT 模型（11 里设 `PUSH_SFT_REPO`）。
- 13 安装时把 TRL 固定到 0.25.1（Unsloth 蒸馏示例测过的版本）；如果开头的"分块损失已接上"检查失败，Runtime → Restart session 后重跑。
- 13 和 14 的 GPQA 评测在独立进程里跑（`scripts/quick_eval.py` / 本地 `evaluate`），只输出聚合数字。

