# 可能对 GPQA 有用的数据集 / Candidate datasets

GPQA 考的是研究生水平的物理、化学、生物选择题：**知识**（模型里有没有这块内容）和**推理**（能不能在生成预算内推完；锁定评测 2026-11-11 起为 8192 token）都要。
Dolci-Instruct 里科学类只占约 7%（run2 的来源统计），所以值得混入专门的科学数据。

> 调研日期 2026-11-11。列名没有逐个核对（调研环境访问不了 Hugging Face）；`scripts/prepare_dolci_sft.py` 运行时会打印列名，
> 能识别 `messages` / `conversations` 对话列，也能识别"问题 / 答案分两列"（`question`/`problem`/`prompt` + `response`/`answer`/`solution`…）。
> 正式抽样前先用 `--list-sources 5000` 看一眼来源分布。

| 数据集 | 规模 / 内容 | 为什么可能有用 | 注意 |
|---|---|---|---|
| [`MegaScience/TextbookReasoning`](https://huggingface.co/datasets/MegaScience/TextbookReasoning) | 约 65 万，来自约 1.2 万本大学教材的题目 + 参考答案 | 大学水平物理 / 化学 / 生物知识，最贴近 GPQA 的"知识"部分；答案短，容易在预算内收尾 | CC-BY-NC-SA-4.0 |
| [`MegaScience/MegaScience`](https://github.com/GAIR-NLP/MegaScience) | 约 125 万，多个公开科学数据集按消融挑选后的混合 | 论文里在 Qwen3 / Qwen2.5 / Llama3.1 base 上训练，超过官方 instruct 模型 | CC-BY-NC-SA-4.0 |
| [`MiniByte-666/Dr.SCI`](https://huggingface.co/datasets/MiniByte-666/Dr.SCI) | 约 89 万（[Dr.SCI 论文](https://arxiv.org/abs/2602.08321)的非官方复现；来源 WebInstruct-Verified、NaturalReasoning、MegaScience、RaR-Science） | 论文从 **Qwen3-4B-Base** 出发（SFT + RL），GPQA-Diamond **36.7 → 63.2**，和我们的设定几乎一样；有可验证题目，可作为 GRPO 题库 | 论文评测用 32k 上下文、avg@10；**没有 SFT 回答，只能当 RL 题库**；可验证题答案多为数字或公式（选项字母约 8%）；复现者说结果略低于论文 |
| [`open-thoughts/OpenThoughts3-1.2M`](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M)（`domain` = science） | 120 万里有 10 万道科学题（StackExchange 物理、有机化学教材抽题），QwQ-32B 长思维链，每题多个回答 | OpenThinker3-7B 的 GPQA-D 53.7 | 思维链很长，≤3k token 的样本少；Dolci 里已经有 OpenThoughts3 科学数据（run2 来源里有 "OpenThoughts3+ Science"），会重叠 |
| `facebook/natural_reasoning` | 约 280 万，从预训练语料里抽的推理题 + 参考答案 | 题目多样，可做 SFT 或 RL | 没找到单独的 GPQA 提升数字 |
| SCP-116K | 约 11.7 万道物理 / 化学 / 生物题，来自教材 | 大学水平，题解完整 | 同上 |

## 规则和去污染

- 都是赛前公开的离线数据集，符合规则（禁止的是比赛期间调用模型 API 现场生成数据）。
- 科学数据集和 GPQA 重叠的风险比通用数据高。notebook 里所有训练数据都会自动过 `scripts/decontam.py`（13-gram），丢掉命中的行；单独用时：
  `python scripts/decontam.py --train-jsonl data/xxx.jsonl`。

## 怎么用

**notebook 11（SFT + GRPO）**：配置格里的 `EXTRA_SFT`，和 Dolci 混在一起训练，长度同样 ≤ `sft_max_tokens`：

```python
EXTRA_SFT = [
    dict(dataset = "MegaScience/TextbookReasoning", n = 4000),
    dict(dataset = "open-thoughts/OpenThoughts3-1.2M", n = 2000, include = ["science"]),
]
```

**notebook 13（蒸馏）**：`DISTILL_SOURCES` 同样的写法。

**命令行**：

```bash
python scripts/prepare_dolci_sft.py --dataset MegaScience/TextbookReasoning --list-sources 5000
python scripts/prepare_dolci_sft.py --dataset MegaScience/TextbookReasoning --out data/textbook.jsonl \
    --n 4000 --tokenizer unsloth/Qwen3-4B-Base --max-tokens 3072
```

每项可选字段：`include` / `exclude`（匹配来源名和 `domain` / `subject` 列）、`config`（数据集子集名）、`split`、`strip_think`、`max_tokens`。

## 建议的对比实验（对应"最佳数据配方"奖）

在同样的 GRPO 设置下，只换 SFT 数据：

| | SFT 数据 | 要回答的问题 |
|---|---|---|
| A | Dolci 科学 / 数学 2 万（= X1） | 基线 |
| B | Dolci 科学 / 数学 8000 + Dolci-Think 短思维链 8000（= X2） | 短思维链有没有用 |
| C | Dolci 科学 / 数学 8000 + TextbookReasoning 8000 | 教材知识（短答案）有没有用 |

再配合 notebook 14 的知识探针看：C 如果让探针分数变高，说明它确实增加了知识。

## Dr.SCI 数据集字段（`MiniByte-666/Dr.SCI`，MIT）

两个 parquet：`Dr_SCI_verifiable.parquet`（41.5 万，约 190 MB）和 `Dr_SCI_open-ended.parquet`（47.6 万，约 710 MB）。每行：
- `prompt`：`[{"role": "user", "content": "Solve the following <subject> problem step by step. The last line of your response should be of the form: 'The final answer is: $\boxed{ANSWER}$' …\n\n<question>"}]`
- `reward_model.ground_truth`（参考答案）、`reward_model.rubric`（开放题的评分细则）
- `extra_info`：`question`、`reference_answer`、`subject`（math / physics / chemistry / biology / cs / medicine / economics / science）、`match_rule`（True = 可规则验证）、`from`（原始数据集）、`difficulty`（Qwen3-32B 8 次采样的通过率，0–1，越大越容易；8/8 的已删掉）

notebook 11 的 `GRPO_DATA="mix"` 只用可验证题里**参考答案能解析成数字**的（`recipe_utils.drsci_numeric_row`，物理/化学/生物），答案写在 `\boxed{}` 里、相对误差 ≤1% 算对；两段课程：通过率 0.5–0.875 → 0.25–0.375（0.0 的题连 32B 都做不对，对 4B 太难）。

## OLMo 3 的 DPO 数据（delta learning，可以上榜）

| 数据集 | 内容 | 适合我们吗 |
|---|---|---|
| [`allenai/Dolci-Instruct-DPO`](https://huggingface.co/datasets/allenai/Dolci-Instruct-DPO)（26 万对） | 约一半是 `preference_type="delta_learning"`：chosen = `qwen3-no_reasoning-32b`、rejected = `qwen3-no_reasoning-0.6b`；另一半是 LLM 判分的对。chosen 中位数约 270 token。来源在 `prompt_id` 里（`OpenThoughts3-full-filtered-science-no-cot-…`、`tulu-3-sft-personas-math-…`、`tulu_v3.9_sciriff_…`） | **适合**：简洁，正好针对"写太长停不下来"。notebook 12 默认 |
| [`allenai/Dolci-Think-DPO-7B`](https://huggingface.co/datasets/allenai/Dolci-Think-DPO-7B)（15 万对） | 全部是 delta learning：chosen = `qwen3-reasoning-32b`、rejected = `qwen3-reasoning-0.6b`（带 `<think>`）；来源在 `dataset_source` | 科学题 chosen 中位数约 7500 token，过滤到预算内后剩得少；可选 |

[Delta Learning](https://arxiv.org/abs/2507.06187)：偏好对两边都可以比被训练的模型弱，只要质量差明显，DPO 就能提分；直接在 chosen 上 SFT 反而有害（OLMo 3 Think：在 chosen 上继续 SFT 让 GPQA 从 45.1 掉到 40.2，DPO 则升到 48.4）。

