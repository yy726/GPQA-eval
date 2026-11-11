# 训练数据：配方、依据和候选数据集

GPQA 考的是研究生水平的物理、化学、生物选择题：**知识**（模型里有没有这块内容）和**推理**（能不能在 8192 token 的生成预算内推完）都要。

## 共享 SFT 检查点 S1 的配方（notebook 15）

### 论文里的证据 → 我们的选择

| 来源 | 发现 | 我们怎么做 |
|---|---|---|
| **MegaScience**（arXiv 2507.16812；在 Qwen2.5-7B 上逐个数据集训练再测） | GPQA-D：官方 Instruct 31.3；TextbookReasoning **34.3**；整个 MegaScience 混合 **36.4**；直接蒸馏 R1 长思维链的 SCP-116K 29.8、Nemotron-Science 29.3。结论：**教材题 + 核对过的短分步解答**比长思维链更有效也更省 token；TextbookReasoning 不做筛选最好（全部保留） | 教材题（物理 / 化学 / 生物为主）占 S1 的一半 |
| 同上 | Nemotron-Science（选择题）让选择题基准变好（MMLU-Pro 62.9）、计算题变差（SciBench 23.4 vs 教材题 44.1）；对它**按难度筛选**最好；LLM 去污染在其中查出 16.4 万条和基准重复 / 改写的题 | 选择题只占约 8%；只用 MegaScience 里去污染 + 难度筛选后的版本，不用原始的 `nvidia/OpenScience` |
| 同上 | Qwen3-4B-Base 全参训练 MegaScience（125 万条 × 3 epoch）：GPQA-D 34.9（官方 Qwen3-4B 非思考模式在同一套评测里 37.4） | 短解答 SFT 的上限大约在 35–40；要更高得靠 DPO / RL / 更长的推理 |
| **OpenThoughts3**（arXiv 2506.04178） | 领域内数据最重要：混入科学数据后，数学 / 代码数据对 GPQA 的迁移收益消失；每个领域混两个最好的问题来源就够，混得越多越差；科学数据的规模效应比数学、代码弱 | 科学约 80%，数学约 15%；科学的问题来源集中在教材题和 StackExchange / 有机化学（Dolci 的 OpenThoughts3 科学题） |
| **Dr.SCI**（arXiv 2602.08321；**同一个 Qwen3-4B-Base**） | 5 万条非思考数据 SFT：GPQA-D 43.6；1 万条思考数据：42.3（base 在他们的评测设定下 36.7）。按 4-gram 覆盖度选数据（EESFT）：SFT 后 +2，RL 后 +3。SFT 后再 RL：+2 到 +3 | 用短解答（同样的 token 能放 5 倍的题）；每个部分从 2 倍候选里按覆盖度挑 |
| **OLMo 3**（arXiv 2512.13961） | 流程 SFT → DPO → RL；Think 版 GPQA 45，Instruct 版 30（但 Think 的评测预算是 32k） | 流程照搬；思维链见下面的长度统计 |

### 抽样统计（2026-11-11，每个数据集随机抽 600–3000 行，用 Qwen3 的 tokenizer 数 token）

| 数据集 | 发现 |
|---|---|
| `MegaScience/MegaScience`（125 万；全量统计） | `textbook_reasoning` 65.1 万（数学 42.4 万、医学 8.2 万、生物 5.3 万、物理 4.1 万、化学 3.2 万）；`natural_reasoning` 42.9 万；`nemotro_science` 17.3 万，其中 **4 选 1 的 24,038 条**、10 选 1 的 12.8 万（含人文社科）。4 选 1 题里重写解答和参考答案**一致的 19,366 条（81%）**；正确选项 A/B/C/D = 5352 / 7220 / 6384 / 4955。回答长度（中位数）：选择题 700、物理 505、化学 358、生物 236、natural 1018 token |
| `allenai/Dolci-Think-SFT-7B`（227 万） | 科学（OpenThoughts3 科学题）约占 5%，回答**中位数 7200 token**（90% 是 `<think>`），≤4k 的 28%、≤6k 的 45%；数学中位数 2 万 token。→ 在 8192 预算和现有算力下不划算，S1 默认不用 |
| `allenai/Dolci-Instruct-DPO`（26 万对） | OpenThoughts3 科学题占 **11%（约 2.9 万对）**，chosen 中位数 1130 token、rejected 740；Tulu 数学约 6%。类型：delta learning 47%、LLM 判分 50%。chosen 比 rejected 长的占 74% → 用长度归一化的 DPO |
| `allenai/Dolci-Think-DPO-7B`（15 万对） | 科学题 chosen 中位数 4660 token（≤2k 的只有 8%），数学 7650 → 太长 |
| `nvidia/OpenScience`（`OS-Q3-235B-4`） | 4 选 1 科学题 + DeepSeek-R1 的 `<think>` 解答，中位数约 1200 token。和 GPQA 很像，但 MegaScience 的去污染在这个数据集里删掉了 27%，原始版本不建议直接用 |

### S1 的配方（`MODE="a100"`；在真实数据上跑过 `scripts/science_mix.py`）

| 部分 | 来源 | 条数 | token | 说明 |
|---|---|---|---|---|
| `mcq` | MegaScience 4 选 1 科学题，题面改成 GPQA 格式，解答结尾 `The answer is (X).` | 9000 | 7.0M | 只留两份解答一致的题；选项字母配平；按题目哈希取 55% |
| `textbook_physics / chemistry / biology` | MegaScience / TextbookReasoning | 20000 / 20000 / 15000 | 12.3M / 9.3M / 6.9M | 物理、化学几乎全用 |
| `textbook_medicine / math` | 同上 | 5000 / 8000 | 2.3M / 4.5M | |
| `natural` | MegaScience / NaturalReasoning | 8000 | 9.3M | 通用推理，解答较长 |
| `dolci_science` | Dolci-Instruct 的 OpenThoughts3 科学题 + SciRIFF | 16000 | 约 17M（估） | StackExchange 物理、有机化学 |
| `dolci_math` | Dolci-Instruct 的 Tulu 3 数学 | 8000 | 约 7M（估） | |
| 合计 | | 约 10.9 万 | 约 76M | 约 3.4 小时（6.2k token/秒）|

MegaScience 的 85,000 条按覆盖度选出来共 51.7M token（随机选是 46.8M）；选择过程本身不到 1 分钟。

4 选 1 题的拆分（按题目文本的哈希，所有 notebook 得到同样的结果）：**55% → S1 的 SFT；35% → GRPO / DPO 的题库（6343 题，字母配平后）；10% → 验证集（1768 题）**。
后两份的解答模型没见过，所以能当 RL 的题目和留出验证集。

### 下游怎么用数据

- **GRPO（notebook 11，`GRPO_DATA="science"`）**：50% 留出的 4 选 1 科学题 + 20% MMLU-Pro STEM + 30% Dr.SCI 数值题；每段开始前按当前模型的通过率预筛（留答对 1–3/4 的）。
- **DPO（notebook 12，`DPO_DATA="both"`）**：Dolci-Instruct-DPO 的科学 + 数学偏好对约 6000 对（delta learning 和 LLM 判分两种都用）+ 在留出的科学题上自己采样的"答对 vs 答错"约 1500 对；`sigmoid_norm`，β=5。
- **蒸馏（notebook 13）**：教材题（物理 / 化学 / 生物）4000 + GPQA 题面的选择题 2000，学生从 S1 开始；结束符位置不算损失。

## 其它候选数据集

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

## 怎么加别的数据集

**notebook 15（共享 SFT）**：配置格里的 `EXTRA_SFT`，和配方里的其它部分一起训练，长度同样 ≤ `sft_max_tokens`：

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

只换 notebook 15 的配方，其它不变（每个约 3–4 个 A100 小时，按额度取舍）：

| | 配方 | 要回答的问题 |
|---|---|---|
| S1 | 上面的默认配方 | 基线 |
| S1-nomcq | `MCQ_N = 0`，其余不变 | GPQA 题面的选择题样本贡献多少 |
| S1-think | `THINK_N = 5000`（Dolci-Think 科学思维链，≤6144 token，约多 1 小时） | 长思维链在 8192 预算下有没有用 |
| X1b | notebook 11 `RUN_SFT=True`（Dolci 科学 / 数学 2 万 + SciQ 格式） | 旧配方：Diamond 37.4 / Main 29.7（单样本） |

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

