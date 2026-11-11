# 可能对 GPQA 有用的数据集 / Candidate datasets

GPQA 考的是研究生水平的物理、化学、生物选择题：**知识**（模型里有没有这块内容）和**推理**（能不能在 4096 token 内推完）都要。
Dolci-Instruct 里科学类只占约 7%（run2 的来源统计），所以值得混入专门的科学数据。

> 调研日期 2026-11-11。列名没有逐个核对（调研环境访问不了 Hugging Face）；`scripts/prepare_dolci_sft.py` 运行时会打印列名，
> 能识别 `messages` / `conversations` 对话列，也能识别"问题 / 答案分两列"（`question`/`problem`/`prompt` + `response`/`answer`/`solution`…）。
> 正式抽样前先用 `--list-sources 5000` 看一眼来源分布。

| 数据集 | 规模 / 内容 | 为什么可能有用 | 注意 |
|---|---|---|---|
| [`MegaScience/TextbookReasoning`](https://huggingface.co/datasets/MegaScience/TextbookReasoning) | 约 65 万，来自约 1.2 万本大学教材的题目 + 参考答案 | 大学水平物理 / 化学 / 生物知识，最贴近 GPQA 的"知识"部分；答案短，适合 4096 预算 | CC-BY-NC-SA-4.0 |
| [`MegaScience/MegaScience`](https://github.com/GAIR-NLP/MegaScience) | 约 125 万，多个公开科学数据集按消融挑选后的混合 | 论文里在 Qwen3 / Qwen2.5 / Llama3.1 base 上训练，超过官方 instruct 模型 | CC-BY-NC-SA-4.0 |
| [`MiniByte-666/Dr.SCI`](https://huggingface.co/datasets/MiniByte-666/Dr.SCI) | 约 89 万（[Dr.SCI 论文](https://arxiv.org/abs/2602.08321)的非官方复现；来源 WebInstruct-Verified、NaturalReasoning、MegaScience、RaR-Science） | 论文从 **Qwen3-4B-Base** 出发（SFT + RL），GPQA-Diamond **36.7 → 63.2**，和我们的设定几乎一样；有可验证题目，可作为 GRPO 题库 | 论文的评测预算可能比 4096 长；复现者说结果略低于论文；答案格式需要先看 |
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
