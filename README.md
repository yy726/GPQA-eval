# GPQA-eval：一张 T4 上的后训练 Hackathon

**用免费的 Colab T4，对开源模型做后训练，在 GPQA 上冲分。**
Post-train an open model on a single free Colab T4 GPU and hill-climb GPQA. Public data, public weights, locked eval, reproducible by anyone.

> 本仓库是 RLHF / 后训练课程 Demo Day 的比赛套件，任何人都可以 fork 使用（Apache-2.0）。
> This repo is the kit for a post-training course demo day. Anyone may use or adapt it (Apache-2.0).

---

## 一句话规则 / TL;DR

| | |
|---|---|
| **起点模型** | [`Qwen/Qwen3.5-4B-Base`](https://huggingface.co/Qwen/Qwen3.5-4B-Base)（主赛道）或 [`Qwen/Qwen3.5-2B-Base`](https://huggingface.co/Qwen/Qwen3.5-2B-Base)（迭代快） |
| **训练数据** | 公开数据集，推荐 AI2 OLMo 3 的 **Dolci** 系列（SFT / DPO / RL） |
| **算力** | 只用 Colab T4（16GB），训练总时长 ≤ 4 小时 |
| **主指标** | **GPQA Diamond**（198 题，4 选 1），lm-evaluation-harness，4 个 seed 取平均 |
| **第二指标** | GPQA Main（448 题） |
| **提交** | 公开 HF 模型 + 训练 notebook + `submissions/<team>.json`，以 PR 形式提交 |
| **天花板参考** | 官方后训练版 [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B)，用同一配置评测 |

**核心问题：一张 T4、几个小时，你能补回 Qwen 官方后训练带来的多少提升？**

详细规则见 [RULES.md](RULES.md)，当前排名见 [LEADERBOARD.md](LEADERBOARD.md)。

---

## 快速开始 / Quick start

| Notebook | 做什么 | 打开 |
|---|---|---|
| `00_smoke_test_t4.ipynb` | **助教赛前跑**：确认 vLLM / lm-eval / LoRA 在 T4 上能用，记录耗时和显存 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yy726/GPQA-eval/blob/main/notebooks/00_smoke_test_t4.ipynb) |
| `01_eval_gpqa.ipynb` | 锁定版评测，生成提交 JSON | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yy726/GPQA-eval/blob/main/notebooks/01_eval_gpqa.ipynb) |
| `02_sft_dolci_lora.ipynb` | 基线：Dolci 数据过滤 → 去污染 → LoRA SFT → 推到 Hub | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yy726/GPQA-eval/blob/main/notebooks/02_sft_dolci_lora.ipynb) |
| `03_grpo_mcq_lora.ipynb` | 进阶：选择题 GRPO（可验证奖励 RL） | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yy726/GPQA-eval/blob/main/notebooks/03_grpo_mcq_lora.ipynb) |

赛前准备（每位同学）：
1. 注册 Hugging Face 账号，在 [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) 页面**接受数据集条款**（需要人工点同意，提前做）。
2. 创建一个 **write** 权限的 HF token，放到 Colab 左侧 🔑 Secrets，名字叫 `HF_TOKEN`。
3. 确认 Colab 能分到 T4（Runtime → Change runtime type → T4 GPU）。

推荐流程：`02`（SFT）→ `01`（评测）→ 调数据和超参 → 可选 `03`（GRPO）→ `01` → 提交。

命令行用户也可以直接用脚本：

```bash
pip install -r requirements-eval.txt
bash scripts/eval_gpqa.sh my-team/my-model results/my-model "0 1 2 3"
python scripts/aggregate_results.py results/my-model --model my-team/my-model --team my-team --out submissions/my-team.json
```

---

## 为什么这样选 / Design choices

**为什么从 Base 开始，而不是 `Qwen3.5-4B`？**
官方后训练版在 GPQA Diamond 上已经约 76 分（官方 model card，thinking 模式）。它用了远超 T4 的算力和更强的教师模型。在 T4 上用 LoRA、拿 Dolci 去微调它，大概率会**掉分**。从 Base 出发上升空间大，而且官方版天然就是"天花板"参考。

**为什么是 GPQA？** 研究生水平的科学选择题，答案客观可验证，公开、被广泛引用。缺点是题少（Diamond 只有 198 题，单次评测标准误约 ±3.4 分），所以我们用 4 个 seed 取平均，并加上 GPQA Main 作为第二指标。榜单上差距小于 3 分视为并列。

**为什么是 Dolci？** AI2 为 OLMo 3 公开了完整的后训练数据（SFT、DPO、RLVR 三个阶段，Think 和 Instruct 两种风格），许可清晰，正好对应课程里的 SFT、偏好优化、强化学习几讲。

**为什么不用公开排行榜？** Hugging Face Open LLM Leaderboard 已经停止接受提交（v1 于 2024 年归档，v2 于 2025 年退役），目前没有免费、可提交、又覆盖 GPQA 的公开榜。所以我们用"**公开权重 + 锁定评测配置 + 自建榜单 + 助教复评**"来保证任何人都能复现。

**为什么是 lm-evaluation-harness？** 社区标准评测框架，自带 GPQA 全部变体，配置可以完全锁定。我们用 `gpqa_diamond_cot_zeroshot`，指标是 `exact_match,flexible-extract`（取输出中最后一个 `(A)`–`(D)`）。

---

## 锁定评测配置 / Locked eval config

见 [`scripts/eval_gpqa.sh`](scripts/eval_gpqa.sh)：

- 任务：`gpqa_diamond_cot_zeroshot`、`gpqa_main_cot_zeroshot`
- 后端：vLLM（跑不通时回退 HF transformers）
- `dtype=float16`（T4 没有 bf16），`max_model_len=8192`，`max_gen_toks=4096`
- `temperature=0.6`，`top_p=0.95`，使用模型自带 chat template（`--apply_chat_template`）
- 4 个 vLLM 采样 seed（0–3）；lm-eval 的 `--seed` 固定为 0，保证所有人的选项顺序一致
- 报告均值 ± 跨 seed 标准误

⚠️ **GPQA 使用条款要求不要在网上公开题目。** `results/` 下的逐题日志包含题目，已加入 `.gitignore`；只提交汇总后的 JSON，不要把模型输出贴到公开的地方。

---

## T4 预算 / Compute budget

| 项目 | 估计 | 状态 |
|---|---|---|
| 4B LoRA（fp16）显存 | ~10 GB（Unsloth 官方数字） | 待 T4 实测 |
| 4B LoRA 训练吞吐 | ~1–2k tok/s → 4 小时约 10–25M token | 待 T4 实测 |
| Dolci 长思维链 | 先按长度过滤（≤ 4k token），再抽 2k–10k 条 | — |
| GRPO | 每步 2 题 × 4 采样，最长 768 token，几百步 | 待 T4 实测 |
| 评测（vLLM，4 seed） | 估计 20–40 分钟；HF 后端慢 5–10 倍 | 待 T4 实测 |

> "待 T4 实测"的数字会在助教跑完 `00_smoke_test_t4.ipynb` 后更新到这里。

### 已知风险 / Known risks

- **vLLM + Qwen3.5 on T4**：Qwen3.5 是 Gated DeltaNet 混合架构。vLLM ≥ 0.17 原生支持，但我们还没找到它在 T4（sm75）fp16 下跑通的确认。`00` 的 Test 1 专门测这个；如果不行，评测改用 HF 后端，主模型可能改为 2B。
- **Unsloth + Qwen3.5**：官方不建议 4bit QLoRA；需要 transformers v5；T4 上第一次运行要编译 Triton kernel，会比较慢。`02`/`03` 里留了 `USE_UNSLOTH=False` 的纯 transformers + PEFT 回退路径。
- **Colab 断线**：训练 checkpoint 会自动推到你的私有 `<repo>-ckpt` 仓库，可以续训。

---

## 仓库结构 / Layout

```
notebooks/      00 冒烟测试 · 01 评测 · 02 SFT 基线 · 03 GRPO 进阶
scripts/
  eval_gpqa.sh          锁定评测（与 01 相同）
  aggregate_results.py  多 seed 结果 → 提交 JSON（只读汇总指标，不含题目）
  prepare_dolci_sft.py  Dolci 流式读取、按来源和长度过滤、抽样，输出 TRL prompt-completion 格式
  decontam.py           训练集 vs GPQA 的 13-gram 重叠检查（不输出题目）
  build_leaderboard.py  校验 submissions/*.json，生成 LEADERBOARD.md
submissions/    schema.json · _example.json · ref-*.json（参考行）· <team>.json
tests/          脚本的 CPU 单元测试
```

---

## 提交流程 / How to submit

1. 用 `01_eval_gpqa.ipynb` 跑完 4 个 seed，得到 `submissions/<team>.json`。
2. 把里面的 `TODO` 填完：方法、数据集、训练时长、去污染结果、训练 notebook 链接。
3. Fork 本仓库，把 JSON 放进 `submissions/`，开 Pull Request。CI 会校验格式。
4. 助教对前几名用同一配置复评，复评分数为准（`verified: true`），合并后排行榜自动更新。

---

## 参考 / References

- GPQA: Rein et al., 2023. [arXiv:2311.12022](https://arxiv.org/abs/2311.12022) · [dataset](https://huggingface.co/datasets/Idavidrein/gpqa)
- Qwen3.5: [Qwen/Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B)
- OLMo 3 / Dolci: [Olmo 3 technical report](https://arxiv.org/abs/2512.13961) · [Dolci-Think-SFT-7B](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B) · [Dolci-Instruct-SFT-7B](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT-7B)
- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Unsloth Qwen3.5 fine-tuning guide](https://unsloth.ai/docs/zh/mo-xing/qwen3.5/fine-tune) · [TRL](https://github.com/huggingface/trl)
- 课程教材：[RLHF Book](https://rlhfbook.com)（第 4 章 SFT、第 6 章策略梯度 / GRPO、第 8 章 DPO）
