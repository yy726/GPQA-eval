# GPQA-eval：一张 T4 上的后训练 Hackathon

> **🔀 这是 `submission-yy726` 分支**：yy726 的参考提交（Qwen3-0.6B + Dolci SFT）。比赛规则以 `main` 分支为准。
> 训练代码和实验记录见 [experiments/README.md](experiments/README.md)，一键实验：[`notebooks/10_iterate_0p6b.ipynb`](https://colab.research.google.com/github/yy726/GPQA-eval/blob/submission-yy726/notebooks/10_iterate_0p6b.ipynb)。

**用免费的 Colab T4，对开源模型做后训练，在 GPQA 上冲分。**
Post-train an open model on a single free Colab T4 GPU and hill-climb GPQA. Public data, public weights, locked eval, reproducible by anyone.

> 本仓库是 RLHF / 后训练课程 Demo Day 的比赛套件，任何人都可以 fork 使用（Apache-2.0）。
> This repo is the kit for a post-training course demo day. Anyone may use or adapt it (Apache-2.0).

---

## 一句话规则 / TL;DR

| | |
|---|---|
| **起点模型** | [`Qwen/Qwen3-4B-Base`](https://huggingface.co/Qwen/Qwen3-4B-Base)（主赛道）、[`Qwen/Qwen3-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-1.7B-Base) 或 [`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base)（迭代快 / 调试） |
| **训练数据** | 公开数据集，推荐 AI2 OLMo 3 的 **Dolci** 系列（SFT / DPO / RL） |
| **算力** | 只用 Colab T4（16GB），训练总时长 ≤ 4 小时 |
| **主指标** | **GPQA Diamond**（198 题，4 选 1），lm-evaluation-harness，4 个 seed 取平均 |
| **第二指标** | GPQA Main（448 题） |
| **提交** | 公开 HF 模型 + 训练 notebook + `submissions/<team>.json`，以 PR 形式提交 |
| **天花板参考** | 官方后训练版 [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B)，用同一配置评测（官方报告 GPQA Diamond：non-thinking 41.7 / thinking 55.9） |

**核心问题：一张 T4、几个小时，你能补回 Qwen 官方后训练带来的多少提升？**

详细规则见 [RULES.md](RULES.md)，当前排名见 [LEADERBOARD.md](LEADERBOARD.md)。

---

## 快速开始 / Quick start

| Notebook | 做什么 | 打开 |
|---|---|---|
| `00_smoke_test_t4.ipynb` | **助教赛前跑**：确认 vLLM / lm-eval / LoRA 在 T4 上能用，记录耗时和显存 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yy726/GPQA-eval/blob/main/notebooks/00_smoke_test_t4.ipynb) |
| `01_eval_gpqa.ipynb` | 锁定版评测，生成提交 JSON | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yy726/GPQA-eval/blob/main/notebooks/01_eval_gpqa.ipynb) |

赛前准备（每位同学）：
1. 注册 Hugging Face 账号，在 [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) 页面**接受数据集条款**（需要人工点同意，提前做）。
2. 创建一个 **write** 权限的 HF token，放到 Colab 左侧 🔑 Secrets，名字叫 `HF_TOKEN`。
3. 确认 Colab 能分到 T4（Runtime → Change runtime type → T4 GPU）。

**训练代码由各队自己实现。** 这个仓库只提供比赛框架（规则、锁定评测、去污染、排行榜），不提供训练解法。常用工具：
- 数据：AI2 OLMo 3 的 Dolci 系列，例如 [Dolci-Think-SFT-7B](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B)、[Dolci-Instruct-SFT-7B](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT-7B)，以及对应的 DPO / RL 数据
- 训练：[Unsloth](https://unsloth.ai/docs/get-started/unsloth-notebooks)（T4 上省显存，有 Qwen3 的官方 Colab notebook）、[TRL](https://github.com/huggingface/trl)（SFTTrainer / DPOTrainer / GRPOTrainer）
- 教材：[RLHF Book](https://rlhfbook.com) 第 4（SFT）、6（策略梯度 / GRPO）、8（DPO）章
- **训练集必须先过 `scripts/decontam.py`**（见 RULES）

参考实现：分支 [`submission-yy726`](https://github.com/yy726/GPQA-eval/tree/submission-yy726)（Qwen3-0.6B + Dolci SFT 的一个示例提交）。

推荐流程：自己的训练 notebook → `01`（评测）→ 调数据和超参 → `01` → 提交。

命令行用户也可以直接用脚本：

```bash
pip install -r requirements-eval.txt
bash scripts/eval_gpqa.sh my-team/my-model results/my-model "0 1 2 3"
python scripts/aggregate_results.py results/my-model --model my-team/my-model --team my-team --out submissions/my-team.json
```

---

## 为什么这样选 / Design choices

**为什么是 Qwen3-4B-Base？——在 T4 + vLLM 上问题最少。**
我们比较了几个候选（2026-11 的状态）：

| 候选 | T4（sm75，fp16）+ vLLM | 结论 |
|---|---|---|
| **Qwen3-4B / 1.7B / 0.6B-Base** | 标准稠密 Transformer（GQA，head_dim 128，纯文本），走 vLLM 最成熟的路径；Unsloth 官方的 Qwen3 (4B) GRPO notebook 就在 T4 上用 vLLM | ✅ 选用 |
| Qwen3.5-4B-Base | Gated DeltaNet 混合架构 + 视觉编码器；sm75 上有未解决的问题（[vllm#36589](https://github.com/vllm-project/vllm/issues/36589)）；**Colab T4 实测 vLLM 报 `Engine core initialization failed`** | ❌ 不可用 |
| Gemma 4 E2B（Google / Unsloth 版架构相同） | 全局注意力 head_dim=512，Triton kernel 需要 96KB 共享内存，T4 只有 64KB，启动即报错（[vllm#38918](https://github.com/vllm-project/vllm/issues/38918)，修复 [PR #39018](https://github.com/vllm-project/vllm/pull/39018) 未合并） | ❌ 需要打补丁 |
| OLMo 3 Base | 只有 7B / 32B；7B 的 fp16 权重约 14–15GB，T4 放不下 | ❌ 尺寸不合适 |

另外 Qwen3 是 Apache-2.0、不需要申请访问，和 [RLHF Book 代码](https://github.com/natolambert/rlhf-book/tree/main/code) 用的 Qwen3-0.6B/1.7B 是同一系列。

**为什么从 Base 开始，而不是官方的 `Qwen3-4B`？**
官方后训练版在 GPQA Diamond 上已经有 41.7（non-thinking）/ 55.9（thinking）（官方 model card）。它用了远超 T4 的算力和更强的教师模型，在 T4 上用 LoRA 再微调它很难涨分。从 Base 出发上升空间大，而且官方版天然就是"天花板"参考。

**为什么是 GPQA？** 研究生水平的科学选择题，答案客观可验证，公开、被广泛引用。缺点是题少（Diamond 只有 198 题，单次评测标准误约 ±3.4 分），所以我们用 4 个 seed 取平均，并加上 GPQA Main 作为第二指标。榜单上差距小于 3 分视为并列。

**为什么是 Dolci？** AI2 为 OLMo 3 公开了完整的后训练数据（SFT、DPO、RLVR 三个阶段，Think 和 Instruct 两种风格），许可清晰，正好对应课程里的 SFT、偏好优化、强化学习几讲。

**为什么不用公开排行榜？** Hugging Face Open LLM Leaderboard 已经停止接受提交（v1 于 2024 年归档，v2 于 2025 年退役），目前没有免费、可提交、又覆盖 GPQA 的公开榜。所以我们用"**公开权重 + 锁定评测配置 + 自建榜单 + 助教复评**"来保证任何人都能复现。

**为什么是 lm-evaluation-harness？** 社区标准评测框架，自带 GPQA 全部变体，配置可以完全锁定。我们用 `gpqa_diamond_cot_zeroshot`，指标是 `exact_match,flexible-extract`（取输出中最后一个 `(A)`–`(D)`）。

---

## 锁定评测配置 / Locked eval config

见 [`scripts/eval_gpqa.sh`](scripts/eval_gpqa.sh)：

- 任务：`gpqa_diamond_cot_zeroshot`、`gpqa_main_cot_zeroshot`
- 后端：vLLM（跑不通时回退 HF transformers）
- **版本锁定**（见 [`requirements-eval.txt`](requirements-eval.txt)）：`vllm==0.11.2`（Unsloth 官方 Colab 专门给 T4 用的版本）、`lm-eval==0.4.11`（能配合这个 vLLM 的最新版；不同 lm-eval 版本的 GPQA 预处理不同，分数不能混用）、`transformers==4.57.6`
- `dtype=float16`（T4 没有 bf16），`max_model_len=8192`，`max_gen_toks=4096`
- `temperature=0.6`，`top_p=0.95`，使用模型自带 chat template（`--apply_chat_template`）
- 4 个 vLLM 采样 seed（0–3）；lm-eval 的 `--seed` 固定为 0，保证所有人的选项顺序一致
- 报告均值 ± 跨 seed 标准误

⚠️ **GPQA 使用条款要求不要在网上公开题目。** `results/` 下的逐题日志包含题目，已加入 `.gitignore`；只提交汇总后的 JSON，不要把模型输出贴到公开的地方。

---

## T4 预算 / Compute budget

| 项目 | 估计 | 状态 |
|---|---|---|
| Qwen3-4B fp16 权重 | ~8 GB；LoRA 训练（seq 4096，梯度检查点）预计 11–14 GB | 待 T4 实测 |
| 4B LoRA 训练吞吐 | ~1–2k tok/s → 4 小时约 10–25M token | 待 T4 实测 |
| Dolci 长思维链 | 先按长度过滤（≤ 4k token），再抽 2k–10k 条 | — |
| GRPO | 每步 2 题 × 4 采样，最长 768 token，几百步 | 待 T4 实测 |
| 评测（vLLM，4 seed） | 估计 20–40 分钟；HF 后端慢 5–10 倍 | 待 T4 实测 |

> "待 T4 实测"的数字会在助教跑完 `00_smoke_test_t4.ipynb` 后更新到这里。

### 已知风险 / Known risks

- **fp16**：T4 没有 bf16，模型以 fp16 运行。Qwen3 在 fp16 下一般稳定，`00` 的 Test 1 会检查输出是否正常（没有乱码或 NaN）。
- **Thinking 模式**：官方 `Qwen3-4B` 的 chat template 默认开 thinking，在 4096 token 预算下长思考会被截断；这是锁定配置的一部分，对所有人一样。
- **Unsloth / vLLM 版本冲突**：Colab 上偶尔会互相覆盖依赖（例如 numpy）。建议评测和训练用不同的 runtime。
- **Colab 断线**：训练 checkpoint 会自动推到你的私有 `<repo>-ckpt` 仓库，可以续训。

---

## 仓库结构 / Layout

```
notebooks/      00 冒烟测试 · 01 评测
scripts/
  eval_gpqa.sh          锁定评测（与 01 相同）
  aggregate_results.py  多 seed 结果 → 提交 JSON（只读汇总指标，不含题目）
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
- Qwen3: [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) · [Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base) · [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
- OLMo 3 / Dolci: [Olmo 3 technical report](https://arxiv.org/abs/2512.13961) · [Dolci-Think-SFT-7B](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B) · [Dolci-Instruct-SFT-7B](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT-7B)
- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Unsloth notebooks（含 Qwen3 4B GRPO）](https://unsloth.ai/docs/get-started/unsloth-notebooks) · [Unsloth 省显存 RL（T4 + Qwen3-4B）](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl) · [TRL](https://github.com/huggingface/trl)
- 课程教材：[RLHF Book](https://rlhfbook.com)（第 4 章 SFT、第 6 章策略梯度 / GRPO、第 8 章 DPO）
