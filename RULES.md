# 比赛规则 / Rules

目标：在 GPQA Diamond 上取得尽可能高的分数，同时保证结果**任何人都能复现**。

## 1. 算力 / Compute

- 只允许使用 Google Colab 的 **T4 GPU**（免费版或 Pro 的 T4 都可以）。
- **训练总墙钟时间 ≤ 4 小时**（SFT + DPO + RL 等所有训练阶段加起来；数据预处理和评测不计）。可以分多个 session 完成，断线续训的时间按实际训练时间累计。
- 提交时在 `train_gpu_hours_t4` 字段如实填写。

## 2. 模型 / Model

- 必须从 `Qwen/Qwen3.5-4B-Base`（主赛道）或 `Qwen/Qwen3.5-2B-Base` 出发。
- 4B 全参训练在 T4 上放不下，实际上只能用 LoRA 等参数高效方法；2B 可以自由选择。
- 提交的模型必须是**公开的 HF 仓库**，LoRA 已合并，fp16 权重，能被 vLLM / transformers 直接加载。
- 不允许：模型融合（merge）其他已经后训练过的模型、加载其他模型的权重或 adapter。

## 3. 数据 / Data

- 只能使用**赛前已公开**的数据集。推荐以 AI2 的 Dolci（OLMo 3 后训练数据）为主，可以混入其他公开数据集。
- **禁止**在比赛期间调用任何闭源或开源大模型 API 生成新数据（不允许在线蒸馏）。
- **严禁 GPQA 的任何数据进入训练**（包括 Diamond / Main / Extended 和它们的改写版）。
- 每个训练集都必须跑 `scripts/decontam.py`（13-gram 重叠检查），删除命中的样本，并把结果填进提交 JSON。
- 在 `datasets` 字段列出所有用到的 HF 数据集 ID 和 split。

## 4. 评测（锁定）/ Evaluation (locked)

- 只认 `notebooks/01_eval_gpqa.ipynb` 或 `scripts/eval_gpqa.sh` 产出的分数，**不得修改评测参数**。
- 主指标：`gpqa_diamond_cot_zeroshot` 的 `exact_match,flexible-extract`，4 个 seed 的均值。第二指标：`gpqa_main_cot_zeroshot`。
- 可以修改模型自带的 chat template（例如改格式），但**不得**在模板里加入 few-shot 例子、领域知识或任何与题目相关的内容。助教复评时会检查。
- 模型生成预算固定为 4096 token；思维链太长会被截断，这本身就是要优化的一部分。

## 5. GPQA 使用条款 / GPQA terms

- 每个人都要在 [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) 页面接受条款。
- 条款要求**不要在网上公开题目**。不要把逐题日志、模型对 GPQA 的输出、题目截图上传到公开仓库、群聊截图或幻灯片里。
- 错误分析可以做，但只在自己的 Colab / Drive 里看；展示时用汇总统计（例如分学科的准确率）。

## 6. 提交 / Submission

一次提交 = 一个 PR，往 `submissions/` 里加一个 `<team>.json`（格式见 `submissions/schema.json` 和 `_example.json`），内容包括：

- 公开 HF 模型仓库 ID
- 训练 notebook 的公开链接（Colab 共享链接或 GitHub）
- 方法简述、数据集列表、训练时长、去污染结果
- `01` 生成的分数（`verified` 保持 `false`，由助教改）

每队可以多次提交，以最后一次复评通过的为准。

## 7. 排名与复评 / Ranking & verification

- 按 GPQA Diamond 4-seed 均值排序；差距 < 3 分视为并列，并列时看 GPQA Main。
- 助教对前 N 名在 T4 上用锁定配置复评，**以复评分数为准**。复评与自报分数差距超过 2 个标准误的，需要说明原因。
- 参考行（Base 起点、官方 `Qwen3.5-4B` 天花板）不参与排名。

## 8. 奖项 / Awards

- 🥇 最高 GPQA Diamond（复评）
- 🧪 最佳 RL 方法（GRPO / DPO 等，看相对 SFT 的提升和分析质量）
- 📊 最佳数据配方（数据选择 / 混合比例上最有说服力的消融）
- 📝 最佳分析报告（失败案例、训练曲线、为什么有效 / 无效）
