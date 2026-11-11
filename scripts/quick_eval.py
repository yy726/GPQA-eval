#!/usr/bin/env python3
"""Quick GPQA check of a saved model in its own process (vLLM), aggregate numbers only.

快速评测（不是排行榜分数）：选项顺序和 notebook 11 一样（与 lm-eval 不同），
只用来比较同一配方训练前后的变化。独立进程运行，结束后显存全部释放，
所以可以在 notebook 里训练完（模型还在显存里）之后直接调用。
每题采样 --samples 次（avg@k）：单样本时 198 题上两次评测之差的标准误约 4 分。

Never prints questions or model outputs. Writes a JSON with {config: metrics} and, under "per_question",
the fraction of samples correct per question index (numbers only; used for paired comparisons).

Usage:
    python scripts/quick_eval.py --model outputs/merged --configs gpqa_diamond gpqa_main \
        --max-tokens 8192 --samples 4 --gpu-util 0.5 --out outputs/eval_after.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recipe_utils import QuickEval, setup_chat  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF id or local path of a merged (non-LoRA) model")
    p.add_argument("--configs", nargs="+", default=["gpqa_diamond"])
    p.add_argument("--n", type=int, default=None, help="Only the first N questions of each config")
    p.add_argument("--max-tokens", type=int, default=8192, help="Locked eval budget is 8192 since 2026-11-11")
    p.add_argument("--max-model-len", type=int, default=12288)
    p.add_argument("--samples", type=int, default=1, help="Samples per question (acc_flex is the average over all)")
    p.add_argument("--chat-style", default="olmo", choices=["olmo", "plain", "chatml"],
                   help="Template to use only if the model's tokenizer has none (base models)")
    p.add_argument("--gpu-util", type=float, default=0.5, help="Fraction of GPU memory vLLM may take")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="eval")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    import torch

    if torch.cuda.get_device_capability()[0] < 8:  # T4: no FlashInfer sampler
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    stop_ids = setup_chat(tok, style=args.chat_style, force=False)
    ev = QuickEval.from_gpqa(tok, args.configs, args.n, max_tokens=args.max_tokens, max_model_len=args.max_model_len,
                             samples=args.samples, stop_ids=stop_ids, seed=args.seed)

    llm = LLM(model=args.model, tokenizer=args.model, dtype="auto", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_util, seed=args.seed, enforce_eager=True)
    results = ev.run(args.tag, lambda texts, kw: llm.generate(texts, SamplingParams(**kw), use_tqdm=False))
    report = {"model": args.model, "tag": args.tag, "max_tokens": ev.max_tokens, "samples": args.samples,
              "results": results, "per_question": ev.per_question[args.tag]}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)  # vLLM can leave worker threads that keep the interpreter alive
