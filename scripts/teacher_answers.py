#!/usr/bin/env python3
"""Answer-level distillation data: a strong teacher answers held-out GPQA-format science MCQs (vLLM, own
process); only answers whose final letter equals the gold letter are kept.

答案级蒸馏的数据：让一个强老师回答留出的 4 选 1 科学题（题面和 GPQA 一样），只保留**最后选对**的回答，
学生再在这些回答上做 SFT。和 logits 蒸馏（GKD）不同，老师和学生**不需要同一个词表**，所以可以用比 Qwen3 新得多的模型：
  Qwen/Qwen3.8-27B-FP8            GPQA-Diamond 89.2（模型卡，思考模式）；词表 248,320，和 Qwen3 的 151,936 不同
  Qwen/Qwen3.5-35B-A3B-GPTQ-Int4  84.2；MoE，生成快，显存占用小
  Qwen/Qwen3-30B-A3B-Instruct-2507-FP8  70.4（非思考）
只作参考：比赛规则不允许比赛期间用别的模型生成训练数据。题目是 MegaScience 里留出的题（不是 GPQA）。

The process exits non-zero if the teacher cannot be loaded, so a notebook can try the next teacher.

Usage:
    python scripts/teacher_answers.py --teacher Qwen/Qwen3.8-27B-FP8 --pool data/s1/rl_mcq.jsonl \
        --n 3000 --out data/teacher/answers.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recipe_utils import LETTER_RE  # noqa: E402
from science_mix import final_letter, gpqa_style_solution, load_mcq_pool  # noqa: E402

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)
_BARE = re.compile(r"(?:^|\n)\s*\**(?:Final Answer|Answer|Correct (?:answer|option|choice))\**\s*[:：]\s*\**\s*\(?([A-D])\)?\**(?![A-Za-z])", re.I)
_OPTION_WORD = re.compile(r"\b(?:option|choice)\s+\(?([A-D])\)?\s+is\s+(?:the\s+)?correct", re.I)


def strip_think(text: str) -> str:
    """Drop a <think>...</think> block (also an unterminated one: everything up to the last </think>)."""
    text = THINK_RE.sub("", text)
    return text.split("</think>")[-1].strip() if "</think>" in text else text.strip()


def teacher_letter(text: str) -> str | None:
    """The option a free-form answer ends on: the last '(X)', else a boxed letter / 'the answer is X',
    else 'Answer: X', else 'option X is correct'."""
    found = LETTER_RE.findall(text)
    if found:
        return found[-1]
    letter = final_letter(text)
    if letter in list("ABCD"):
        return letter
    for pattern in (_BARE, _OPTION_WORD):
        m = pattern.findall(text)
        if m:
            return m[-1].upper()
    return None


def finalize(text: str, gold: str) -> str | None:
    """Teacher answer -> training completion, or None if it does not end on the gold letter.

    The reasoning is kept as written; only the ending is normalised to 'The answer is (X).', the sentence
    both lm-eval extractors read (a trailing \\boxed{X} is replaced by it)."""
    text = strip_think(text)
    if not text or teacher_letter(text) != gold:
        return None
    ending = f"The answer is ({gold})."
    if text.rstrip().endswith(ending):
        return text.rstrip()
    return gpqa_style_solution(text, gold)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", required=True)
    p.add_argument("--pool", type=Path, required=True, help="rl_mcq.jsonl / val_mcq.jsonl from scripts/science_mix.py")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=3000, help="Number of questions to ask")
    p.add_argument("--skip", type=int, default=0, help="Skip this many questions of the (seed-shuffled) pool first")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--max-model-len", type=int, default=6144)
    p.add_argument("--gpu-util", type=float, default=0.92)
    p.add_argument("--thinking", choices=["off", "on"], default="off",
                   help="off: the teacher writes its reasoning in the answer (what a short-CoT student should imitate)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    rows = load_mcq_pool(str(args.pool), seed=0)[args.skip: args.skip + args.n]
    import torch

    if torch.cuda.get_device_capability()[0] < 8:
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    from vllm import LLM, SamplingParams

    t0 = time.time()
    kwargs = dict(model=args.teacher, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_util, seed=args.seed)
    try:  # text only: do not reserve memory for images / video on a vision-language teacher
        llm = LLM(**kwargs, limit_mm_per_prompt={"image": 0, "video": 0})
    except (TypeError, ValueError):
        llm = LLM(**kwargs)
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=20, max_tokens=args.max_tokens, seed=args.seed)
    messages = [r["prompt"] for r in rows]
    think = args.thinking == "on"
    try:
        outs = llm.chat(messages, sp, chat_template_kwargs={"enable_thinking": think}, use_tqdm=True)
    except TypeError:  # older vLLM: render the prompts ourselves
        tok = llm.get_tokenizer()
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=think) for m in messages]
        outs = llm.generate(prompts, sp, use_tqdm=True)

    kept, answered, correct, stopped, lengths = [], 0, 0, 0, []
    for r, o in zip(rows, outs):
        c = o.outputs[0]
        text = strip_think(c.text)
        letter = teacher_letter(text)
        answered += letter is not None
        correct += letter == r["answer"]
        stopped += c.finish_reason != "length"
        if c.finish_reason == "length":
            continue
        completion = finalize(c.text, r["answer"])
        if completion is None:
            continue
        lengths.append(len(c.token_ids))
        kept.append({"messages": [r["prompt"][0], {"role": "assistant", "content": completion}], "answer": r["answer"],
                     "source": f"teacher/{args.teacher.split('/')[-1]}", "part": "teacher", "n_tokens": len(c.token_ids)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for k in kept:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    n = max(1, len(rows))
    lengths.sort()
    stats = {"teacher": args.teacher, "thinking": args.thinking, "questions": len(rows), "kept": len(kept),
             "teacher_accuracy": round(100 * correct / n, 1), "answered": round(100 * answered / n, 1),
             "stop_rate": round(100 * stopped / n, 1),
             "kept_tokens": {"p50": lengths[len(lengths) // 2], "p90": lengths[int(0.9 * len(lengths))]} if lengths else {},
             "minutes": round((time.time() - t0) / 60, 1)}
    args.out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)  # vLLM can leave worker threads that keep the interpreter alive
