#!/usr/bin/env python3
"""Aggregate diagnostics from lm-eval GPQA sample logs (no question text is ever printed).

从 lm-eval 的逐题日志（`samples_*.jsonl`，只存在你自己的 Colab 里）算聚合诊断指标：
  * accuracy            —— 按 strict-match / flexible-extract 两种答案提取方式
  * extract_rate        —— 能从输出里提取出 (A)-(D) 的比例（Base 模型常常低）
  * truncation_rate     —— 输出长度打满生成预算的比例（需要 --tokenizer 才能按 token 算）
  * mean / median 输出长度（字符；有 tokenizer 时也给 token）
  * pred_letter_dist    —— 预测字母分布（检查是否总偏向某个选项）
  * has_think_rate      —— 输出里出现 <think> 的比例

输出只有数字，可以安全地贴到聊天里或提交到仓库；GPQA 条款不允许公开题目本身。

Usage:
    python scripts/analyze_samples.py results/E1 --tokenizer outputs/E1/merged --max-gen-toks 4096
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

TASK_RE = re.compile(r"samples_(gpqa_[a-z]+)_cot_zeroshot_")
INVALID = "[invalid]"
LETTER_RE = re.compile(r"\(([A-D])\)")


def response_text(sample: dict) -> str:
    resps = sample.get("resps") or [[""]]
    first = resps[0]
    return first[0] if isinstance(first, list) else str(first)


def filtered_value(sample: dict) -> str:
    fr = sample.get("filtered_resps") or [INVALID]
    v = fr[0]
    return v[0] if isinstance(v, list) and v else str(v)


def normalize_letter(v: str) -> str | None:
    m = LETTER_RE.search(v or "")
    if m:
        return m.group(1)
    v = (v or "").strip().strip(".").upper()
    return v if v in {"A", "B", "C", "D"} else None


def analyze_file(path: Path, tokenizer=None, max_gen_toks: int = 4096) -> dict:
    by_filter: dict[str, list[dict]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                by_filter[s.get("filter", "none")].append(s)

    out: dict = {}
    lengths_chars: list[int] = []
    lengths_toks: list[int] = []
    think = 0
    seen_docs: set = set()
    for filt, samples in sorted(by_filter.items()):
        n = len(samples)
        correct = sum(float(s.get("exact_match", 0)) for s in samples)
        extracted = [filtered_value(s) for s in samples]
        n_valid = sum(1 for v in extracted if v and v != INVALID)
        letters = Counter(normalize_letter(v) or "none" for v in extracted)
        out[filt] = {
            "n": n,
            "accuracy": round(100 * correct / n, 2) if n else None,
            "extract_rate": round(100 * n_valid / n, 2) if n else None,
            "pred_letter_dist": dict(sorted(letters.items())),
        }
        for s in samples:  # length stats once per doc (identical across filters)
            if s.get("doc_id") in seen_docs:
                continue
            seen_docs.add(s.get("doc_id"))
            text = response_text(s)
            lengths_chars.append(len(text))
            think += "<think>" in text
            if tokenizer is not None:
                lengths_toks.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))

    n_docs = len(seen_docs)
    out["length"] = {
        "n_docs": n_docs,
        "mean_chars": round(statistics.fmean(lengths_chars), 1) if lengths_chars else None,
        "median_chars": statistics.median(lengths_chars) if lengths_chars else None,
        "has_think_rate": round(100 * think / n_docs, 2) if n_docs else None,
    }
    if lengths_toks:
        out["length"].update(
            {
                "mean_tokens": round(statistics.fmean(lengths_toks), 1),
                "median_tokens": statistics.median(lengths_toks),
                "truncation_rate": round(100 * sum(t >= max_gen_toks - 8 for t in lengths_toks) / len(lengths_toks), 2),
            }
        )
    return out


def analyze_dir(root: Path, tokenizer=None, max_gen_toks: int = 4096) -> dict:
    """Return {task: {seed_dir: stats}} for every samples_*.jsonl under root."""
    report: dict = defaultdict(dict)
    files = sorted(root.rglob("samples_*.jsonl"))
    if not files:
        raise SystemExit(f"No samples_*.jsonl under {root} (did you run eval with --log_samples?)")
    for f in files:
        m = TASK_RE.search(f.name)
        task = m.group(1) if m else f.stem
        seed = next((p for p in f.relative_to(root).parts if p.startswith("seed")), "run")
        report[task][seed] = analyze_file(f, tokenizer, max_gen_toks)
    return dict(report)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", type=Path)
    p.add_argument("--tokenizer", default=None, help="HF id or local path; enables token lengths + truncation rate")
    p.add_argument("--max-gen-toks", type=int, default=4096)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
    report = analyze_dir(args.results_dir, tok, args.max_gen_toks)
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
