#!/usr/bin/env python3
"""13-gram decontamination check: training data vs. GPQA.

训练集 vs GPQA 的 13-gram 重叠检查。只输出统计数字和"训练集里被命中的行号"，
绝不打印 GPQA 原题（遵守 GPQA 使用条款）。

Training data can be:
  * a local JSONL file (one example per line; text is collected from `messages[*].content`,
    `prompt`, `completion`, `question`, `answer`, `text` fields), or
  * a Hugging Face dataset id (`--hf-dataset allenai/Dolci-Think-SFT-7B --split train`).

GPQA is loaded from the gated HF dataset `Idavidrein/gpqa` (needs `hf auth login`), or from a
local JSONL with a `Question` field via `--eval-jsonl` (used by the unit tests).

Usage:
    python scripts/decontam.py --train-jsonl data/sft.jsonl
    python scripts/decontam.py --hf-dataset allenai/Dolci-Think-SFT-7B --split train --limit 20000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

N = 13
WORD_RE = re.compile(r"[a-z0-9]+")
TEXT_FIELDS = ("prompt", "completion", "question", "answer", "text", "response", "solution")
GPQA_FIELDS = ("Question", "Correct Answer")


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def ngrams(text: str, n: int = N) -> set[tuple[str, ...]]:
    w = words(text)
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def example_text(ex: dict) -> str:
    parts: list[str] = []
    for key in ("messages", "conversations"):
        msgs = ex.get(key)
        if isinstance(msgs, list):
            for m in msgs:
                if isinstance(m, dict):
                    parts.append(str(m.get("content") or m.get("value") or ""))
    for key in TEXT_FIELDS:
        v = ex.get(key)
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list):  # conversational prompt/completion
            parts.extend(str(m.get("content", "")) for m in v if isinstance(m, dict))
    return "\n".join(parts)


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_hf(dataset: str, split: str, config: str | None) -> Iterator[dict]:
    from datasets import load_dataset  # imported lazily so tests need no extra deps

    yield from load_dataset(dataset, config, split=split, streaming=True)


def load_gpqa_ngrams(eval_jsonl: Path | None) -> tuple[set[tuple[str, ...]], int]:
    rows: Iterable[dict]
    if eval_jsonl is not None:
        rows = list(iter_jsonl(eval_jsonl))
    else:
        from datasets import load_dataset

        # gpqa_extended is a superset of main, which is a superset of diamond.
        rows = load_dataset("Idavidrein/gpqa", "gpqa_extended", split="train")
    grams: set[tuple[str, ...]] = set()
    count = 0
    for r in rows:
        count += 1
        for field in GPQA_FIELDS:
            if r.get(field):
                grams |= ngrams(str(r[field]))
    return grams, count


def check(train_rows: Iterable[dict], gpqa_grams: set[tuple[str, ...]], limit: int | None = None) -> dict:
    n_rows = 0
    flagged: list[int] = []
    for i, ex in enumerate(train_rows):
        if limit is not None and i >= limit:
            break
        n_rows += 1
        if ngrams(example_text(ex)) & gpqa_grams:
            flagged.append(i)
    return {
        "n_gram": N,
        "train_rows_checked": n_rows,
        "train_rows_flagged": len(flagged),
        "overlap_rate": round(len(flagged) / n_rows, 6) if n_rows else 0.0,
        "flagged_row_indices": flagged[:1000],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--train-jsonl", type=Path)
    src.add_argument("--hf-dataset")
    p.add_argument("--split", default="train")
    p.add_argument("--config", default=None)
    p.add_argument("--limit", type=int, default=None, help="Only check the first N training rows")
    p.add_argument("--eval-jsonl", type=Path, default=None, help="Local GPQA-format file (tests only)")
    p.add_argument("--out", type=Path, default=None, help="Write the JSON report here")
    args = p.parse_args(argv)

    gpqa_grams, n_gpqa = load_gpqa_ngrams(args.eval_jsonl)
    rows = iter_jsonl(args.train_jsonl) if args.train_jsonl else iter_hf(args.hf_dataset, args.split, args.config)
    report = check(rows, gpqa_grams, args.limit)
    report["gpqa_questions_indexed"] = n_gpqa
    report["train_source"] = str(args.train_jsonl or f"{args.hf_dataset}:{args.split}")

    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    if report["train_rows_flagged"]:
        print(
            f"\n[!] {report['train_rows_flagged']} training rows share a 13-gram with GPQA. "
            "Drop them (see flagged_row_indices) before training.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
