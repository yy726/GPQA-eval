#!/usr/bin/env python3
"""Build a small, T4-sized SFT set from the OLMo 3 Dolci SFT mixtures.

从 OLMo 3 的 Dolci SFT 数据里，按来源关键词和长度过滤、抽样，输出 T4 能训得完的小数据集。
输出为 TRL 的 conversational prompt-completion 格式（JSONL）：
    {"prompt": [{"role": "user", "content": ...}], "completion": [{"role": "assistant", "content": ...}], "source": ...}
TRL 默认只在 completion 上算 loss。

The Dolci datasets are large (~2M rows), so we stream them and stop once enough rows pass
the filters. Column names are detected at runtime (`messages`/`conversations`, and a source
column among `source`/`dataset_source`/`dataset`), because they can change between releases.

Usage:
    python scripts/prepare_dolci_sft.py --out data/sft.jsonl --n 5000 \
        --tokenizer Qwen/Qwen3.5-4B-Base --max-tokens 4096 \
        --include math science stem --exclude safety
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

DEFAULT_DATASET = "allenai/Dolci-Think-SFT-7B"
SOURCE_KEYS = ("source", "dataset_source", "dataset", "origin")
MSG_KEYS = ("messages", "conversations")


def get_messages(ex: dict) -> list[dict] | None:
    for k in MSG_KEYS:
        msgs = ex.get(k)
        if isinstance(msgs, list) and msgs:
            out = []
            for m in msgs:
                role = m.get("role") or {"human": "user", "gpt": "assistant"}.get(m.get("from", ""), m.get("from"))
                content = m.get("content") if "content" in m else m.get("value")
                if role is None or content is None:
                    return None
                out.append({"role": role, "content": content})
            return out
    return None


def get_source(ex: dict) -> str:
    for k in SOURCE_KEYS:
        if isinstance(ex.get(k), str):
            return ex[k]
    return "unknown"


def to_prompt_completion(msgs: list[dict]) -> dict | None:
    """Use the last assistant turn as the completion; everything before is the prompt."""
    if msgs[-1]["role"] != "assistant":
        return None
    prompt = msgs[:-1]
    if not prompt or not any(m["role"] == "user" for m in prompt):
        return None
    return {"prompt": prompt, "completion": [msgs[-1]]}


def keep_source(source: str, include: list[str], exclude: list[str]) -> bool:
    s = source.lower()
    if any(x.lower() in s for x in exclude):
        return False
    return not include or any(x.lower() in s for x in include)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--split", default="train")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=5000, help="Number of examples to keep")
    p.add_argument("--max-tokens", type=int, default=4096, help="Drop examples longer than this")
    p.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B-Base")
    p.add_argument("--include", nargs="*", default=[], help="Keep only sources containing any of these substrings")
    p.add_argument("--exclude", nargs="*", default=[], help="Drop sources containing any of these substrings")
    p.add_argument("--shuffle-buffer", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-scan", type=int, default=2_000_000, help="Stop after scanning this many rows")
    p.add_argument("--list-sources", type=int, default=0, metavar="K",
                   help="Just scan K rows and print the source distribution, then exit")
    args = p.parse_args(argv)

    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    if args.list_sources:
        c = Counter(get_source(ex) for _, ex in zip(range(args.list_sources), ds))
        for src, k in c.most_common():
            print(f"{k:8d}  {src}")
        return 0

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    rng = random.Random(args.seed)

    kept: list[dict] = []
    stats = Counter()
    for i, ex in enumerate(ds):
        if i >= args.max_scan or len(kept) >= args.n:
            break
        stats["scanned"] += 1
        source = get_source(ex)
        if not keep_source(source, args.include, args.exclude):
            stats["drop_source"] += 1
            continue
        msgs = get_messages(ex)
        if msgs is None:
            stats["drop_format"] += 1
            continue
        pc = to_prompt_completion(msgs)
        if pc is None:
            stats["drop_format"] += 1
            continue
        n_tok = len(tok.apply_chat_template(msgs, tokenize=True))
        if n_tok > args.max_tokens:
            stats["drop_length"] += 1
            continue
        pc["source"] = source
        pc["n_tokens"] = n_tok
        kept.append(pc)

    rng.shuffle(kept)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for ex in kept:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    total_tokens = sum(ex["n_tokens"] for ex in kept)
    by_source = Counter(ex["source"] for ex in kept)
    print(json.dumps({"kept": len(kept), "total_tokens": total_tokens, **stats}, indent=2))
    print("top sources:", by_source.most_common(15))
    if len(kept) < args.n:
        print(f"[!] only {len(kept)} rows passed the filters (asked for {args.n})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
