#!/usr/bin/env python3
"""Pick SFT examples that maximise 4-gram coverage (Exploration-Expanding SFT, Dr.SCI, arXiv 2602.08321).

按 4-gram 覆盖度挑 SFT 数据：每一步选"新增不同 4-gram 最多"的样本（Dr.SCI 的 EESFT，Algorithm 1）。
论文里同样数量的数据，这样挑比随机挑：SFT 后约 +2 分，RL 后约 +3 分（GPQA-Diamond）。

Lazy greedy (the coverage gain only shrinks as the selected set grows, so a stale upper bound that still
beats every other bound is the true maximum). 4-grams are hashed into a fixed-size bit table, so memory is
~256 MB regardless of pool size; the rare hash collision only makes a 4-gram look already covered.

norm:
  none  -> gain = number of new 4-grams (exactly the paper's rule; favours long examples)
  sqrt  -> gain / sqrt(#words)  (default here: our models already write too long, so do not reward length)
  len   -> gain / #words

Usage:
    python scripts/select_coverage.py --in data/pool.jsonl --out data/selected.jsonl --k 20000
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

WORD_RE = re.compile(r"\w+|[^\w\s]")
TABLE_BITS = 28


def example_text(ex: dict) -> str:
    msgs = ex.get("messages") or (ex.get("prompt", []) + ex.get("completion", []))
    return "\n".join(str(m.get("content", "")) for m in msgs if isinstance(m, dict))


def ngram_hashes(text: str, n: int = 4, bits: int = TABLE_BITS) -> tuple[np.ndarray, int]:
    """Unique hashed n-grams of the lower-cased word sequence, and the word count."""
    words = WORD_RE.findall(text.lower())
    if len(words) < n:
        return np.zeros(0, dtype=np.int64), len(words)
    ids = np.array([hash(w) & 0xFFFFFFFF for w in words], dtype=np.uint64)
    h = np.zeros(len(words) - n + 1, dtype=np.uint64)
    for k in range(n):  # polynomial rolling combination of n consecutive word ids
        h = h * np.uint64(1_000_003) + ids[k:len(words) - n + 1 + k]
    h ^= h >> np.uint64(29)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(32)
    return np.unique((h & np.uint64((1 << bits) - 1)).astype(np.int64)), len(words)


def select(texts: list[str], k: int, norm: str = "sqrt", n: int = 4, bits: int = TABLE_BITS) -> list[int]:
    """Indices of the k texts chosen greedily by (normalised) new n-gram coverage, in selection order."""
    grams, scale = [], []
    for t in texts:
        g, nw = ngram_hashes(t, n, bits)
        grams.append(g)
        scale.append(1.0 if norm == "none" else (math.sqrt(max(nw, 1)) if norm == "sqrt" else max(nw, 1)))
    covered = np.zeros(1 << bits, dtype=bool)
    heap = [(-len(g) / s, i) for i, (g, s) in enumerate(zip(grams, scale))]
    heapq.heapify(heap)
    chosen: list[int] = []
    while heap and len(chosen) < k:
        neg_bound, i = heapq.heappop(heap)
        gain = int(np.count_nonzero(~covered[grams[i]])) / scale[i]
        if heap and gain < -heap[0][0] - 1e-12:
            heapq.heappush(heap, (-gain, i))  # stale bound: re-insert with the true gain
            continue
        chosen.append(i)
        covered[grams[i]] = True
    return chosen


def hashed_coverage(texts: list[str], n: int = 4, bits: int = TABLE_BITS) -> int:
    """Approximate number of unique n-grams (hashed; fast enough for tens of thousands of examples)."""
    arrays = [ngram_hashes(t, n, bits)[0] for t in texts]
    return int(np.unique(np.concatenate(arrays)).size) if arrays else 0


def covered_ngrams(texts: list[str], n: int = 4) -> int:
    """Exact number of unique n-grams (for reporting; uses Python sets)."""
    seen: set = set()
    for t in texts:
        w = WORD_RE.findall(t.lower())
        seen.update(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    return len(seen)


def select_per_part(rows: list[dict], targets: dict, method: str = "coverage", norm: str = "sqrt",
                    seed: int = 0) -> tuple[list[dict], dict]:
    """Pick targets[part] rows from each part (row["part"]) of a candidate pool.

    method "coverage": greedy 4-gram coverage inside the part (a part with no more candidates than its
    target is kept whole); "random": a seeded random sample. Parts without a target are dropped.
    Returns (selected rows, {part: {"pool", "target", "selected", "tokens"}}).
    """
    import random

    by_part: dict = {}
    for r in rows:
        by_part.setdefault(r.get("part"), []).append(r)
    out, info = [], {}
    for part, target in targets.items():
        cand = by_part.get(part, [])
        if len(cand) <= target:
            chosen = list(cand)
        elif method == "coverage":
            idx = select([example_text(r) for r in cand], target, norm=norm)
            chosen = [cand[i] for i in idx]
        else:
            chosen = random.Random(seed).sample(cand, target)
        out.extend(chosen)
        info[part] = {"pool": len(cand), "target": target, "selected": len(chosen),
                      "tokens": sum(r.get("n_tokens") or 0 for r in chosen)}
    return out, info


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", type=Path, required=True, help="JSONL with messages or prompt/completion")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--norm", choices=["none", "sqrt", "len"], default="sqrt")
    args = p.parse_args(argv)

    rows = [json.loads(line) for line in args.inp.open() if line.strip()]
    texts = [example_text(r) for r in rows]
    idx = select(texts, args.k, args.norm)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for i in idx:
            f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
    print(json.dumps({"pool": len(rows), "selected": len(idx), "norm": args.norm}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
