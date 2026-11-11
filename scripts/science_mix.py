#!/usr/bin/env python3
"""Build the science SFT mix (short step-by-step solutions) and the held-out MCQ pools from MegaScience.

共享 SFT 检查点（notebook 15）用的科学数据配方，以及留给 GRPO / DPO / 蒸馏的选择题题库。
数据来源 MegaScience/MegaScience（arXiv 2507.16812；CC-BY-NC-SA-4.0），三个子集：
  textbook_reasoning  大学教材里的题目 + 核对过的分步解答（论文里单个数据源中 GPQA 最高，全部保留）
  nemotro_science     NVIDIA OpenScience 的选择题（DeepSeek-V3 重写的短解答；论文按难度筛过）
  natural_reasoning   通用推理题（DeepSeek-V3 分步解答）
MegaScience 对 GPQA 等 13 个基准做过"嵌入检索 + LLM 判是否改写"的去污染，比 13-gram 严格；
notebook 里仍然会再跑一遍 scripts/decontam.py。

Why these choices (see experiments/DATASETS.md for the evidence table):
  * 4-option science MCQs are rewritten into the exact GPQA prompt and end with "The answer is (X).",
    so the model reasons through the options on GPQA instead of reciting a paragraph (X1b's SciQ rows).
  * Only questions whose rewritten solution and reference answer give the same letter are kept (about 1 in 6
    disagree: a wrong gold letter is a wrong SFT label and a wrong RL reward); gold letters are balanced.
  * MCQs are split by a stable hash of the question: one part for SFT, the rest held out as the RL / DPO
    question pool and a validation set (the model must not have seen their solutions).

Usage:
    python scripts/science_mix.py --out-dir data/s1 --tokenizer unsloth/Qwen3-4B-Base \
        --mcq 9000 --textbook physics=20000 chemistry=20000 biology=15000 medicine=5000 math=8000 \
        --natural 8000 --pool-factor 2
    python scripts/science_mix.py --out-dir data/s1 --pools-only      # only the held-out MCQ pools
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recipe_utils import LETTERS, format_mcq  # noqa: E402

DATASET = "MegaScience/MegaScience"
SRC_MCQ, SRC_TEXTBOOK, SRC_NATURAL = "nemotro_science", "textbook_reasoning", "natural_reasoning"  # spelling as in the dataset
SCIENCE_SUBJECTS = ("physics", "chemistry", "biology")

# "A: text" / "A. text" / "A) text" / "(A) text" at the start of a line
_OPTION = re.compile(r"^[ \t]*\(?([A-J])[\):\.][ \t]+(\S.*?)[ \t]*$", re.M)
_BOXED_LETTER = re.compile(r"\\boxed\{\s*(?:\\text(?:bf)?\{)?\s*\(?([A-J])\)?\s*\}?\s*\}")
_ANSWER_IS = re.compile(r"[Tt]he (?:correct |final )?answer is:?\s*\**\s*\(?([A-J])\)?(?![A-Za-z])")
# trailing "final answer" decoration: optional heading, optional \[ or $, the boxed letter, optional closers
_FINAL_BOX = re.compile(
    r"(?:\n[ \t]*(?:#{1,6}[ \t]*)?\**(?:Final Answer|Answer|Conclusion)\**:?\**[ \t]*)?"
    r"\s*(?:\\\[|\\\(|\$\$|\$)?\s*\\boxed\{\s*(?:\\text(?:bf)?\{)?\s*\(?[A-J]\)?\s*\}?\s*\}\s*(?:\\\]|\\\)|\$\$|\$)?[\s\.\*]*$"
)


def parse_mcq(question: str) -> tuple[str, list[str]] | None:
    """Split 'stem + lettered options' into (stem, [option texts]). None unless the options are the last
    thing in the question, one per line, lettered A, B, C, ... without gaps."""
    matches = list(_OPTION.finditer(question))
    if len(matches) < 2:
        return None
    letters = [m.group(1) for m in matches]
    if letters != list("ABCDEFGHIJ"[: len(letters)]):
        return None
    stem = question[: matches[0].start()].strip()
    between = [question[a.end(): b.start()] for a, b in zip(matches, matches[1:])] + [question[matches[-1].end():]]
    if not stem or any(s.strip() for s in between):
        return None  # multi-line options or text after the options: not safe to reformat
    return stem, [m.group(2).strip() for m in matches]


def final_letter(text: str | None) -> str | None:
    """Letter of the last \\boxed{X} (or, failing that, the last 'the answer is (X)')."""
    if not text:
        return None
    found = _BOXED_LETTER.findall(text) or _ANSWER_IS.findall(text)
    return found[-1] if found else None


def gpqa_style_solution(answer: str, letter: str) -> str:
    """Drop the trailing '\\boxed{X}' decoration and end with the sentence the eval extracts."""
    body = _FINAL_BOX.sub("", answer.rstrip()).rstrip()
    return f"{body}\n\nThe answer is ({letter})."


def question_bucket(question: str) -> float:
    """Stable number in [0, 1) from the question text: decides SFT / RL pool / validation membership."""
    h = hashlib.sha1(" ".join(question.lower().split()).encode()).hexdigest()
    return int(h[:8], 16) / 0x100000000


def mcq_row(row: dict, n_options: int = 4, require_agreement: bool = True) -> dict | None:
    """MegaScience MCQ row -> GPQA-format row, or None.

    {"prompt": GPQA prompt, "completion": solution ending with 'The answer is (X).', "answer": X,
     "answer_type": "mcq", "bucket": float, "source": ...}. The option order is kept (the solution refers
    to the options by letter). require_agreement: the solution's final letter must equal the reference's.
    """
    parsed = parse_mcq(row.get("question") or "")
    if parsed is None or len(parsed[1]) != n_options:
        return None
    stem, choices = parsed
    gold = final_letter(row.get("reference_answer"))
    if gold is None or gold not in LETTERS[:n_options]:
        return None
    sol = final_letter(row.get("answer"))
    agrees = sol == gold
    if require_agreement and not agrees:
        return None
    return {"prompt": format_mcq(stem, choices), "completion": gpqa_style_solution(row.get("answer") or "", gold),
            "answer": gold, "answer_type": "mcq", "agrees": agrees, "bucket": question_bucket(stem),
            "source": "megascience/nemotron_science_mcq"}


def balance_letters(rows: list[dict], ratio: float = 1.25, seed: int = 0) -> list[dict]:
    """Downsample so that no gold letter has more than ratio x the rarest letter's count
    (LLM-written MCQs favour some letters; a model trained on them would learn the prior)."""
    by = {}
    for r in rows:
        by.setdefault(r["answer"], []).append(r)
    if len(by) < 2:
        return list(rows)
    cap = int(ratio * min(len(v) for v in by.values()))
    rng = random.Random(seed)
    keep = set()
    for letter in sorted(by):
        group = by[letter]
        chosen = group if len(group) <= cap else rng.sample(group, cap)
        keep.update(id(r) for r in chosen)
    return [r for r in rows if id(r) in keep]


def qa_row(row: dict, part: str) -> dict | None:
    """Open-ended MegaScience row -> {"prompt", "completion", "source", "part"}."""
    q, a = (row.get("question") or "").strip(), (row.get("answer") or "").strip()
    if not q or not a:
        return None
    subject = (row.get("subject") or "").strip()
    return {"prompt": q, "completion": a, "part": part,
            "source": f"megascience/{row.get('source')}" + (f"/{subject}" if subject else "")}


def split_mcq(rows: list[dict], sft_share: float = 0.55, val_share: float = 0.10) -> dict:
    """{"sft", "rl", "val"} by question bucket: [0, sft_share) SFT, then RL pool, the last val_share validation."""
    out = {"sft": [], "rl": [], "val": []}
    for r in rows:
        b = r["bucket"]
        out["sft" if b < sft_share else ("val" if b >= 1.0 - val_share else "rl")].append(r)
    return out


def to_messages(r: dict) -> dict:
    return {"messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["completion"]}],
            "source": r.get("source"), "part": r.get("part"), "n_tokens": r.get("n_tokens")}


def pool_row(r: dict) -> dict:
    """Held-out MCQ as an RL / DPO question (no solution)."""
    return {"prompt": [{"role": "user", "content": r["prompt"]}], "answer": r["answer"], "answer_type": "mcq",
            "source": r["source"]}


def load_mcq_pool(path: str, n: int | None = None, seed: int = 0) -> list[dict]:
    """Read a pool written by this script (rl_mcq.jsonl / val_mcq.jsonl), shuffled with a fixed seed."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    random.Random(seed).shuffle(rows)
    return rows[:n] if n else rows


def parse_counts(items: list[str]) -> dict:
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        out[k.strip()] = int(v)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--tokenizer", default=None, help="Count tokens with this tokenizer (needed for --max-tokens)")
    p.add_argument("--max-tokens", type=int, default=2560, help="Drop rows whose prompt + answer is longer")
    p.add_argument("--mcq", type=int, default=9000, help="4-option science MCQs for SFT (after balancing)")
    p.add_argument("--textbook", nargs="*", default=["physics=20000", "chemistry=20000", "biology=15000",
                                                     "medicine=5000", "math=8000"], metavar="SUBJECT=N")
    p.add_argument("--natural", type=int, default=8000)
    p.add_argument("--pool-factor", type=float, default=2.0,
                   help="Write this many times the requested rows per part (the notebook picks by 4-gram coverage)")
    p.add_argument("--sft-share", type=float, default=0.55, help="Share of MCQ questions SFT may use")
    p.add_argument("--val-share", type=float, default=0.10)
    p.add_argument("--pools-only", action="store_true", help="Only write the held-out MCQ pools (rl / val)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    from datasets import load_dataset

    ds = load_dataset(args.dataset, split="train")
    sources, subjects = ds["source"], ds["subject"]
    rng = random.Random(args.seed)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict = {"dataset": args.dataset, "rows": len(ds), "sources": dict(Counter(sources))}

    # ---- MCQ: every 4-option row -> GPQA format; split by question hash
    mcq_idx = [i for i, s in enumerate(sources) if s == SRC_MCQ]
    mcq_all, n_opts = [], Counter()
    for row in ds.select(mcq_idx):
        parsed = parse_mcq(row["question"] or "")
        n_opts[len(parsed[1]) if parsed else 0] += 1
        r = mcq_row(row, require_agreement=False)
        if r:
            mcq_all.append(r)
    # keep only questions where two independent solutions (reference and rewritten) give the same letter:
    # about 1 in 6 disagree, and a wrong gold letter is a wrong label for SFT and a wrong reward for RL
    parts = split_mcq([r for r in mcq_all if r["agrees"]], args.sft_share, args.val_share)
    stats["mcq"] = {"rows": len(mcq_idx), "by_option_count": dict(sorted(n_opts.items())), "four_option": len(mcq_all),
                    "solution_agrees": sum(r["agrees"] for r in mcq_all),
                    "gold_letters": dict(sorted(Counter(r["answer"] for r in mcq_all).items())),
                    "split": {k: len(v) for k, v in parts.items()}}
    for name in ("rl", "val"):
        rows = balance_letters(parts[name], seed=args.seed)
        rng.shuffle(rows)
        (out_dir / f"{name}_mcq.jsonl").write_text("".join(json.dumps(pool_row(r), ensure_ascii=False) + "\n" for r in rows))
        stats["mcq"][f"{name}_pool"] = len(rows)
    if args.pools_only:
        (out_dir / "mix_stats.json").write_text(json.dumps(stats, indent=1) + "\n")
        print(json.dumps(stats, indent=1))
        return 0

    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def with_tokens(rows: list[dict]) -> list[dict]:
        if tok is None:
            return rows
        kept = []
        for i in range(0, len(rows), 2000):
            chunk = rows[i:i + 2000]
            ids = tok([r["prompt"] + "\n" + r["completion"] for r in chunk], add_special_tokens=False)["input_ids"]
            for r, x in zip(chunk, ids):
                r["n_tokens"] = len(x) + 8
                if r["n_tokens"] <= args.max_tokens:
                    kept.append(r)
        return kept

    pool = []
    # MCQ for SFT: solution must agree with the reference; balanced letters; pool-factor does not apply (scarce)
    sft_mcq = balance_letters(parts["sft"], seed=args.seed)
    rng.shuffle(sft_mcq)
    sft_mcq = with_tokens([dict(r, part="mcq") for r in sft_mcq])[: args.mcq]
    stats["mcq"]["sft_used"] = len(sft_mcq)
    stats["mcq"]["sft_gold_letters"] = dict(sorted(Counter(r["answer"] for r in sft_mcq).items()))
    pool += sft_mcq

    def sample_part(indices: list[int], n: int, part: str) -> list[dict]:
        want = int(n * args.pool_factor)
        take = rng.sample(indices, min(len(indices), int(want * 1.15) + 50))
        rows = [r for r in (qa_row(row, part) for row in ds.select(sorted(take))) if r]
        rng.shuffle(rows)
        return with_tokens(rows)[:want]

    textbook = parse_counts(args.textbook)
    by_subject: dict = {}
    for i, (s, sub) in enumerate(zip(sources, subjects)):
        if s == SRC_TEXTBOOK:
            by_subject.setdefault(sub, []).append(i)
    stats["textbook_subjects"] = {k: len(v) for k, v in sorted(by_subject.items(), key=lambda kv: -len(kv[1]))}
    stats["parts"] = {"mcq": {"requested": args.mcq, "pool": len(sft_mcq)}}
    for sub, n in textbook.items():
        rows = sample_part(by_subject.get(sub, []), n, f"textbook_{sub}")
        stats["parts"][f"textbook_{sub}"] = {"requested": n, "pool": len(rows)}
        pool += rows
    if args.natural > 0:
        rows = sample_part([i for i, s in enumerate(sources) if s == SRC_NATURAL], args.natural, "natural")
        stats["parts"]["natural"] = {"requested": args.natural, "pool": len(rows)}
        pool += rows

    with (out_dir / "sft_pool.jsonl").open("w") as f:
        for r in pool:
            f.write(json.dumps(to_messages(r), ensure_ascii=False) + "\n")
    stats["pool_rows"] = len(pool)
    stats["pool_tokens"] = sum(r.get("n_tokens") or 0 for r in pool)
    (out_dir / "mix_stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
