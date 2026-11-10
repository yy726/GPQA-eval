#!/usr/bin/env python3
"""Aggregate lm-eval GPQA results across seeds into a hackathon submission JSON.

把 lm-eval 每个 seed 的输出汇总成提交用的 JSON（均值 ± 标准误）。

The script only reads lm-eval's `results_*.json` files (aggregate metrics). It never reads
or copies the per-sample logs, so the output is safe to publish under the GPQA terms.

Usage:
    python scripts/aggregate_results.py results/my-team --model my-team/qwen3-4b-sft \
        --team my-team --out submissions/my-team.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

TASKS = {
    "gpqa_diamond_cot_zeroshot": "gpqa_diamond",
    "gpqa_main_cot_zeroshot": "gpqa_main",
}
DEFAULT_METRIC = "exact_match,flexible-extract"
FALLBACK_METRIC = "exact_match,strict-match"
SEED_DIR_RE = re.compile(r"seed(\d+)")


def find_result_files(root: Path) -> list[Path]:
    return sorted(root.rglob("results_*.json"))


def seed_of(path: Path, root: Path) -> str:
    for part in path.relative_to(root).parts:
        m = SEED_DIR_RE.fullmatch(part)
        if m:
            return m.group(1)
    return path.stem  # fall back to the timestamped filename


def load_scores(root: Path, metric: str) -> dict[str, dict[str, float]]:
    """Return {benchmark: {seed: accuracy_in_percent}}."""
    scores: dict[str, dict[str, float]] = {name: {} for name in TASKS.values()}
    files = find_result_files(root)
    if not files:
        raise SystemExit(f"No results_*.json found under {root}")
    for f in files:
        data = json.loads(f.read_text())
        results = data.get("results", {})
        seed = seed_of(f, root)
        for task, name in TASKS.items():
            if task not in results:
                continue
            r = results[task]
            value = r.get(metric, r.get(FALLBACK_METRIC))
            if value is None:
                raise SystemExit(f"{f}: task {task} has neither {metric} nor {FALLBACK_METRIC}")
            scores[name][seed] = round(100.0 * float(value), 3)
    return {k: v for k, v in scores.items() if v}


def summarize(per_seed: dict[str, float]) -> dict:
    vals = list(per_seed.values())
    mean = statistics.fmean(vals)
    se = statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else None
    return {
        "mean": round(mean, 2),
        "stderr": round(se, 2) if se is not None else None,
        "per_seed": dict(sorted(per_seed.items(), key=lambda kv: kv[0])),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", type=Path, help="Directory passed as out_dir to eval_gpqa.sh")
    p.add_argument("--model", required=True, help="HF model id that was evaluated")
    p.add_argument("--team", default=None, help="Team name (defaults to the model owner)")
    p.add_argument("--base-model", default="Qwen/Qwen3-4B-Base")
    p.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    p.add_argument("--metric", default=DEFAULT_METRIC)
    p.add_argument("--out", type=Path, default=None, help="Write JSON here (default: stdout)")
    args = p.parse_args(argv)

    scores = load_scores(args.results_dir, args.metric)
    if "gpqa_diamond" not in scores:
        raise SystemExit("gpqa_diamond results missing; the leaderboard requires them")

    sub = {
        "team": args.team or args.model.split("/")[0],
        "hf_model_id": args.model,
        "base_model": args.base_model,
        "method": "TODO: e.g. LoRA SFT on Dolci-Think-SFT (5k samples) + GRPO on SciQ",
        "datasets": ["TODO: e.g. allenai/Dolci-Think-SFT-7B:train"],
        "train_gpu_hours_t4": None,
        "eval": {
            "harness": "lm-evaluation-harness",
            "backend": args.backend,
            "metric": args.metric,
            "config": "scripts/eval_gpqa.sh (locked)",
        },
        "scores": {name: summarize(v) for name, v in scores.items()},
        "decontam": {"max_13gram_overlap_rate": None, "report": "TODO: output of scripts/decontam.py"},
        "training_notebook": "TODO: public link to your Colab/GitHub notebook",
        "submitted_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "verified": False,
    }
    text = json.dumps(sub, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
