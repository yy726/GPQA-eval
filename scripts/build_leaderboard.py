#!/usr/bin/env python3
"""Validate submissions/*.json and render LEADERBOARD.md.

校验 submissions/*.json 并生成 LEADERBOARD.md。以 `_` 开头的文件（如 `_example.json`）会被跳过。

Usage:
    python scripts/build_leaderboard.py            # validate + rewrite LEADERBOARD.md
    python scripts/build_leaderboard.py --check    # validate only; exit 1 if anything is invalid
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUB_DIR = ROOT / "submissions"
SCHEMA = SUB_DIR / "schema.json"
OUT = ROOT / "LEADERBOARD.md"
ALLOWED_BASES = {"Qwen/Qwen3.5-4B-Base", "Qwen/Qwen3.5-2B-Base"}

HEADER = """# Leaderboard / 排行榜

> 自动生成，请勿手改。由 `scripts/build_leaderboard.py` 从 `submissions/*.json` 生成。
> Auto-generated from `submissions/*.json`; do not edit by hand.

排序依据：GPQA Diamond 多 seed 均值；括号内为跨 seed 标准误。差距 < 3 分视为并列。
✅ = 助教已在 T4 上用锁定配置复评。参考行（Base 起点 / 官方后训练天花板）不参与排名。

| # | Team | Model | Base | GPQA Diamond | GPQA Main | Method | Verified |
|---|------|-------|------|--------------|-----------|--------|----------|
"""


def fmt(score: dict | None) -> str:
    if not score or score.get("mean") is None:
        return "待测 / TBD"
    se = score.get("stderr")
    return f"{score['mean']:.1f}" + (f" (±{se:.1f})" if se is not None else "")


def validate(sub: dict, schema: dict) -> list[str]:
    errors: list[str] = []
    try:
        import jsonschema

        v = jsonschema.Draft202012Validator(schema)
        errors += [f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in v.iter_errors(sub)]
    except ImportError:
        for key in schema.get("required", []):
            if key not in sub:
                errors.append(f"missing required field: {key}")
    if not sub.get("reference") and sub.get("base_model") not in ALLOWED_BASES:
        errors.append(f"base_model must be one of {sorted(ALLOWED_BASES)}")
    return errors


def load(check_only: bool) -> tuple[list[dict], int]:
    schema = json.loads(SCHEMA.read_text())
    subs, n_bad = [], 0
    for path in sorted(SUB_DIR.glob("*.json")):
        if path.name.startswith("_") or path.name == "schema.json":
            continue
        try:
            sub = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            print(f"✗ {path.name}: invalid JSON: {e}")
            n_bad += 1
            continue
        errs = validate(sub, schema)
        if errs:
            n_bad += 1
            print(f"✗ {path.name}")
            for e in errs:
                print(f"    - {e}")
        else:
            print(f"✓ {path.name}")
            sub["_file"] = path.name
            subs.append(sub)
    return subs, n_bad


def diamond_mean(sub: dict) -> float:
    m = sub.get("scores", {}).get("gpqa_diamond", {}).get("mean")
    return m if m is not None else float("-inf")


def render(subs: list[dict]) -> str:
    refs = sorted((s for s in subs if s.get("reference")), key=diamond_mean, reverse=True)
    ranked = sorted((s for s in subs if not s.get("reference")), key=diamond_mean, reverse=True)
    lines = [HEADER.rstrip("\n")]
    for i, s in enumerate(ranked, 1):
        lines.append(row(str(i), s))
    if not ranked:
        lines.append("| | *(no submissions yet / 暂无提交)* | | | | | | |")
    for s in refs:
        lines.append(row("ref", s))
    return "\n".join(lines) + "\n"


def row(rank: str, s: dict) -> str:
    model = s["hf_model_id"]
    link = f"[{model}](https://huggingface.co/{model})"
    sc = s.get("scores", {})
    method = str(s.get("method", "")).replace("|", "/")
    verified = "✅" if s.get("verified") else ""
    return (
        f"| {rank} | {s['team']} | {link} | {s.get('base_model', '')} | "
        f"{fmt(sc.get('gpqa_diamond'))} | {fmt(sc.get('gpqa_main'))} | {method} | {verified} |"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="Validate only; do not write LEADERBOARD.md")
    args = p.parse_args(argv)

    subs, n_bad = load(args.check)
    if n_bad:
        print(f"\n{n_bad} invalid submission(s).")
        return 1
    if not args.check:
        OUT.write_text(render(subs))
        print(f"\nwrote {OUT.relative_to(ROOT)} ({len(subs)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
