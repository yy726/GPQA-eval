"""Shared helpers for the recipe notebooks (12 DPO, 13 distillation, 14 decision probe).

配方 notebook 共用的代码。题面、答案提取、打分口径和 notebook 11 完全一致，
这样不同配方的快速评测数字可以直接比较。

Only aggregate numbers ever leave these helpers: nothing here prints or logs GPQA questions or
model outputs on GPQA (GPQA terms ask that examples not be revealed online).

Usage in a notebook (after `%cd GPQA-eval`):
    import sys; sys.path.insert(0, "scripts")
    from recipe_utils import *
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import re
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

# Same wording as lm-eval's gpqa_*_cot_zeroshot prompt (and notebook 11).
TEMPLATE = (
    "What is the correct answer to this question:{q}\nChoices:\n"
    "(A) {a}\n(B) {b}\n(C) {c}\n(D) {d}\nLet's think step by step: "
)
STRICT_RE = re.compile(r"The answer is \(([A-D])\)")
LETTER_RE = re.compile(r"\(([A-D])\)")
LETTERS = "ABCD"
MMLU_PRO_STEM = ("physics", "chemistry", "biology", "engineering", "math")

# Plain ChatML (what Qwen3 uses). unsloth/Qwen3-*-Base tokenizers ship without a chat template.
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)
STOP_STRS = ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]


# ---------------------------------------------------------------- prompts / data


def clean(text) -> str:
    """Whitespace + '[title]' cleanup, as in lm-eval's GPQA preprocessing."""
    return " ".join(str(text).replace(" [title]", ". ").split())


def format_mcq(question: str, choices: list[str]) -> str:
    a, b, c, d = choices
    return TEMPLATE.format(q=question, a=a, b=b, c=c, d=d)


def shuffle_choices(correct: str, wrong: list[str], seed: int) -> tuple[list[str], str]:
    """Return the four choices in a seeded random order and the letter of the correct one."""
    choices = list(wrong) + [correct]
    random.Random(seed).shuffle(choices)
    return choices, LETTERS[choices.index(correct)]


def sciq_mcq(ex: dict, idx: int) -> tuple[str, str]:
    """SciQ row -> (GPQA-style prompt, gold letter). Same order as notebook 11 for the same idx."""
    choices, letter = shuffle_choices(ex["correct_answer"], [ex["distractor1"], ex["distractor2"], ex["distractor3"]], idx)
    return format_mcq(ex["question"], choices), letter


def mmlu_pro_choices(ex: dict, idx: int) -> tuple[list[str], str] | None:
    """MMLU-Pro row (up to 10 options) -> 4 options (correct + 3 random distractors) and the gold letter."""
    opts = [o for o in ex["options"] if str(o).strip() and str(o).strip() != "N/A"]
    correct = ex["options"][ex["answer_index"]]
    others = [o for o in opts if o != correct]
    if len(others) < 3:
        return None
    rng = random.Random(idx)
    choices = rng.sample(others, 3) + [correct]
    rng.shuffle(choices)
    return choices, LETTERS[choices.index(correct)]


def mmlu_pro_mcq(ex: dict, idx: int) -> tuple[str, str] | None:
    """MMLU-Pro row -> (GPQA-style prompt, gold letter); same conversion as notebook 11."""
    out = mmlu_pro_choices(ex, idx)
    if out is None:
        return None
    choices, letter = out
    return format_mcq(ex["question"], choices), letter


def gpqa_choices(d: dict, i: int) -> tuple[str, list[str], str]:
    """GPQA row -> (question, shuffled choices, gold letter). Seed 1000+i as in notebook 11."""
    wrong = [clean(d["Incorrect Answer 1"]), clean(d["Incorrect Answer 2"]), clean(d["Incorrect Answer 3"])]
    choices, letter = shuffle_choices(clean(d["Correct Answer"]), wrong, 1000 + i)
    return clean(d["Question"]), choices, letter


def build_gpqa(config: str, n: int | None = None) -> list[tuple[str, str]]:
    """[(prompt, gold letter)] for a GPQA config (gpqa_diamond / gpqa_main). Gated: needs HF login."""
    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", config, split="train")
    rows = []
    for i, d in enumerate(ds):
        q, choices, letter = gpqa_choices(d, i)
        rows.append((format_mcq(q, choices), letter))
    return rows[:n] if n else rows


def load_mcq_items(source: str, n: int, skip: int = 0) -> list[dict]:
    """Public 4-option MCQ items: [{"question", "choices", "answer", "source"}].

    source = "mmlu_pro_stem" (TIGER-Lab/MMLU-Pro test, STEM categories, 4 of up to 10 options) or
    "sciq" (allenai/sciq train). `skip` drops the first rows (e.g. SciQ rows notebook 11 used for SFT/GRPO).
    """
    from datasets import load_dataset

    items: list[dict] = []
    if source == "mmlu_pro_stem":
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test").shuffle(seed=0)
        for idx, ex in enumerate(ds):
            if len(items) >= n:
                break
            if idx < skip or ex["category"] not in MMLU_PRO_STEM:
                continue
            out = mmlu_pro_choices(ex, idx)
            if out:
                items.append({"question": ex["question"], "choices": out[0], "answer": out[1],
                              "source": f"mmlu_pro/{ex['category']}"})
    elif source == "sciq":
        ds = load_dataset("allenai/sciq", split="train").shuffle(seed=0)
        for idx in range(skip, len(ds)):
            if len(items) >= n:
                break
            ex = ds[idx]
            choices, letter = shuffle_choices(ex["correct_answer"], [ex["distractor1"], ex["distractor2"], ex["distractor3"]], idx)
            items.append({"question": ex["question"], "choices": choices, "answer": letter, "source": "sciq"})
    else:
        raise ValueError(f"unknown MCQ source {source!r} (use 'mmlu_pro_stem' or 'sciq')")
    return items


def load_mcq_rows(source: str, n: int, skip: int = 0) -> list[dict]:
    """Same items as load_mcq_items, as GPQA-style prompts: [{"prompt", "answer", "source"}]."""
    return [{"prompt": format_mcq(it["question"], it["choices"]), "answer": it["answer"], "source": it["source"]}
            for it in load_mcq_items(source, n, skip)]


def decision_row(question: str, choices: list[str], letter: str) -> dict:
    """One 4-option MCQ as a FastDecisionModel row (state / questions / gold)."""
    return {
        "state": question,
        "questions": {"answer": {"type": "choice",
                                 "instructions": "Which option correctly answers the science question?",
                                 "criteria": dict(zip(LETTERS, choices))}},
        "gold": {"answer": letter},
    }


def sciq_format_rows(n: int, skip: int = 0) -> list[dict]:
    """SciQ answers in the target format: support paragraph + 'The answer is (X).' (as notebook 11's SFT)."""
    from datasets import load_dataset

    ds = load_dataset("allenai/sciq", split="train").shuffle(seed=0)
    rows = []
    for idx in range(skip, len(ds)):
        if len(rows) >= n:
            break
        ex = ds[idx]
        if not ex["support"].strip():
            continue
        prompt, letter = sciq_mcq(ex, idx)
        rows.append({"messages": [{"role": "user", "content": prompt},
                                  {"role": "assistant", "content": ex["support"].strip() + f"\n\nThe answer is ({letter})."}]})
    return rows


# ---------------------------------------------------------------- scoring


def score_completion(text: str, gold: str, finish_reason: str | None = None, n_tokens: int | None = None) -> dict:
    """Per-completion score. strict = first 'The answer is (X)' (lm-eval strict-match);
    flex = last '(X)' anywhere (lm-eval flexible-extract, the leaderboard metric)."""
    strict = STRICT_RE.findall(text)
    found = LETTER_RE.findall(text)
    return {
        "strict": bool(strict) and strict[0] == gold,
        "flex": bool(found) and found[-1] == gold,
        "extracted": bool(found),
        "truncated": finish_reason == "length",
        "letter": found[-1] if found else "none",
        "n_tokens": n_tokens if n_tokens is not None else len(text.split()),
    }


def aggregate(scores: list[dict]) -> dict:
    n = len(scores)
    if not n:
        return {"n": 0}

    def pct(key):
        return round(100 * sum(bool(s[key]) for s in scores) / n, 1)

    return {
        "n": n,
        "acc_flex": pct("flex"),
        "acc_strict": pct("strict"),
        "extract_rate": pct("extracted"),
        "truncation_rate": pct("truncated"),
        "mean_tokens": round(statistics.fmean(s["n_tokens"] for s in scores)),
        "letters": dict(sorted(Counter(s["letter"] for s in scores).items())),
    }


def score_vllm_outputs(outputs, golds: list[str]) -> dict:
    """Aggregate a list of vLLM RequestOutput against gold letters."""
    scores = []
    for o, gold in zip(outputs, golds):
        c = o.outputs[0]
        scores.append(score_completion(c.text, gold, c.finish_reason, len(c.token_ids)))
    return aggregate(scores)


def is_correct_and_stopped(text: str, gold: str, finish_reason: str | None) -> bool:
    s = score_completion(text, gold, finish_reason)
    return s["flex"] and not s["truncated"]


# ---------------------------------------------------------------- tokenizer / config helpers


def setup_chatml_stop(tokenizer, model=None) -> list[int]:
    """Make a Qwen3 *Base* tokenizer stop after one ChatML turn (runs 1-2 never stopped).

    Adds the ChatML template if missing, sets eos to <|im_end|>, pad to <|endoftext|>, and returns the
    token ids sampling should stop on: <|im_end|>, <|endoftext|>, <|im_start|> (start of a new turn).
    """
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = CHATML_TEMPLATE
    ids = [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>", "<|im_start|>")]
    tokenizer.eos_token = "<|im_end|>"
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token = "<|endoftext|>"
    if model is not None and getattr(model, "generation_config", None) is not None:
        model.generation_config.eos_token_id = ids[:2]
    return ids


def chat_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)


def n_chat_tokens(tokenizer, messages: list[dict], add_generation_prompt: bool = False) -> int:
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt)
    if hasattr(ids, "keys"):  # transformers v5 may return a BatchEncoding
        ids = ids["input_ids"]
    return len(ids)


def config_kwargs(cls, **kwargs) -> dict:
    """Keep only the fields `cls` (a TRL/transformers config dataclass) really has.

    TRL renames and drops arguments between releases (e.g. DPOConfig.rpo_alpha); passing an unknown
    one raises. Dropped names are printed so a silently ignored knob is visible.
    """
    names = {f.name for f in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else set(kwargs)
    kept = {k: v for k, v in kwargs.items() if k in names}
    dropped = sorted(set(kwargs) - set(kept))
    if dropped:
        print(f"[i] {cls.__name__} has no {dropped}; ignored")
    return kept


def wandb_init(project: str | None, run_name: str, config: dict, entity: str | None = None):
    """W&B as in rlhf-book code/policy_gradients/train.py: disabled without a key or project.

    Returns (wandb module, report_to string for TRL configs).
    """
    import wandb

    use = bool(os.environ.get("WANDB_API_KEY")) and bool(project)
    wandb.init(project=project, entity=entity, name=run_name, config=config, mode="online" if use else "disabled")
    print("W&B:", wandb.run.url if use else "disabled (no WANDB_API_KEY)")
    return wandb, ("wandb" if use else "none")


def log_eval(wandb, results: dict, tag: str) -> None:
    """results = {config: aggregate dict}; numbers only go to the W&B run summary."""
    if wandb.run is None:
        return
    for cfg, r in results.items():
        for k, v in r.items():
            if isinstance(v, (int, float)):
                wandb.run.summary[f"gpqa/{cfg}/{tag}/{k}"] = v


def print_table(results: dict) -> list[list]:
    """results = {stage: {config: aggregate}} -> printed table; returns the rows (for wandb.Table)."""
    cols = ["stage", "set", "acc_flex", "acc_strict", "extract_rate", "truncation_rate", "mean_tokens"]
    print(" ".join(f"{c:>15s}" for c in cols))
    rows = []
    for stage, per_set in results.items():
        for cfg, v in per_set.items():
            row = [stage, cfg] + [v.get(c) for c in cols[2:]]
            rows.append(row)
            print(" ".join(f"{str(x):>15s}" for x in row))
    return rows


# ---------------------------------------------------------------- subprocess helpers (data prep, decontam)


def run_script(args: list[str], show_tail: int = 30) -> subprocess.CompletedProcess:
    """Run a repo script with the current Python and print its output (a plain child process's output
    does not show up in a Colab cell otherwise). Raises if the script fails."""
    proc = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)
    out = (proc.stdout + ("\n" + proc.stderr if proc.stderr.strip() else "")).strip().splitlines()
    if out:
        print("\n".join(out[-show_tail:]))
    proc.check_returncode()
    return proc


def decontam_rows(rows: list, to_record, name: str, workdir: str = "data", eval_jsonl: str | None = None):
    """Drop rows that share a 13-gram with GPQA (scripts/decontam.py). Returns (kept rows, n dropped).

    `to_record(row)` gives the JSON record decontam.py reads (messages / prompt / completion / text ...).
    Only counts are printed, never GPQA text.
    """
    os.makedirs(workdir, exist_ok=True)
    path, report = Path(workdir) / f"{name}.jsonl", Path(workdir) / f"{name}_decontam.json"
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(to_record(r), ensure_ascii=False) + "\n")
    args = [SCRIPTS / "decontam.py", "--train-jsonl", path, "--out", report]
    if eval_jsonl:
        args += ["--eval-jsonl", eval_jsonl]
    run_script(args, show_tail=0)
    bad = set(json.loads(report.read_text())["flagged_row_indices"])
    print(f"[decontam] {name}: dropped {len(bad)} of {len(rows)} rows (13-gram overlap with GPQA)")
    return [r for i, r in enumerate(rows) if i not in bad], len(bad)


def prepare_sft_sources(sources: list[dict], tokenizer: str, max_tokens: int, scale: float = 1.0,
                        shuffle_buffer: int = 10_000, workdir: str = "data") -> list[dict]:
    """Sample SFT rows from several HF datasets with scripts/prepare_dolci_sft.py.

    Each source: dict(dataset=..., n=..., include=[...], exclude=[...], config=None, split="train",
    strip_think=False, max_tokens=None). Returns [{"messages", "source", "dataset", "n_tokens"}].
    """
    rows: list[dict] = []
    os.makedirs(workdir, exist_ok=True)
    for i, src in enumerate(sources):
        n = max(1, int(src["n"] * scale))
        out = Path(workdir) / f"src{i}_{src['dataset'].split('/')[-1]}.jsonl"
        args = [SCRIPTS / "prepare_dolci_sft.py", "--dataset", src["dataset"], "--split", src.get("split", "train"),
                "--out", out, "--n", n, "--tokenizer", tokenizer, "--max-tokens", src.get("max_tokens") or max_tokens,
                "--shuffle-buffer", shuffle_buffer]
        if src.get("config"):
            args += ["--config", src["config"]]
        if src.get("include"):
            args += ["--include", *src["include"]]
        if src.get("exclude"):
            args += ["--exclude", *src["exclude"]]
        if src.get("strip_think"):
            args.append("--strip-think")
        print(f"== {src['dataset']} (n={n})")
        run_script(args)
        for line in out.read_text().splitlines():
            r = json.loads(line)
            rows.append({"messages": r["prompt"] + r["completion"], "source": r.get("source"),
                         "dataset": src["dataset"], "n_tokens": r.get("n_tokens")})
    return rows
