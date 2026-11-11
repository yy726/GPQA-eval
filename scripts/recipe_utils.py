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
import hashlib
import json
import math
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

# Chat templates for Qwen3 *Base* models (their tokenizers ship without one).
#
# "olmo" (default) follows OLMo 3 / open-instruct: turns are ChatML, but the final assistant turn ends
# with the end-of-document token <|endoftext|> instead of <|im_end|>. Qwen3-4B-Base has (almost) never
# produced <|im_end|>, its output row is undertrained and tied to the input embedding, so LoRA cannot fix
# it: in run X1 the model learned to emit the glitch token "\tNdrFc" (id 70564) where <|im_end|> should be
# and then kept writing. <|endoftext|> is what the base model already emits at the end of a document.
# "plain" follows OLMo's RL-Zero prompts: no special tokens at all, plain-text headers + <|endoftext|>.
# "chatml" is the old template (assistant turns end with <|im_end|>), kept for comparison only.
EOT = "<|endoftext|>"
CHAT_TEMPLATES = {
    "olmo": (
        "{% for message in messages %}"
        "{% if message['role'] == 'assistant' %}"
        "{{ '<|im_start|>assistant\\n' + message['content'] }}"
        "{% if loop.last %}{{ '<|endoftext|>' }}{% else %}{{ '<|im_end|>\\n' }}{% endif %}"
        "{% else %}"
        "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    ),
    "plain": (
        "{% for message in messages %}"
        "{% if message['role'] == 'assistant' %}"
        "{{ '### Response:\\n' + message['content'] }}"
        "{% if loop.last %}{{ '<|endoftext|>' }}{% else %}{{ '\\n\\n' }}{% endif %}"
        "{% else %}"
        "{{ '### Question:\\n' + message['content'] + '\\n\\n' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '### Response:\\n' }}{% endif %}"
    ),
    "chatml": (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    ),
}
CHATML_TEMPLATE = CHAT_TEMPLATES["chatml"]
# (instruction_part, response_part) for unsloth.chat_templates.train_on_responses_only
RESPONSE_PARTS = {
    "olmo": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "chatml": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "plain": ("### Question:\n", "### Response:\n"),
}
GLITCH_TOKEN = "\tNdrFc"   # id 70564 in the Qwen3 vocabulary; what X1 emitted instead of <|im_end|>
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


def question_key(question: str) -> str:
    """Stable id of a question (hash of its normalised text). Used only in memory, to tell which
    gpqa_main questions are also in gpqa_diamond; never printed or logged."""
    return hashlib.sha1(" ".join(clean(question).lower().split()).encode()).hexdigest()[:16]


def build_gpqa_keyed(config: str, n: int | None = None) -> list[tuple[str, str, str]]:
    """[(prompt, gold letter, question key)] for a GPQA config (gpqa_diamond / gpqa_main). Gated: needs HF login."""
    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", config, split="train")
    rows = []
    for i, d in enumerate(ds):
        q, choices, letter = gpqa_choices(d, i)
        rows.append((format_mcq(q, choices), letter, question_key(q)))
    return rows[:n] if n else rows


def build_gpqa(config: str, n: int | None = None) -> list[tuple[str, str]]:
    """[(prompt, gold letter)] for a GPQA config (gpqa_diamond / gpqa_main). Gated: needs HF login."""
    return [(p, g) for p, g, _ in build_gpqa_keyed(config, n)]


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


def _has_repeat(text: str, n: int = 8, times: int = 4) -> bool:
    """Some n-word span occurs at least `times` times (the model is looping)."""
    words = text.split()
    if len(words) < n * times:
        return False
    counts = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return max(counts.values()) >= times


def score_completion(text: str, gold: str, finish_reason: str | None = None, n_tokens: int | None = None) -> dict:
    """Per-completion score. strict = first 'The answer is (X)' (lm-eval strict-match);
    flex = last '(X)' anywhere (lm-eval flexible-extract, the leaderboard metric).
    Diagnostics: repeat (looping), answer_then_more (>=30% of the text after the first
    'The answer is (X)'), glitch (the '\\tNdrFc' token run X1 emitted instead of an end token)."""
    strict = STRICT_RE.findall(text)
    found = LETTER_RE.findall(text)
    m = STRICT_RE.search(text)
    return {
        "strict": bool(strict) and strict[0] == gold,
        "flex": bool(found) and found[-1] == gold,
        "extracted": bool(found),
        "truncated": finish_reason == "length",
        "letter": found[-1] if found else "none",
        "n_tokens": n_tokens if n_tokens is not None else len(text.split()),
        "repeat": _has_repeat(text),
        "answer_then_more": bool(m) and (len(text) - m.end()) >= 0.3 * len(text),
        "glitch": "NdrFc" in text,
    }


def aggregate(scores: list[dict]) -> dict:
    n = len(scores)
    if not n:
        return {"n": 0}

    def pct(key):
        return round(100 * sum(bool(s.get(key)) for s in scores) / n, 1)

    return {
        "n": n,
        "acc_flex": pct("flex"),
        "acc_strict": pct("strict"),
        "extract_rate": pct("extracted"),
        "truncation_rate": pct("truncated"),
        "mean_tokens": round(statistics.fmean(s["n_tokens"] for s in scores)),
        "repeat_rate": pct("repeat"),
        "answer_then_more": pct("answer_then_more"),
        "glitch_rate": pct("glitch"),
        "letters": dict(sorted(Counter(s["letter"] for s in scores).items())),
    }


def score_vllm_outputs(outputs, golds: list[str], detail: bool = False):
    """Aggregate a list of vLLM RequestOutput against gold letters.

    Every sample of every prompt is scored (SamplingParams(n=k) gives k per prompt), so acc_flex is avg@k.
    Extra keys: n = questions, samples = k, acc_flex_se = standard error of acc_flex over questions
    (percentage points). detail=True also returns the per-question fraction correct (kept in memory only).
    """
    scores, per_q = [], []
    for o, gold in zip(outputs, golds):
        s = [score_completion(c.text, gold, c.finish_reason, len(c.token_ids)) for c in o.outputs]
        scores.extend(s)
        per_q.append(sum(x["flex"] for x in s) / max(1, len(s)))
    agg = aggregate(scores)
    agg["n"] = len(per_q)
    agg["samples"] = max((len(o.outputs) for o in outputs), default=0)
    agg["acc_flex_se"] = round(100 * statistics.pstdev(per_q) / math.sqrt(len(per_q)), 1) if len(per_q) > 1 else None
    return (agg, per_q) if detail else agg


def paired_delta(a: list[float], b: list[float]) -> dict:
    """b - a on the same questions (per-question fraction correct): mean difference and its standard error,
    both in percentage points. |z| < 2 means the two stages cannot be told apart."""
    if len(a) != len(b) or len(a) < 2:
        return {}
    d = [y - x for x, y in zip(a, b)]
    se = statistics.stdev(d) / math.sqrt(len(d))
    delta = statistics.fmean(d)
    return {"delta": round(100 * delta, 1), "se": round(100 * se, 1), "z": round(delta / se, 2) if se > 0 else 0.0,
            "n": len(d)}


class QuickEval:
    """GPQA quick eval shared by notebooks 11 / 12 and scripts/quick_eval.py (aggregate numbers only).

    sets = {config: [(prompt text with chat template, gold letter, question key)]}.
    run(tag, generate, samples) calls generate(texts, sampling_kwargs) -> vLLM outputs, where sampling_kwargs
    is a dict for vllm.SamplingParams (n = samples). One sample per question is noisy: on the 198 Diamond
    questions the difference between two stages has a standard error of about 4 points, so use samples >= 4
    and read compare() (paired difference +- SE) instead of comparing two single numbers.
    gpqa_main contains every gpqa_diamond question, so gpqa_main also reports the accuracy on that subset
    (acc_flex_diamond_part) and on the rest (acc_flex_rest).
    """

    def __init__(self, sets: dict, max_tokens: int, samples: int = 1, stop_ids=(), stop=tuple(STOP_STRS),
                 temperature: float = 0.6, top_p: float = 0.95, seed: int = 0):
        self.sets = sets
        self.max_tokens, self.samples = max_tokens, samples
        self.stop_ids, self.stop = list(stop_ids), list(stop)
        self.temperature, self.top_p, self.seed = temperature, top_p, seed
        self.results: dict = {}
        self.per_question: dict = {}
        self.comparisons: dict = {}

    @classmethod
    def from_gpqa(cls, tokenizer, configs, eval_n=None, max_tokens: int = 8192, max_model_len: int | None = None, **kw):
        sets, longest = {}, 0
        for cfg in configs:
            rows = [(chat_prompt(tokenizer, p), g, k) for p, g, k in build_gpqa_keyed(cfg, eval_n)]
            longest = max([longest] + [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t, _, _ in rows])
            sets[cfg] = rows
        if max_model_len is not None and max_tokens > max_model_len - longest - 8:
            max_tokens = max_model_len - longest - 8
            print(f"[!] eval max_tokens clamped to {max_tokens} to fit max_model_len={max_model_len}")
        return cls(sets, max_tokens, **kw)

    def sampling_kwargs(self, samples: int | None = None) -> dict:
        return dict(n=samples or self.samples, temperature=self.temperature, top_p=self.top_p,
                    max_tokens=self.max_tokens, seed=self.seed, stop_token_ids=self.stop_ids, stop=self.stop)

    def run(self, tag: str, generate, samples: int | None = None) -> dict:
        import time

        self.results[tag], self.per_question[tag] = {}, {}
        for cfg, rows in self.sets.items():
            t0 = time.time()
            outs = generate([t for t, _, _ in rows], self.sampling_kwargs(samples))
            r, per_q = score_vllm_outputs(outs, [g for _, g, _ in rows], detail=True)
            if cfg == "gpqa_main" and "gpqa_diamond" in self.sets:
                sub = {k for _, _, k in self.sets["gpqa_diamond"]}
                inside = [x for x, (_, _, k) in zip(per_q, rows) if k in sub]
                rest = [x for x, (_, _, k) in zip(per_q, rows) if k not in sub]
                if inside and rest:
                    r["acc_flex_diamond_part"] = round(100 * statistics.fmean(inside), 1)
                    r["acc_flex_rest"] = round(100 * statistics.fmean(rest), 1)
                    r["n_diamond_part"] = len(inside)
            r["minutes"] = round((time.time() - t0) / 60, 1)
            self.results[tag][cfg] = r
            self.per_question[tag][cfg] = per_q
            print(tag, cfg, r)
        return self.results[tag]

    def compare(self, tag_a: str, tag_b: str) -> dict:
        """Paired difference tag_b - tag_a per config (percentage points)."""
        out = {}
        for cfg in self.sets:
            a, b = self.per_question.get(tag_a, {}).get(cfg), self.per_question.get(tag_b, {}).get(cfg)
            if a and b:
                out[cfg] = paired_delta(a, b)
        self.comparisons[f"{tag_a} -> {tag_b}"] = out
        for cfg, d in out.items():
            verdict = "不显著（|z|<2，看不出差别）" if abs(d.get("z", 0)) < 2 else "显著"
            print(f"{tag_a} -> {tag_b} {cfg}: {d.get('delta'):+.1f} ± {d.get('se')} 分 (z={d.get('z')}, {verdict})")
        return out

    def compare_all(self) -> dict:
        tags = list(self.results)
        for a, b in zip(tags, tags[1:]):
            self.compare(a, b)
        return self.comparisons


# ---------------------------------------------------------------- rewards (GRPO) and answer checking

BOXED_RE = re.compile(r"\\boxed\{")
_NUM = r"[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[-+−]?\.\d+"
_SCI = re.compile(
    rf"^\s*\$?\s*(?P<num>{_NUM})\s*(?:(?:\\times|×|x|\\cdot)\s*10\s*\^\s*\{{?\s*(?P<exp>[-+−]?\d+)\s*\}}?|[eE](?P<eexp>[-+]?\d+))?"
    r"\s*(?P<unit>.*?)\s*\$?\s*$"
)


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (balanced braces), or None."""
    starts = [m.end() for m in BOXED_RE.finditer(text)]
    if not starts:
        return None
    i, depth, out = starts[-1], 1, []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
        out.append(ch)
        i += 1
    return None


def parse_number(s: str | None) -> float | None:
    """Parse '8.85 \\times 10^{-8}', '1.2e3', '865.018 m^3', '-3,200 J' -> float. None if the string is
    not a single number (optionally followed by a unit without digits other than exponents)."""
    if s is None:
        return None
    s = s.strip().replace("−", "-").replace("\\,", "").replace("\\!", "").replace("~", " ")
    s = re.sub(r"\\(?:text|mathrm|operatorname)\{([^}]*)\}", r" \1", s)
    m = _SCI.match(s)
    if not m:
        return None
    unit = re.sub(r"\^\{?-?\d+\}?", "", m.group("unit") or "")
    if re.search(r"\d", unit) or "=" in unit or "\\frac" in unit:
        return None
    exp = m.group("exp") or m.group("eexp")
    num = m.group("num").replace(",", "").replace("−", "-")
    try:
        return float(f"{num}e{int(exp.replace('−', '-'))}" if exp is not None else num)
    except ValueError:
        return None


def numeric_match(pred: float | None, gold: float | None, rel_tol: float = 0.01) -> bool:
    if pred is None or gold is None:
        return False
    if gold == 0:
        return abs(pred) < 1e-6
    return abs(pred - gold) <= rel_tol * abs(gold)


def correctness_score(text: str, answer: str, answer_type: str = "mcq") -> float:
    """+2 correct, 0 wrong, -1 no answer found. mcq: last '(X)'; numeric: last \\boxed{} within 1%."""
    if answer_type == "numeric":
        boxed = last_boxed(text)
        if boxed is None:
            return -1.0
        return 2.0 if numeric_match(parse_number(boxed), parse_number(answer)) else 0.0
    found = LETTER_RE.findall(text)
    if not found:
        return -1.0
    return 2.0 if found[-1] == answer else 0.0


def format_score(text: str, answer_type: str = "mcq", weight: float = 0.5) -> float:
    """`weight` if the answer is stated exactly once in the expected form."""
    if answer_type == "numeric":
        return weight if len(BOXED_RE.findall(text)) == 1 else 0.0
    return weight if len(STRICT_RE.findall(text)) == 1 else 0.0


def soft_overlong_penalty(n_tokens: int, max_len: int, cache: int = 1024) -> float:
    """DAPO's soft overlong punishment: 0 up to max_len - cache, then linear down to -1 at max_len."""
    if n_tokens >= max_len:
        return -1.0
    start = max_len - cache
    if n_tokens <= start:
        return 0.0
    return -(n_tokens - start) / cache


def is_correct(text: str, answer: str, answer_type: str = "mcq") -> bool:
    return correctness_score(text, answer, answer_type) > 0


# ---------------------------------------------------------------- Dr.SCI (RL questions with reference answers)

DRSCI_SUBJECTS = ("physics", "chemistry", "biology")


def drsci_numeric_row(row: dict, subjects=DRSCI_SUBJECTS, difficulty=(0.25, 0.875)) -> dict | None:
    """MiniByte-666/Dr.SCI verifiable row -> GRPO row with a numeric answer, or None.

    difficulty = Qwen3-32B pass rate over 8 tries (larger = easier); keep lo <= d <= hi.
    The row's own prompt already asks for 'The final answer is: $\\boxed{ANSWER}$'.
    """
    info = row.get("extra_info") or {}
    if not info.get("match_rule"):
        return None
    if subjects and info.get("subject") not in subjects:
        return None
    d = info.get("difficulty")
    if d is None or not (difficulty[0] <= d <= difficulty[1]):
        return None
    ref = info.get("reference_answer") or (row.get("reward_model") or {}).get("ground_truth")
    if parse_number(ref) is None:
        return None
    prompt = row.get("prompt") or []
    content = prompt[0]["content"] if prompt and isinstance(prompt[0], dict) else info.get("question")
    if not content:
        return None
    return {"prompt": [{"role": "user", "content": content}], "answer": str(ref), "answer_type": "numeric",
            "difficulty": d, "source": f"drsci/{info.get('subject')}"}


# ---------------------------------------------------------------- Dolci DPO (OLMo 3 preference data)


def dolci_dpo_pair(ex: dict, include=(), exclude=(), types=None) -> dict | None:
    """allenai/Dolci-Instruct-DPO or Dolci-Think-DPO-7B row -> {"prompt","chosen","rejected", ...} or None.

    The source of an Instruct-DPO row is only in `prompt_id` (e.g. 'OpenThoughts3-full-filtered-science-
    no-cot-request-99-738'); Think-DPO has `dataset_source`. `types` filters `preference_type`
    (e.g. ["delta_learning"]: chosen from Qwen3-32B, rejected from Qwen3-0.6B).
    """
    sys.path.insert(0, str(SCRIPTS))
    from prepare_dolci_sft import filter_key, get_messages, keep_source

    if types and ex.get("preference_type") not in types:
        return None
    key = filter_key(ex)
    if not keep_source(key, list(include), list(exclude)):
        return None
    ch, rj = get_messages({"messages": ex.get("chosen")}), get_messages({"messages": ex.get("rejected")})
    if not ch or not rj or ch[-1]["role"] != "assistant" or rj[-1]["role"] != "assistant":
        return None
    prompt = ch[:-1]
    if not any(m["role"] == "user" for m in prompt) and isinstance(ex.get("prompt"), str):
        prompt = [{"role": "user", "content": ex["prompt"]}]
    if not any(m["role"] == "user" for m in prompt):
        return None
    return {"prompt": prompt, "chosen": [ch[-1]], "rejected": [rj[-1]], "source": key,
            "chosen_model": ex.get("chosen_model"), "rejected_model": ex.get("rejected_model")}


def is_correct_and_stopped(text: str, gold: str, finish_reason: str | None, answer_type: str = "mcq") -> bool:
    return is_correct(text, gold, answer_type) and finish_reason != "length"


def pass_stats(outputs, rows: list[dict]) -> dict:
    """Pass rates of sampled answers on RL-style rows ({"answer", "answer_type"}): overall and per answer type.

    correct = right answer and not cut off. hist = number of prompts with c of k samples correct;
    zero_std = share of prompts where all k samples agree (all right or all wrong: no GRPO signal).
    stopped_lengths (token counts of finished answers) is returned for suggest_completion_cap.
    """
    by_type: dict = {}
    hist: Counter = Counter()
    correct_counts, stopped, n_samples, n_stopped = [], [], 0, 0
    for r, o in zip(rows, outputs):
        c = sum(is_correct_and_stopped(x.text, r["answer"], x.finish_reason, r["answer_type"]) for x in o.outputs)
        k = len(o.outputs)
        correct_counts.append(c)
        hist[c] += 1
        t = by_type.setdefault(r["answer_type"], [0, 0])
        t[0] += c
        t[1] += k
        n_samples += k
        for x in o.outputs:
            if x.finish_reason != "length":
                n_stopped += 1
                stopped.append(len(x.token_ids))
    total = sum(v[1] for v in by_type.values())
    k = max((len(o.outputs) for o in outputs), default=0)
    return {
        "prompts": len(correct_counts), "samples": k,
        "pass_rate": round(100 * sum(v[0] for v in by_type.values()) / total, 1) if total else None,
        "pass_by_type": {t: round(100 * v[0] / v[1], 1) for t, v in sorted(by_type.items()) if v[1]},
        "zero_std": round(100 * (hist[0] + hist[k]) / len(correct_counts), 1) if correct_counts and k else None,
        "stop_rate": round(100 * n_stopped / n_samples, 1) if n_samples else None,
        "mean_stopped_tokens": round(statistics.fmean(stopped)) if stopped else None,
        "hist": dict(sorted(hist.items())),
        "correct_counts": correct_counts,
        "stopped_lengths": stopped,
    }


def suggest_completion_cap(stopped_lengths: list[int], preset_cap: int, floor: int = 1024, mult: float = 2.0,
                           q: float = 0.99, step: int = 256) -> int:
    """Completion cap for GRPO rollouts: mult x the q-quantile of the lengths the model stops at by itself,
    rounded up to `step`, within [floor, preset_cap].

    A rollout that never stops costs the whole cap in wall-clock time (one stream is ~60-100 token/s), and with
    100+ rollouts per step there is almost always one. A cap far above what the model ever writes only buys
    waiting time; 2 x p99 still leaves room for answers to grow.
    """
    if not stopped_lengths:
        return preset_cap
    xs = sorted(stopped_lengths)
    p = xs[min(len(xs) - 1, int(q * len(xs)))]
    cap = int(math.ceil(mult * p / step) * step)
    return max(min(floor, preset_cap), min(preset_cap, cap))


def lora_delta(path_a: str, path_b: str) -> dict:
    """How far adapter b is from adapter a (two save_lora directories): ||b - a|| / ||a|| over all LoRA tensors.

    A sanity check that a training stage moved the weights at all, and by how much compared with another
    stage (e.g. GRPO vs the SFT adapter it started from)."""
    import torch
    from safetensors.torch import load_file

    a = load_file(str(Path(path_a) / "adapter_model.safetensors"))
    b = load_file(str(Path(path_b) / "adapter_model.safetensors"))
    num = den = 0.0
    max_abs = 0.0
    for k, va in a.items():
        if k not in b or va.shape != b[k].shape:
            continue
        d = b[k].float() - va.float()
        num += float(torch.sum(d * d))
        den += float(torch.sum(va.float() ** 2))
        max_abs = max(max_abs, float(d.abs().max()))
    return {"rel_change": round(math.sqrt(num / den), 5) if den > 0 else None,
            "abs_change": round(math.sqrt(num), 4), "max_abs_change": round(max_abs, 6), "norm_a": round(math.sqrt(den), 3)}


def dpo_loss_kwargs(cls, loss_type: str = "sigmoid_norm", sft_weight: float | None = None,
                    beta: float | None = None) -> dict:
    """DPOConfig loss settings that work across TRL versions.

    loss_type "sigmoid_norm" = length-normalised DPO (Tulu 3 / OLMo 3 `dpo_norm`): the log-ratio is averaged
    over the answer's tokens, so a longer answer does not win just by being longer. Falls back to "sigmoid"
    if this TRL does not have it. beta None -> 5.0 for sigmoid_norm (OLMo 3), 0.1 for the others.
    sft_weight adds an NLL term on the chosen answer (RPO / IRPO): TRL >= 1.x takes it as a second loss
    ("sft" with loss_weights), older TRL as rpo_alpha.
    """
    names = {f.name for f in dataclasses.fields(cls)}
    if loss_type == "sigmoid_norm" and "sigmoid_norm" not in (cls.__doc__ or ""):
        print("[i] this TRL has no 'sigmoid_norm' DPO loss; using 'sigmoid'")
        loss_type = "sigmoid"
    out: dict = {"loss_type": loss_type, "beta": beta if beta is not None else (5.0 if loss_type == "sigmoid_norm" else 0.1)}
    if sft_weight:
        if "loss_weights" in names:
            out.update(loss_type=[loss_type, "sft"], loss_weights=[1.0, float(sft_weight)])
        elif "rpo_alpha" in names:
            out["rpo_alpha"] = float(sft_weight)
        else:
            print("[i] this TRL has neither loss_weights nor rpo_alpha: no NLL term on the chosen answer")
    return out


# ---------------------------------------------------------------- tokenizer / config helpers


def sft_repo_name(hf_user: str, model_name: str) -> str:
    """HF repo of the shared SFT checkpoint written by notebook 15 (and read by 11 / 12 / 13 when INIT_MODEL="auto"):
    "<user>/qwen3-4b-s1-sft" for unsloth/Qwen3-4B-Base, "<user>/qwen3-0.6b-s1-sft" for the 0.6B quick mode."""
    tag = model_name.split("/")[-1].lower()
    for suffix in ("-unsloth-bnb-4bit", "-bnb-4bit", "-base"):
        tag = tag.replace(suffix, "")
    return f"{hf_user}/{tag}-s1-sft"


def resolve_init_model(init_model, hf_user: str, model_name: str):
    """INIT_MODEL "auto" -> the shared SFT checkpoint repo; anything else (None, a repo id, a local path) unchanged."""
    return sft_repo_name(hf_user, model_name) if init_model == "auto" else init_model


def setup_chat(tokenizer, model=None, style: str = "olmo", force: bool = True) -> list[int]:
    """Install a chat template on a Qwen3 Base tokenizer and return the token ids sampling should stop on.

    style "olmo"/"plain": eos = <|endoftext|> (what the final assistant turn is trained to end with);
    style "chatml": eos = <|im_end|> (old behaviour). generation_config.eos_token_id gets both end tokens
    (as OLMo 3 ships), so vLLM / lm-eval stop on either. Stop ids also include <|im_start|> and the glitch
    token "\tNdrFc" as a safety net. force=False keeps a template the tokenizer already has (e.g. a merged
    model saved by these notebooks).
    """
    if force or not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = CHAT_TEMPLATES[style]
    im_end, eot, im_start = (tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", EOT, "<|im_start|>"))
    tokenizer.eos_token = "<|im_end|>" if style == "chatml" else EOT
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token = "<|vision_pad|>" if "<|vision_pad|>" in tokenizer.get_vocab() else "<|fim_pad|>"
    if model is not None and getattr(model, "generation_config", None) is not None:
        model.generation_config.eos_token_id = [tokenizer.eos_token_id, im_end if style != "chatml" else eot]
    ids = [eot, im_end, im_start]
    glitch = tokenizer.convert_tokens_to_ids("ĉNdrFc")
    if isinstance(glitch, int) and glitch != getattr(tokenizer, "unk_token_id", None) and glitch >= 0:
        ids.append(glitch)
    return ids


def setup_chatml_stop(tokenizer, model=None) -> list[int]:
    """Old behaviour (assistant turns end with <|im_end|>); kept for comparison."""
    return setup_chat(tokenizer, model, style="chatml", force=not getattr(tokenizer, "chat_template", None))


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


TABLE_COLS = ["stage", "set", "acc_flex", "acc_flex_se", "samples", "acc_strict", "extract_rate", "truncation_rate",
              "mean_tokens"]


def print_table(results: dict) -> list[list]:
    """results = {stage: {config: aggregate}} -> printed table; returns the rows (columns = TABLE_COLS)."""
    print(" ".join(f"{c:>15s}" for c in TABLE_COLS))
    rows = []
    for stage, per_set in results.items():
        for cfg, v in per_set.items():
            row = [stage, cfg] + [v.get(c) for c in TABLE_COLS[2:]]
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
    strip_think=False, max_tokens=None, part=None). Returns [{"messages", "source", "dataset", "n_tokens", "part"}].
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
                         "dataset": src["dataset"], "n_tokens": r.get("n_tokens"), "part": src.get("part")})
    return rows
