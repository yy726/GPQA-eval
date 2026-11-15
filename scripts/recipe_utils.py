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
import shutil
import statistics
import subprocess
import sys
import time
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
LEADERBOARD_BUDGET = 8192   # max_gen_toks of the locked eval on main; the reference branch may generate more
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


GPQA_DOMAIN_COLUMN = "High-level domain"   # Physics / Chemistry / Biology


def build_gpqa_keyed(config: str, n: int | None = None, with_domain: bool = False) -> list[tuple]:
    """[(prompt, gold letter, question key)] for a GPQA config (gpqa_diamond / gpqa_main). Gated: needs HF login.
    with_domain=True appends the question's subject ("Physics" / "Chemistry" / "Biology", "unknown" if the
    column is missing) as a fourth field; it is only used to split aggregate scores by subject."""
    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", config, split="train")
    rows = []
    for i, d in enumerate(ds):
        q, choices, letter = gpqa_choices(d, i)
        row = (format_mcq(q, choices), letter, question_key(q))
        rows.append(row + (str(d.get(GPQA_DOMAIN_COLUMN) or "unknown").strip(),) if with_domain else row)
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

    def acc_where(keep):
        part = [s for s in scores if keep(s)]
        return round(100 * sum(bool(s.get("flex")) for s in part) / len(part), 1) if part else None

    return {
        "n": n,
        "acc_flex": pct("flex"),
        "acc_strict": pct("strict"),
        "extract_rate": pct("extracted"),
        # where the points are lost: accuracy among answers that contain a letter (the reasoning itself),
        # among answers that ended by themselves, and among answers cut off at the budget
        "acc_answered": acc_where(lambda s: s.get("extracted")),
        "acc_stopped": acc_where(lambda s: not s.get("truncated")),
        "acc_truncated": acc_where(lambda s: s.get("truncated")),
        "truncation_rate": pct("truncated"),
        "mean_tokens": round(statistics.fmean(s["n_tokens"] for s in scores)),
        "repeat_rate": pct("repeat"),
        "answer_then_more": pct("answer_then_more"),
        "glitch_rate": pct("glitch"),
        "letters": dict(sorted(Counter(s["letter"] for s in scores).items())),
    }


def group_means(per_q: list[float], groups: list[str] | None) -> dict:
    """{group: {"n": questions, "acc_flex": mean per-question accuracy in points}} for per-question fractions."""
    if not groups or len(groups) != len(per_q):
        return {}
    out = {}
    for g in sorted(set(groups)):
        part = [x for x, name in zip(per_q, groups) if name == g]
        out[g] = {"n": len(part), "acc_flex": round(100 * statistics.fmean(part), 1)}
    return out


def score_vllm_outputs(outputs, golds: list[str], detail: bool = False, cap_tokens: int | None = None, decode=None,
                       groups: list[str] | None = None):
    """Aggregate a list of vLLM RequestOutput against gold letters.

    Every sample of every prompt is scored (SamplingParams(n=k) gives k per prompt), so acc_flex is avg@k.
    Extra keys: n = questions, samples = k, acc_flex_se = standard error of acc_flex over questions
    (percentage points). detail=True also returns the per-question fraction correct (kept in memory only).

    cap_tokens + decode(token_ids) -> str: also score every answer as if generation had stopped after
    cap_tokens tokens (the text is cut there). Used when the reference branch evaluates with a larger budget
    than the leaderboard: acc_flex_at_<cap> is what the same answers would score under the leaderboard budget,
    over_<cap>_rate the share of answers that needed more.

    groups: one label per question (GPQA subject). Adds by_domain = {label: {n, acc_flex, extract_rate,
    truncation_rate}}, so it is visible which subject loses points and whether it loses them to wrong
    answers or to answers that never finish.
    """
    scores, per_q, capped, over, per_q_scores = [], [], [], 0, []
    for o, gold in zip(outputs, golds):
        s = [score_completion(c.text, gold, c.finish_reason, len(c.token_ids)) for c in o.outputs]
        scores.extend(s)
        per_q_scores.append(s)
        per_q.append(sum(x["flex"] for x in s) / max(1, len(s)))
        if cap_tokens and decode is not None:
            for c, x in zip(o.outputs, s):
                if len(c.token_ids) > cap_tokens:
                    over += 1
                    capped.append(score_completion(decode(list(c.token_ids[:cap_tokens])), gold, "length", cap_tokens)["flex"])
                else:
                    capped.append(x["flex"])
    agg = aggregate(scores)
    agg["n"] = len(per_q)
    agg["samples"] = max((len(o.outputs) for o in outputs), default=0)
    agg["acc_flex_se"] = round(100 * statistics.pstdev(per_q) / math.sqrt(len(per_q)), 1) if len(per_q) > 1 else None
    if capped:
        agg[f"acc_flex_at_{cap_tokens}"] = round(100 * sum(capped) / len(capped), 1)
        agg[f"over_{cap_tokens}_rate"] = round(100 * over / len(capped), 1)
    by_domain = group_means(per_q, groups)
    for g, d in by_domain.items():
        part = [x for s, name in zip(per_q_scores, groups) if name == g for x in s]
        if part:
            d["extract_rate"] = round(100 * sum(x["extracted"] for x in part) / len(part), 1)
            d["truncation_rate"] = round(100 * sum(x["truncated"] for x in part) / len(part), 1)
    if by_domain:
        agg["by_domain"] = by_domain
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
                 temperature: float = 0.6, top_p: float = 0.95, seed: int = 0, cap_tokens: int | None = None, decode=None,
                 groups: dict | None = None):
        self.sets = sets
        self.groups = groups or {}   # {config: [subject of each question]}: scores are also reported per subject
        # leaderboard budget, reported next to the main score when this eval generates more than that
        self.cap_tokens, self.decode = (cap_tokens, decode) if cap_tokens and cap_tokens < max_tokens else (None, None)
        self.max_tokens, self.samples = max_tokens, samples
        self.stop_ids, self.stop = list(stop_ids), list(stop)
        self.temperature, self.top_p, self.seed = temperature, top_p, seed
        self.results: dict = {}
        self.per_question: dict = {}
        self.comparisons: dict = {}

    @classmethod
    def from_gpqa(cls, tokenizer, configs, eval_n=None, max_tokens: int = 8192, max_model_len: int | None = None,
                  leaderboard_budget: int = LEADERBOARD_BUDGET, **kw):
        sets, groups, longest = {}, {}, 0
        for cfg in configs:
            keyed = build_gpqa_keyed(cfg, eval_n, with_domain=True)
            rows = [(chat_prompt(tokenizer, p), g, k) for p, g, k, _ in keyed]
            longest = max([longest] + [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t, _, _ in rows])
            sets[cfg] = rows
            groups[cfg] = [d for _, _, _, d in keyed]
        kw.setdefault("groups", groups)
        if max_model_len is not None and max_tokens > max_model_len - longest - 8:
            max_tokens = max_model_len - longest - 8
            print(f"[!] eval max_tokens clamped to {max_tokens} to fit max_model_len={max_model_len}")
        if max_tokens > leaderboard_budget:
            kw.update(cap_tokens=leaderboard_budget, decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=True))
            print(f"[i] eval budget {max_tokens} > leaderboard budget {leaderboard_budget}: "
                  f"also reporting acc_flex_at_{leaderboard_budget} (answers cut at the leaderboard budget)")
        return cls(sets, max_tokens, **kw)

    def sampling_kwargs(self, samples: int | None = None) -> dict:
        return dict(n=samples or self.samples, temperature=self.temperature, top_p=self.top_p,
                    max_tokens=self.max_tokens, seed=self.seed, stop_token_ids=self.stop_ids, stop=self.stop)

    def run(self, tag: str, generate, samples: int | None = None, cache=None) -> dict:
        """cache: a StageCache. A finished eval with the same settings is read back instead of generated again
        (the per-question fractions are stored with it, so paired comparisons still work after a resume)."""
        import time

        settings = {"samples": samples or self.samples, "max_tokens": self.max_tokens, "temperature": self.temperature,
                    "n": {cfg: len(rows) for cfg, rows in self.sets.items()}}
        if cache is not None and cache.has(f"eval_{tag}"):
            saved = cache.load(f"eval_{tag}")
            if saved.get("settings") == settings:
                self.results[tag], self.per_question[tag] = saved["results"], saved["per_question"]
                for cfg, r in saved["results"].items():
                    print(tag, cfg, "(cached)", r)
                return self.results[tag]
        self.results[tag], self.per_question[tag] = {}, {}
        for cfg, rows in self.sets.items():
            t0 = time.time()
            outs = generate([t for t, _, _ in rows], self.sampling_kwargs(samples))
            r, per_q = score_vllm_outputs(outs, [g for _, g, _ in rows], detail=True, cap_tokens=self.cap_tokens,
                                          decode=self.decode, groups=self.groups.get(cfg))
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
            print_breakdown(tag, cfg, r)
        if cache is not None:
            cache.save(f"eval_{tag}", {"settings": settings, "results": self.results[tag],
                                       "per_question": self.per_question[tag]})
        return self.results[tag]

    def adopt(self, tag: str, saved: dict, source: str = "") -> bool:
        """Use an eval that was already made (a saved "eval_<tag>" of another run) as stage `tag` here instead
        of generating it again. Accepted only when it covers every question set of this eval with the same
        generation budget and temperature; the number of samples per question may differ (a paired comparison
        uses the per-question fraction correct). Returns False, and says why, when it does not fit."""
        st, res, pq = saved.get("settings") or {}, saved.get("results") or {}, saved.get("per_question") or {}
        why = []
        if st.get("max_tokens") != self.max_tokens:
            why.append(f"max_tokens {st.get('max_tokens')} != {self.max_tokens}")
        if st.get("temperature") != self.temperature:
            why.append(f"temperature {st.get('temperature')} != {self.temperature}")
        for cfg, rows in self.sets.items():
            if cfg not in res or len(pq.get(cfg) or []) != len(rows):
                why.append(f"{cfg}: {len(pq.get(cfg) or [])} questions saved, {len(rows)} here")
        if why:
            print(f"[i] 已有的评测结果不能复用（{'; '.join(why)}）{source}")
            return False
        self.results[tag], self.per_question[tag] = {}, {}
        for cfg in self.sets:
            r = dict(res[cfg])
            r["reused_from"] = source
            if "by_domain" not in r and self.groups.get(cfg):   # an older result: subject split from the per-question data
                r["by_domain"] = group_means(pq[cfg], self.groups[cfg])
            self.results[tag][cfg], self.per_question[tag][cfg] = r, list(pq[cfg])
            print(tag, cfg, "(reused)", r)
            print_breakdown(tag, cfg, r)
        return True

    def compare(self, tag_a: str, tag_b: str) -> dict:
        """Paired difference tag_b - tag_a per config (percentage points)."""
        out = {}
        for cfg in self.sets:
            a, b = self.per_question.get(tag_a, {}).get(cfg), self.per_question.get(tag_b, {}).get(cfg)
            if a and b:
                out[cfg] = paired_delta(a, b)
                groups = self.groups.get(cfg) or []
                if out[cfg] and len(groups) == len(a):
                    out[cfg]["by_domain"] = {
                        g: paired_delta([x for x, n in zip(a, groups) if n == g], [x for x, n in zip(b, groups) if n == g])
                        for g in sorted(set(groups))}
        self.comparisons[f"{tag_a} -> {tag_b}"] = out
        for cfg, d in out.items():
            if not d:
                continue
            verdict = "不显著（|z|<2，看不出差别）" if abs(d.get("z", 0)) < 2 else "显著"
            print(f"{tag_a} -> {tag_b} {cfg}: {d.get('delta'):+.1f} ± {d.get('se')} 分 (z={d.get('z')}, {verdict})")
            parts = [f"{g} {v['delta']:+.1f} ± {v['se']}" for g, v in (d.get("by_domain") or {}).items() if v]
            if parts:
                print("    按学科：" + " | ".join(parts))
        return out

    def compare_all(self) -> dict:
        tags = list(self.results)
        for a, b in zip(tags, tags[1:]):
            self.compare(a, b)
        return self.comparisons


def print_breakdown(tag: str, cfg: str, r: dict) -> None:
    """One readable line per result: where the points are lost (answers without a letter / cut-off answers /
    wrong reasoning) and the score per subject."""
    def pct(v):
        return "没有" if v is None else f"答对 {v}%"

    if r.get("acc_answered") is not None:
        print(f"    {tag} {cfg} | 给出答案的回答：{pct(r['acc_answered'])}（没给出答案的占 "
              f"{round(100 - r.get('extract_rate', 100), 1)}%）| 自己收尾的回答：{pct(r.get('acc_stopped'))} | "
              f"被截断的回答：{pct(r.get('acc_truncated'))}（占 {r.get('truncation_rate')}%）")
    parts = []
    for g, d in (r.get("by_domain") or {}).items():
        extra = f"，无答案 {round(100 - d['extract_rate'], 1)}%，截断 {d['truncation_rate']}%" if "extract_rate" in d else ""
        parts.append(f"{g} {d['acc_flex']}（{d['n']} 题{extra}）")
    if parts:
        print(f"    {tag} {cfg} | 按学科：" + " | ".join(parts))


def find_sft_eval(meta: dict | None, root: str) -> tuple[dict | None, str]:
    """The GPQA eval that the SFT notebook (15) saved for the checkpoint described by `meta` (its handoff.json).

    Notebook 15 keeps its results in <root>/<exp>_<mode>/: stage_sft.json (the training run) and
    stage_eval_1_sft.json (scores and per-question fractions). Returns (saved eval, path), or (None, reason)
    when the folder is not there or does not belong to this checkpoint: the step count and final loss in
    stage_sft.json must equal the ones in handoff.json, and the eval must be newer than the training record.
    """
    if not meta or not meta.get("exp") or not meta.get("mode"):
        return None, "起点检查点没有 handoff.json 记录"
    folder = Path(root) / f"{meta['exp']}_{meta['mode']}"
    tag = "1_think" if meta.get("variant") == "think" else "1_sft"
    sft_file, eval_file = folder / "stage_sft.json", folder / f"stage_eval_{tag}.json"
    if not eval_file.exists() or not sft_file.exists():
        return None, f"{folder} 里没有 SFT notebook 的评测结果"
    try:
        sft, saved = json.loads(sft_file.read_text()), json.loads(eval_file.read_text())
    except Exception as e:
        return None, f"{folder} 里的结果读不出来（{type(e).__name__}）"
    for key in ("steps", "loss_last"):
        if meta.get(key) is not None and sft.get(key) != meta.get(key):
            return None, f"{folder} 是另一次 SFT 训练（{key}: {sft.get(key)} != {meta.get(key)}）"
    if eval_file.stat().st_mtime < sft_file.stat().st_mtime:
        return None, f"{eval_file.name} 比 stage_sft.json 旧：不是这次训练之后做的评测"
    if not saved.get("per_question") or not saved.get("results"):
        return None, f"{eval_file.name} 里没有逐题结果"
    return saved, str(eval_file)


def reuse_sft_eval(ev: "QuickEval", tag: str, meta: dict | None, root: str, keep_samples: bool = False) -> bool:
    """Take the SFT notebook's eval of the starting checkpoint as stage `tag` of this run, so the starting model
    is not evaluated a second time. Later stages are then evaluated with the same number of samples per
    question as that eval (both sides of the comparison equally noisy) unless keep_samples is set."""
    saved, where = find_sft_eval(meta, root)
    if saved is None:
        print("[i] 没有可复用的起点评测：", where)
        return False
    if not ev.adopt(tag, saved, source=where):
        return False
    k = (saved.get("settings") or {}).get("samples")
    if k and not keep_samples and k != ev.samples:
        print(f"[i] 起点评测是每题 {k} 次采样：后面的评测也改成每题 {k} 次（原来 {ev.samples} 次）。想保持原来的次数就填 EVAL_SAMPLES。")
        ev.samples = k
    print(f"[i] 起点模型的 GPQA 分数直接用 notebook 15 测过的结果（{where}），这次不再评测起点。"
          "那次评测用的是合并前的 base + LoRA，这里加载的是合并后的权重，是同一个模型。")
    return True


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


def correctness_score(text: str, answer: str, answer_type: str = "mcq", truncated: bool = False) -> float:
    """+2 correct, 0 wrong, -1 no answer found. mcq: last '(X)'; numeric: last \\boxed{} within 1%.

    truncated=True (the rollout ran into the completion cap) scores like "no answer", whatever the text
    contains. Without this, a rollout that never stops but mentions the right letter somewhere earned
    +2 - 1 (overlong) = +1, more than a finished wrong answer (0): run R1 learned exactly that, and once every
    rollout was truncated there was no signal left to recover from (held-out pass rate 42 -> 0).
    """
    if truncated:
        return -1.0
    if answer_type == "numeric":
        boxed = last_boxed(text)
        if boxed is None:
            return -1.0
        return 2.0 if numeric_match(parse_number(boxed), parse_number(answer)) else 0.0
    found = LETTER_RE.findall(text)
    if not found:
        return -1.0
    return 2.0 if found[-1] == answer else 0.0


def format_score(text: str, answer_type: str = "mcq", weight: float = 0.5, truncated: bool = False) -> float:
    """`weight` if the answer is stated exactly once in the expected form (never for a truncated rollout)."""
    if truncated:
        return 0.0
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


def clip_stop_threshold(base: float, baseline_stop_rate: float | None, margin: float = 0.10) -> float:
    """Clipped-ratio threshold for TruncationGuard: `base`, or the starting model's own share of rollouts that
    do not stop (100 - baseline_stop_rate, in percent) plus `margin` when that is higher."""
    if baseline_stop_rate is None:
        return base
    return round(max(base, (100.0 - baseline_stop_rate) / 100.0 + margin), 3)


class TruncationGuard:
    """Stop a GRPO stage when rollouts keep running into the completion cap.

    update(clipped_ratio) returns True once the ratio has been >= threshold for `patience` logged steps in a
    row. In run R1 the ratio went 0.01 -> 0.16 within six steps of stage 1 and reached 1.0 by step 44; from
    there every step cost 9 minutes and the adapter was unusable. Stopping early costs a few steps instead.
    """

    def __init__(self, threshold: float = 0.15, patience: int = 3):
        self.threshold, self.patience = threshold, patience
        self.streak, self.tripped, self.last = 0, False, None

    def update(self, clipped_ratio) -> bool:
        if clipped_ratio is None:
            return self.tripped
        self.last = float(clipped_ratio)
        self.streak = self.streak + 1 if self.last >= self.threshold else 0
        if self.streak >= self.patience:
            self.tripped = True
        return self.tripped


def truncation_guard_callback(guard: TruncationGuard, key: str = "completions/clipped_ratio", on_step=None):
    """A transformers TrainerCallback that ends training when `guard` trips on the logged clipped ratio.

    on_step(global_step, logs), when given, is called at every log line (the notebook uses it to write the
    progress file); an error in it never stops training.
    """
    from transformers import TrainerCallback

    class _Callback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if on_step is not None and logs:
                try:
                    on_step(state.global_step, logs)
                except Exception as e:
                    print("on_step skipped:", type(e).__name__, e)
            if logs and key in logs and guard.update(logs[key]) and not control.should_training_stop:
                print(f"[!] 截断比例连续 {guard.patience} 步 >= {guard.threshold}（当前 {guard.last:.2f}）：提前结束这一段，"
                      f"在第 {state.global_step} 步停止。")
                control.should_training_stop = True

    return _Callback()


def stage_accept(val: dict, min_stop_rate: float = 90.0, min_z: float = -2.0,
                 baseline_stop_rate: float | None = None, stop_tolerance: float = 5.0) -> tuple[bool, str]:
    """Should a finished GRPO stage be kept? `val` is its held-out result from pass_stats (+ the paired
    comparison "vs_<previous>" that rl_val adds).

    Rejected when the model stopped answering properly (stop_rate below the floor) or when the held-out
    pass rate dropped significantly against the previous stage (paired z below min_z). A rejected stage is
    not merged, evaluated or pushed: the notebook goes back to the previous adapter.

    The floor is min_stop_rate, lowered to baseline_stop_rate - stop_tolerance when the starting model
    itself stops less often than that: a stage must not be thrown away for a stop rate it started with.
    """
    if not val:
        return True, "no held-out validation set"
    floor = min_stop_rate
    if baseline_stop_rate is not None:
        floor = min(min_stop_rate, baseline_stop_rate - stop_tolerance)
    stop = val.get("stop_rate")
    if stop is not None and stop < floor:
        return False, f"stop_rate {stop} < {round(floor, 1)}"
    for key, cmp in val.items():
        if key.startswith("vs_") and isinstance(cmp, dict) and cmp.get("z") is not None and cmp["z"] < min_z:
            return False, f"held-out pass rate dropped {key}: {cmp.get('delta')} points (z={cmp['z']})"
    return True, "ok"


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


HANDOFF_FILE = "handoff.json"


def handoff_repo_name(repo: str, mode: str, finished: bool) -> str:
    """Where notebook 15 pushes: the shared name only for a finished full ("a100") run.

    A quick / medium run goes to "<repo>-quick" / "<repo>-medium", an unfinished full run to "<repo>-partial".
    Before this, every mode pushed to the same repo: a 60-step quick run created "<user>/qwen3-4b-s1-sft", the
    full run was cut off before it pushed, and notebook 11 (INIT_MODEL="auto") trained GRPO on the quick model.
    """
    if mode != "a100":
        return f"{repo}-{mode}"
    return repo if finished else f"{repo}-partial"


def write_handoff_meta(repo: str, meta: dict, token: str | None = None) -> None:
    """Record how a pushed checkpoint was made (mode, steps, finished ...) as handoff.json in the repo / directory."""
    text = json.dumps(meta, indent=1, default=str)
    if os.path.isdir(repo):
        Path(repo, HANDOFF_FILE).write_text(text)
        return
    from huggingface_hub import HfApi

    HfApi(token=token).upload_file(path_or_fileobj=text.encode(), path_in_repo=HANDOFF_FILE, repo_id=repo,
                                   commit_message="handoff record")


def read_handoff_meta(repo: str, token: str | None = None) -> dict | None:
    """handoff.json of a checkpoint (Hub repo id or local directory); None when there is none or it cannot be read."""
    try:
        if os.path.isdir(repo):
            return json.loads(Path(repo, HANDOFF_FILE).read_text())
        from huggingface_hub import hf_hub_download

        return json.loads(Path(hf_hub_download(repo, HANDOFF_FILE, token=token)).read_text())
    except Exception:
        return None


def init_model_problem(meta: dict | None) -> str | None:
    """Why a starting checkpoint should not be trusted for an expensive run, or None when it is a finished full run."""
    if not meta:
        return "没有 handoff.json：无法确认这个检查点是哪一次运行推上去的（旧版 notebook 15 的 quick 试跑和正式训练推到同一个仓库）"
    if meta.get("mode") != "a100":
        return f"这是 MODE={meta.get('mode')!r} 的试跑检查点（{meta.get('steps')} 步），不是正式训练"
    if not meta.get("finished"):
        return f"SFT 没有训完：{meta.get('steps')}/{meta.get('total_steps')} 步"
    return None


def abort_unfinished_requests(model) -> int:
    """Abort whatever the vLLM engine behind model.fast_generate still has queued or running; returns how many.

    vLLM's offline generate() does not abort its requests when it is interrupted (Ctrl-C / the Colab stop
    button): they stay in the engine, and the next generate() call finishes them first and returns their
    outputs in front of its own (outputs are sorted by request id). The caller then pairs answers with the
    wrong prompts. Best effort: the engine internals differ between vLLM versions, so failures are reported,
    not raised - guard_fast_generate also trims leftover outputs.
    """
    try:
        engine = model.vllm_engine.llm_engine
        proc = engine.output_processor
        n = len(proc.request_states)  # one entry per unfinished sample (a prompt with n=8 has up to 8)
        if not n:
            return 0
        external = getattr(proc, "external_req_ids", None)
        if external:  # the ids generate() handed out; aborts every sample of each prompt
            engine.abort_request(list(external), internal=False)
        else:
            engine.abort_request(list(proc.request_states), internal=True)
        getattr(proc, "parent_requests", {}).clear()  # parents of aborted n>1 requests are not removed by vLLM
        return n
    except Exception as e:
        print(f"[!] could not abort leftover vLLM requests ({type(e).__name__}: {e})")
        return 0


def guard_fast_generate(model) -> bool:
    """Wrap model.fast_generate so an interrupted call cannot corrupt the next one (see
    abort_unfinished_requests): leftovers are aborted before every call and when a call is interrupted, and
    if some finish anyway, only the outputs of this call's own prompts (the last ones) are returned.
    Idempotent; returns True when the wrapper was installed by this call."""
    inner = getattr(model, "fast_generate", None)
    if inner is None or getattr(inner, "_stale_guard", False):
        return False

    def fast_generate(*args, **kwargs):
        prompts = args[0] if args else kwargs.get("prompts")
        n_stale = abort_unfinished_requests(model)
        if n_stale:
            print(f"[i] aborted {n_stale} leftover vLLM requests from an interrupted generation")
        try:
            outs = inner(*args, **kwargs)
        except BaseException:
            abort_unfinished_requests(model)
            raise
        n = 1 if isinstance(prompts, str) else (len(prompts) if hasattr(prompts, "__len__") else None)
        if n is not None and isinstance(outs, list) and len(outs) > n:
            print(f"[!] generate returned {len(outs)} outputs for {n} prompts: dropping {len(outs) - n} leftover "
                  "outputs of an interrupted call")
            outs = outs[-n:]
        return outs

    fast_generate._stale_guard = True
    model.fast_generate = fast_generate
    return True


def list_grpo_checkpoints(ckpt_dir: str) -> list[tuple[str, int]]:
    """Adapter checkpoints of a GRPO stage as (path, stage step), newest first.

    Each (re)start of a stage writes to <ckpt_dir>/from_<steps already done>/checkpoint-<steps in this run>,
    so the stage step is the sum of the two numbers. Only checkpoints with adapter weights count. A resume
    walks this list: a checkpoint cut short by a runtime that died mid-write fails to load, and the one
    before it is used instead.
    """
    found = []
    root = Path(ckpt_dir)
    if not root.is_dir():
        return found
    for run in root.glob("from_*"):
        try:
            base = int(run.name.split("_", 1)[1])
        except ValueError:
            continue
        for ck in run.glob("checkpoint-*"):
            try:
                step = base + int(ck.name.split("-", 1)[1])
            except ValueError:
                continue
            if step > 0 and (ck / "adapter_model.safetensors").exists():
                found.append((str(ck), step))
    return sorted(found, key=lambda x: -x[1])


def latest_grpo_checkpoint(ckpt_dir: str) -> tuple[str | None, int]:
    """Newest adapter checkpoint of a GRPO stage and the stage step it belongs to: (path, step) or (None, 0)."""
    found = list_grpo_checkpoints(ckpt_dir)
    return found[0] if found else (None, 0)


def lora_weights_finite(model) -> bool:
    """False if any LoRA weight is NaN / inf (a diverged run must not become the checkpoint to resume from)."""
    import torch

    return all(bool(torch.isfinite(p).all()) for name, p in model.named_parameters() if "lora_" in name)


def save_emergency_checkpoint(model, run_dir: str, global_step: int, check=lora_weights_finite) -> str | None:
    """Save the LoRA weights in memory as <run_dir>/checkpoint-<global_step> when training stops unexpectedly
    (manual interrupt, out of memory, CUDA error), in the layout list_grpo_checkpoints reads, so the next
    run continues from the last optimizer step instead of the last periodic save. Returns the path, or None
    when there is nothing worth saving: no step finished, or the weights are not finite.

    It cannot help when the runtime itself is taken away (Colab disconnect, compute units used up): no code
    runs then, and only the periodic saves remain.
    """
    if global_step <= 0:
        return None
    path = os.path.join(run_dir, f"checkpoint-{global_step}")
    if os.path.exists(os.path.join(path, "adapter_model.safetensors")):
        return path  # the periodic save already wrote this step
    if not check(model):
        print("[!] LoRA weights are not finite: not saved; the last periodic checkpoint stays the newest one")
        return None
    tmp = path + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    (getattr(model, "save_lora", None) or model.save_pretrained)(tmp)
    shutil.rmtree(path, ignore_errors=True)
    os.replace(tmp, path)  # a half-written folder never carries the name a resume looks for
    return path


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
        for g, d in (r.get("by_domain") or {}).items():
            wandb.run.summary[f"gpqa/{cfg}/{tag}/acc_flex_{g}"] = d.get("acc_flex")


TABLE_COLS = ["stage", "set", "acc_flex", "acc_flex_se", "samples", "acc_strict", "extract_rate", "acc_answered",
              "truncation_rate", "mean_tokens", f"acc_flex_at_{LEADERBOARD_BUDGET}"]


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


# ---------------------------------------------------------------- re-runnable, resumable notebooks (Colab)


def reload_scripts(names=("recipe_utils", "science_mix", "select_coverage", "grpo_run")) -> list[str]:
    """After `git pull` in a running notebook: re-import the helper modules that are already loaded, so edits
    to scripts/ take effect without restarting the runtime (and without reloading the model)."""
    import importlib

    done = []
    for name in names:
        if name in sys.modules:
            importlib.reload(sys.modules[name])
            done.append(name)
    return done


def load_guard(ns: dict, key: tuple, var: str = "model") -> bool:
    """Should the model-loading cell load the model now?

    True  -> nothing is loaded yet: load it, then set ns["_LOADED_KEY"] = key.
    False -> the same model is already loaded with the same settings: reuse it. Re-running the cell that
             creates the in-process vLLM engine a second time is what used to fail (the first engine still
             holds the GPU memory), forcing a full runtime restart.
    Raises if a model is loaded with *different* settings: the vLLM engine cannot be rebuilt in the same
    process, so that case needs Runtime -> Restart session; saying so beats an out-of-memory traceback.
    """
    old = ns.get("_LOADED_KEY")
    if old is None or ns.get(var) is None:
        return True
    if tuple(old) == tuple(key):
        return False
    raise RuntimeError(
        f"已经加载了一个模型（{old}），而现在的配置是 {key}。vLLM 引擎不能在同一个进程里换模型 / 换上下文长度："
        "Runtime → Restart session，然后 Run all。只改训练或数据的配置不需要重启。")


def restore_lora(model, path: str | None = None) -> dict:
    """Put the LoRA weights of an already-loaded PEFT model back to a known state, in place.

    path None: fresh initialisation (A random, B zero), i.e. the base model again.
    path: the weights saved by model.save_lora(path) / save_pretrained(path).
    Training cells call this first, so re-running a training cell starts from the same point instead of
    continuing from whatever the previous run left in memory, and nothing has to be reloaded.
    """
    if path:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        res = set_peft_model_state_dict(model, load_file(os.path.join(path, "adapter_model.safetensors")))
        return {"loaded": path, "unexpected_keys": len(getattr(res, "unexpected_keys", None) or [])}
    import torch

    n = 0
    for module in model.modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue
        for name in list(module.lora_A.keys()):
            try:
                module.reset_lora_parameters(name, True)
            except Exception:  # PEFT without that method / signature: do the standard init by hand
                torch.nn.init.kaiming_uniform_(module.lora_A[name].weight, a=math.sqrt(5))
                torch.nn.init.zeros_(module.lora_B[name].weight)
            n += 1
    return {"reset_layers": n}


def persist_dir(name: str, use_drive: bool = True, root: str = "/content/drive/MyDrive/gpqa_ckpt",
                mountpoint: str = "/content/drive") -> str:
    """Directory for everything that must survive a Colab runtime reset (adapters, checkpoints, stage and
    eval results): Google Drive when use_drive, else ./outputs/<name> (lost with the runtime).

    If Drive cannot be mounted this raises instead of quietly falling back to local disk: a multi-hour run
    whose checkpoints vanish with the runtime is exactly what this is meant to prevent. Set use_drive=False
    to run without Drive on purpose. Outside Colab it falls back to ./outputs/<name>.
    """
    path = os.path.join("outputs", name)
    if use_drive:
        try:
            from google.colab import drive
        except Exception:
            print(f"[i] not on Colab: saving to {path}")
            drive = None
        if drive is not None:
            if not os.path.ismount(mountpoint):
                _mount_drive(drive, mountpoint)
            path = os.path.join(root, name)
    os.makedirs(path, exist_ok=True)
    return path


def _mount_drive(drive, mountpoint: str) -> None:
    import time

    # A previous run can leave files in an unmounted /content/drive (e.g. writes after the mount dropped);
    # drive.mount then fails with "Mountpoint must not already contain files". Move them aside and retry.
    if os.path.isdir(mountpoint) and os.listdir(mountpoint):
        aside = f"{mountpoint}_local_{int(time.time())}"
        os.rename(mountpoint, aside)
        print(f"[!] {mountpoint} had local files but was not mounted; moved them to {aside}")
    try:
        drive.mount(mountpoint)
    except Exception as e:
        raise RuntimeError(
            f"Google Drive 挂载失败：{type(e).__name__}: {e}\n"
            "常见原因：授权弹窗被关掉 / 超时，或者授权页上没勾选 Google Drive 权限。重跑这一格，在弹窗里选账号并"
            "允许全部权限。\n不用 Drive（检查点只存在本地，runtime 重置后会丢）：把 USE_DRIVE 设为 False 再重跑。"
        ) from e


def write_progress(directory, phase: str, **info) -> None:
    """Write <directory>/progress.json: what the run is doing right now and when (UTC).

    When a runtime dies (disconnect, compute units used up) the notebook output often stops earlier than the
    run did. This file is the record of how far it really got; the next run prints it. Never raises.
    """
    try:
        path = Path(directory) / "progress.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(phase=phase, utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), **info),
                                  default=str))
        tmp.replace(path)
    except Exception as e:
        print("progress file not written:", type(e).__name__, e)


def read_progress(directory) -> dict | None:
    """The last thing a previous run wrote with write_progress, or None."""
    try:
        return json.loads((Path(directory) / "progress.json").read_text())
    except Exception:
        return None


def init_eval_now(run_init_eval, has_init_model: bool, run_grpo: bool, after_grpo: bool) -> bool:
    """Should the GPQA eval of the starting model run at this point of the notebook?

    run_init_eval: True = before training, "after" = after GRPO training (training progress is on disk before
    the hour of evaluation starts), False = never. With "after" and no GRPO in this run, it runs up front.
    """
    if not has_init_model or not run_init_eval:
        return False
    late = run_init_eval == "after" and run_grpo
    return late if after_grpo else not late


class StageCache:
    """Results of finished stages as small JSON files in the persistent directory.

    A notebook asks `cache.has("grpo_s0")` before an expensive stage and skips it when the result is there, so
    after a disconnect "Run all" continues where it stopped instead of starting over. enabled=False (RESUME off)
    ignores what is stored but still writes new results.
    """

    def __init__(self, directory: str, enabled: bool = True):
        self.dir, self.enabled = Path(directory), enabled
        self.dir.mkdir(parents=True, exist_ok=True)

    def check_config(self, config: dict) -> bool:
        """Stored stage results only make sense for the settings that produced them. If this run's settings
        differ from the ones stored next to the results, stop using the stored results (they stay on disk
        until overwritten) and say so; returns whether resuming is still on."""
        f = self.dir / "run_config.json"
        new = json.loads(json.dumps(config, default=str, sort_keys=True))
        if self.enabled and f.exists():
            old = json.loads(f.read_text())
            if old != new:
                changed = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
                print(f"[!] 配置和这个目录里已有的阶段结果不一致（{', '.join(changed)}）：旧结果不再使用，所有阶段重新跑。"
                      "想保留旧结果请换一个 EXP_NAME。")
                self.enabled = False
        f.write_text(json.dumps(new, sort_keys=True))
        return self.enabled

    def path(self, name: str) -> Path:
        return self.dir / f"stage_{name}.json"

    def has(self, name: str) -> bool:
        return self.enabled and self.path(name).exists()

    def load(self, name: str):
        return json.loads(self.path(name).read_text())

    def save(self, name: str, obj) -> None:
        tmp = self.path(name).with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, default=str))
        tmp.replace(self.path(name))  # a half-written file never counts as a finished stage

    def done(self) -> list[str]:
        return sorted(p.stem[len("stage_"):] for p in self.dir.glob("stage_*.json")) if self.enabled else []


def end_runtime(disconnect: bool = True, wait_seconds: int = 60) -> None:
    """Release the Colab runtime at the end of an unattended run so it stops consuming compute units.
    Everything worth keeping must already be on Drive / the Hub. No-op outside Colab."""
    if not disconnect:
        print("AUTO_DISCONNECT=False：runtime 保持连接（空闲时也在消耗 CU，记得手动断开）。")
        return
    try:
        from google.colab import runtime
    except Exception:
        print("not on Colab: nothing to disconnect")
        return
    import time

    print(f"全部完成。{wait_seconds} 秒后断开并释放 runtime（不想断开就中断这个格）。")
    time.sleep(wait_seconds)
    runtime.unassign()


# ---------------------------------------------------------------- subprocess helpers (data prep, decontam)


def read_jsonl(path) -> list[dict]:
    """Read a JSONL file line by line. Do not use str.splitlines(): it also splits on U+2028 / U+2029 /
    U+0085, which json.dumps(..., ensure_ascii=False) leaves unescaped inside strings, so a row whose
    text contains one gets cut in half (JSONDecodeError: Unterminated string)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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
        for r in read_jsonl(out):
            rows.append({"messages": r["prompt"] + r["completion"], "source": r.get("source"),
                         "dataset": src["dataset"], "n_tokens": r.get("n_tokens"), "part": src.get("part")})
    return rows
