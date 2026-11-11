"""CPU-only tests for scripts/recipe_utils.py (shared by notebooks 12-14). No GPQA data is used."""

from __future__ import annotations

import dataclasses
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import recipe_utils as ru  # noqa: E402

SCIQ_EX = {"question": "What gas do plants absorb?", "distractor1": "oxygen", "distractor2": "helium",
           "distractor3": "neon", "correct_answer": "carbon dioxide", "support": "Plants use CO2."}


def test_sciq_matches_notebook_11_order():
    prompt, letter = ru.sciq_mcq(SCIQ_EX, 7)
    # notebook 11: choices = [d1, d2, d3, correct]; random.Random(idx).shuffle(choices)
    choices = [SCIQ_EX["distractor1"], SCIQ_EX["distractor2"], SCIQ_EX["distractor3"], SCIQ_EX["correct_answer"]]
    random.Random(7).shuffle(choices)
    assert letter == "ABCD"[choices.index("carbon dioxide")]
    assert prompt.startswith("What is the correct answer to this question:What gas")
    assert f"({letter}) carbon dioxide" in prompt
    assert prompt.endswith("Let's think step by step: ")


def test_mmlu_pro_conversion():
    ex = {"question": "2+2?", "options": ["3", "4", "5", "N/A", "6", "7"], "answer_index": 1}
    prompt, letter = ru.mmlu_pro_mcq(ex, 3)
    assert f"({letter}) 4" in prompt
    assert "N/A" not in prompt
    assert prompt.count("\n(") == 4
    assert ru.mmlu_pro_mcq({"question": "q", "options": ["a", "b", "N/A"], "answer_index": 0}, 0) is None


def test_gpqa_choices_shuffle_and_clean():
    d = {"Question": "Q [title]  x", "Correct Answer": "right", "Incorrect Answer 1": "w1",
         "Incorrect Answer 2": "w2", "Incorrect Answer 3": "w3"}
    q, choices, letter = ru.gpqa_choices(d, 5)
    assert q == "Q. x"
    assert choices[ru.LETTERS.index(letter)] == "right"
    expected = ["w1", "w2", "w3", "right"]
    random.Random(1005).shuffle(expected)
    assert choices == expected


def test_score_strict_first_flex_last():
    text = "The answer is (A). Hmm, actually (C)."
    s = ru.score_completion(text, "C", "stop", 10)
    assert s["flex"] and not s["strict"] and s["letter"] == "C" and not s["truncated"]
    s = ru.score_completion("The answer is (B).", "B", "length", 5)
    assert s["strict"] and s["flex"] and s["truncated"]
    s = ru.score_completion("no letter", "A")
    assert not s["extracted"] and s["letter"] == "none"
    assert ru.is_correct_and_stopped("The answer is (D).", "D", "stop")
    assert not ru.is_correct_and_stopped("The answer is (D).", "D", "length")


def test_aggregate():
    scores = [ru.score_completion("The answer is (A).", "A", "stop", 4),
              ru.score_completion("(B)", "A", "length", 8)]
    r = ru.aggregate(scores)
    assert r == {"n": 2, "acc_flex": 50.0, "acc_strict": 50.0, "extract_rate": 100.0,
                 "truncation_rate": 50.0, "mean_tokens": 6, "repeat_rate": 0.0, "answer_then_more": 0.0,
                 "glitch_rate": 0.0, "letters": {"A": 1, "B": 1}}
    assert ru.aggregate([]) == {"n": 0}


def test_config_kwargs_filters_unknown_fields(capsys):
    @dataclasses.dataclass
    class FakeConfig:
        beta: float = 0.1
        learning_rate: float = 1e-5

    kept = ru.config_kwargs(FakeConfig, beta=0.2, rpo_alpha=1.0)
    assert kept == {"beta": 0.2}
    assert "rpo_alpha" in capsys.readouterr().out


class FakeTok:
    """Just enough of a HF tokenizer for setup_chat."""
    vocab = {"<|im_end|>": 1, "<|endoftext|>": 2, "<|im_start|>": 3, "<|vision_pad|>": 4, "ĉNdrFc": 70564}

    def __init__(self):
        self.chat_template = None
        self.pad_token = None
        self.eos_token = "<|endoftext|>"
        self.unk_token_id = None

    @property
    def eos_token_id(self):
        return self.vocab[self.eos_token]

    def get_vocab(self):
        return dict(self.vocab)

    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t)


class FakeModel:
    class generation_config:
        eos_token_id = None


def test_setup_chat_olmo_style_ends_with_endoftext():
    tok, model = FakeTok(), FakeModel()
    ids = ru.setup_chat(tok, model, style="olmo")
    assert ids == [2, 1, 3, 70564]                 # <|endoftext|>, <|im_end|>, <|im_start|>, glitch
    assert tok.chat_template == ru.CHAT_TEMPLATES["olmo"]
    assert tok.eos_token == "<|endoftext|>" and tok.pad_token == "<|vision_pad|>"
    assert model.generation_config.eos_token_id == [2, 1]


def test_setup_chatml_stop_keeps_old_behaviour():
    tok = FakeTok()
    ids = ru.setup_chatml_stop(tok)
    assert tok.chat_template == ru.CHATML_TEMPLATE and tok.eos_token == "<|im_end|>"
    assert set(ids[:3]) == {1, 2, 3}


def _render(style, messages, add_generation_prompt=False):
    import jinja2
    return jinja2.Template(ru.CHAT_TEMPLATES[style]).render(messages=messages, add_generation_prompt=add_generation_prompt)


def test_olmo_template_renders_like_open_instruct():
    one = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    assert _render("olmo", one) == "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\nA<|endoftext|>"
    multi = one + [{"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}]
    assert _render("olmo", multi) == ("<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\nA<|im_end|>\n"
                                      "<|im_start|>user\nQ2<|im_end|>\n<|im_start|>assistant\nA2<|endoftext|>")
    assert _render("olmo", one[:1], True) == "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n"


def test_plain_template_has_no_special_tokens_except_final_eos():
    out = _render("plain", [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}])
    assert out == "### Question:\nQ\n\n### Response:\nA<|endoftext|>"
    assert _render("plain", [{"role": "user", "content": "Q"}], True) == "### Question:\nQ\n\n### Response:\n"
    for style in ru.CHAT_TEMPLATES:
        instr, resp = ru.RESPONSE_PARTS[style]
        rendered = _render(style, [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}])
        assert instr in rendered and resp in rendered


def test_diagnostic_metrics():
    looping = "The sequence is determined by the sequence of nucleotides in the parent molecule. " * 6
    s = ru.score_completion(looping, "A", "length")
    assert s["repeat"] and s["truncated"]
    s = ru.score_completion("Reasoning... The answer is (B).\tNdrFc\n\nWhat is the correct answer to this question: "
                            "blah blah blah blah blah blah", "B", "stop")
    assert s["glitch"] and s["answer_then_more"] and s["strict"]
    s = ru.score_completion("Short reasoning. The answer is (C).", "C", "stop")
    assert not s["repeat"] and not s["answer_then_more"] and not s["glitch"]
    agg = ru.aggregate([s])
    assert agg["repeat_rate"] == 0.0 and agg["glitch_rate"] == 0.0 and agg["answer_then_more"] == 0.0


def test_parse_number_and_numeric_match():
    assert ru.parse_number("8.85 \\times 10^{-8}") == 8.85e-8
    assert ru.parse_number("1.2e3") == 1200.0
    assert ru.parse_number("865.018 m^3") == 865.018
    assert ru.parse_number("-3,200 \\text{ J}") == -3200.0
    assert ru.parse_number("0") == 0.0
    assert ru.parse_number("1 \\text{ in } 10,000") is None
    assert ru.parse_number("\\frac{1}{2}") is None
    assert ru.parse_number("\\mu (3 - 2 \\ln 2)") is None
    assert ru.numeric_match(0.0362, 0.0359) and not ru.numeric_match(0.0400, 0.0359)
    assert ru.numeric_match(0.0, 0.0)


def test_rewards():
    assert ru.correctness_score("so \\boxed{8.8 \\times 10^{-8}} V/m", "8.85 \\times 10^{-8}", "numeric") == 2.0
    assert ru.correctness_score("\\boxed{12}", "8.85", "numeric") == 0.0
    assert ru.correctness_score("no box", "8.85", "numeric") == -1.0
    assert ru.last_boxed("a \\boxed{\\frac{1}{2}} b \\boxed{3.0}") == "3.0"
    assert ru.correctness_score("The answer is (D).", "D") == 2.0
    assert ru.correctness_score("nothing", "D") == -1.0
    assert ru.format_score("The answer is (D).", "mcq") == 0.5
    assert ru.format_score("\\boxed{1} and \\boxed{2}", "numeric") == 0.0
    assert ru.is_correct_and_stopped("\\boxed{5}", "5", "stop", "numeric")
    assert not ru.is_correct_and_stopped("\\boxed{5}", "5", "length", "numeric")


def test_soft_overlong_penalty():
    assert ru.soft_overlong_penalty(5000, 6144) == 0.0
    assert ru.soft_overlong_penalty(5120, 6144) == 0.0
    assert ru.soft_overlong_penalty(5632, 6144) == -0.5
    assert ru.soft_overlong_penalty(6144, 6144) == -1.0


def test_drsci_numeric_row():
    row = {"prompt": [{"role": "user", "content": "Solve ... \\boxed{ANSWER}\n\nQ"}],
           "reward_model": {"ground_truth": "1.0"},
           "extra_info": {"match_rule": True, "subject": "physics", "difficulty": 0.5, "reference_answer": "1.0",
                          "question": "Q"}}
    out = ru.drsci_numeric_row(row)
    assert out["answer_type"] == "numeric" and out["answer"] == "1.0" and out["prompt"][0]["content"].endswith("Q")
    assert ru.drsci_numeric_row({**row, "extra_info": {**row["extra_info"], "difficulty": 0.0}}) is None
    assert ru.drsci_numeric_row({**row, "extra_info": {**row["extra_info"], "subject": "economics"}}) is None
    assert ru.drsci_numeric_row({**row, "extra_info": {**row["extra_info"], "reference_answer": "x^2+1"}}) is None
    assert ru.drsci_numeric_row({**row, "extra_info": {**row["extra_info"], "match_rule": False}}) is None


def test_dolci_dpo_pair_uses_prompt_id_and_type():
    ex = {"chosen": [{"role": "user", "content": "Why?"}, {"role": "assistant", "content": "Because."}],
          "rejected": [{"role": "user", "content": "Why?"}, {"role": "assistant", "content": "Dunno."}],
          "chosen_model": "qwen3-no_reasoning-32b", "rejected_model": "qwen3-no_reasoning-0.6b",
          "prompt_id": "OpenThoughts3-full-filtered-science-no-cot-request-99-738", "preference_type": "delta_learning"}
    pair = ru.dolci_dpo_pair(ex, include=["science"], types=["delta_learning"])
    assert pair["prompt"] == [{"role": "user", "content": "Why?"}]
    assert pair["chosen"][0]["content"] == "Because." and pair["rejected"][0]["content"] == "Dunno."
    assert ru.dolci_dpo_pair(ex, include=["math"]) is None
    assert ru.dolci_dpo_pair(ex, types=["llm_judged"]) is None


def test_chatml_template_renders():
    import jinja2

    out = jinja2.Template(ru.CHATML_TEMPLATE).render(
        messages=[{"role": "user", "content": "hi"}], add_generation_prompt=True)
    assert out == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"


def test_decision_row():
    row = ru.decision_row("Which is a noble gas?", ["neon", "iron", "carbon", "sodium"], "A")
    q = row["questions"]["answer"]
    assert row["state"] == "Which is a noble gas?" and q["type"] == "choice"
    assert q["criteria"] == {"A": "neon", "B": "iron", "C": "carbon", "D": "sodium"}
    assert row["gold"] == {"answer": "A"}


def test_decontam_rows_uses_local_eval_file(tmp_path):
    import json

    gpqa = tmp_path / "gpqa.jsonl"
    secret = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron"
    gpqa.write_text(json.dumps({"Question": secret, "Correct Answer": "x"}) + "\n")
    rows = [{"prompt": "unrelated short text"}, {"prompt": "prefix " + secret}]
    kept, dropped = ru.decontam_rows(rows, lambda r: r, "t", workdir=str(tmp_path), eval_jsonl=str(gpqa))
    assert dropped == 1 and kept == [{"prompt": "unrelated short text"}]


def test_select_coverage_greedy():
    import select_coverage as sc

    a = "the cat sat on the mat and looked at the dog"
    texts = [a, a, "quantum tunnelling lets particles cross barriers they classically could not",
             "the cat sat on the mat"]
    idx = sc.select(texts, 2, norm="none")
    assert sorted(idx) == [0, 2] or sorted(idx) == [1, 2]       # never both copies of the same text
    assert 0 not in idx or 1 not in idx
    # paper rule (norm none) favours the longest; sqrt normalisation can prefer a short novel example
    long_text = " ".join(f"word{i}" for i in range(100)) + (" " + " ".join(f"word{i}" for i in range(100))) * 3
    short_new = "entropy of an ideal gas increases under free expansion"
    assert sc.select([long_text, short_new], 1, norm="none") == [0]
    assert sc.select([long_text, short_new], 1, norm="len") == [1]
    assert sc.covered_ngrams(["a b c d e"]) == 2
