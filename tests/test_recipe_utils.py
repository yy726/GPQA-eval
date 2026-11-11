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
                 "truncation_rate": 50.0, "mean_tokens": 6, "letters": {"A": 1, "B": 1}}
    assert ru.aggregate([]) == {"n": 0}


def test_config_kwargs_filters_unknown_fields(capsys):
    @dataclasses.dataclass
    class FakeConfig:
        beta: float = 0.1
        learning_rate: float = 1e-5

    kept = ru.config_kwargs(FakeConfig, beta=0.2, rpo_alpha=1.0)
    assert kept == {"beta": 0.2}
    assert "rpo_alpha" in capsys.readouterr().out


def test_setup_chatml_stop_on_base_like_tokenizer():
    class Tok:
        chat_template = None
        pad_token = None
        eos_token = "<|endoftext|>"
        vocab = {"<|im_end|>": 1, "<|endoftext|>": 2, "<|im_start|>": 3}

        def convert_tokens_to_ids(self, t):
            return self.vocab[t]

    tok = Tok()
    ids = ru.setup_chatml_stop(tok)
    assert ids == [1, 2, 3]
    assert tok.chat_template == ru.CHATML_TEMPLATE
    assert tok.eos_token == "<|im_end|>" and tok.pad_token == "<|endoftext|>"


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
