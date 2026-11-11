"""CPU-only tests for scripts/science_mix.py (science SFT mix and held-out MCQ pools). No GPQA data is used."""

from __future__ import annotations

import dataclasses
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import recipe_utils as ru  # noqa: E402
import science_mix as sm  # noqa: E402

Q4 = ("Which reagent adds to a ketone to give a tertiary alcohol?\n\n"
      "A: Grignard reagent (RMgX)\nB: LDA\nC: A Gilman reagent, R2CuLi\nD: Sodium amide")


def mega_row(question=Q4, answer="Step 1 ...\n\n**Answer:** \\boxed{A}", reference="... \\boxed{A}", source="nemotro_science",
             subject=""):
    return {"question": question, "answer": answer, "reference_answer": reference, "source": source, "subject": subject}


def test_parse_mcq_styles():
    stem, choices = sm.parse_mcq(Q4)
    assert stem == "Which reagent adds to a ketone to give a tertiary alcohol?"
    assert choices == ["Grignard reagent (RMgX)", "LDA", "A Gilman reagent, R2CuLi", "Sodium amide"]
    assert sm.parse_mcq("Stem?\n(A) one\n(B) two\n(C) three\n(D) four")[1] == ["one", "two", "three", "four"]
    assert sm.parse_mcq("Stem?\nA. one\nB. two")[1] == ["one", "two"]
    assert len(sm.parse_mcq("Stem?\n" + "\n".join(f"{c}: opt {c}" for c in "ABCDEFGHIJ"))[1]) == 10


def test_parse_mcq_rejects_unsafe_layouts():
    assert sm.parse_mcq("No options here.") is None
    assert sm.parse_mcq("Stem?\nA: one\nC: three") is None                     # gap in the letters
    assert sm.parse_mcq("Stem?\nA: one\nB: two\nWhich is right?") is None       # text after the options
    assert sm.parse_mcq("Stem?\nA: one\n   continued\nB: two") is None          # multi-line option
    assert sm.parse_mcq("A: one\nB: two") is None                               # no stem


def test_final_letter():
    assert sm.final_letter("x \\boxed{B} then \\boxed{C}") == "C"
    assert sm.final_letter("\\boxed{\\text{D}}") == "D" and sm.final_letter("\\(\\boxed{(A)}\\)") == "A"
    assert sm.final_letter("So the answer is (B).") == "B" and sm.final_letter("The answer is **C**.") == "C"
    assert sm.final_letter("\\boxed{42}") is None and sm.final_letter(None) is None


def test_gpqa_style_solution_strips_the_box():
    for tail in ["\n\n**Answer:** \\boxed{C}", "\n\n### Final Answer:\n\\[\n\\boxed{C}\n\\]", "\n\n\\(\\boxed{C}\\)",
                 "\n\n\\boxed{\\text{C}}", " $\\boxed{C}$."]:
        assert sm.gpqa_style_solution("Reasoning about C." + tail, "C") == "Reasoning about C.\n\nThe answer is (C)."
    # a box in the middle of the text is left alone
    out = sm.gpqa_style_solution("First \\boxed{A} is wrong.\nMore text.", "B")
    assert "\\boxed{A}" in out and out.endswith("The answer is (B).")


def test_mcq_row_is_a_gpqa_prompt_scored_by_the_eval_regexes():
    r = sm.mcq_row(mega_row())
    assert r["answer"] == "A" and r["answer_type"] == "mcq" and r["agrees"]
    assert r["prompt"] == ru.format_mcq("Which reagent adds to a ketone to give a tertiary alcohol?",
                                        ["Grignard reagent (RMgX)", "LDA", "A Gilman reagent, R2CuLi", "Sodium amide"])
    assert r["prompt"].endswith("Let's think step by step: ")
    score = ru.score_completion(r["completion"], "A")
    assert score["strict"] and score["flex"]                    # both lm-eval extractors read the right letter
    assert "\\boxed" not in r["completion"]


def test_mcq_row_filters():
    assert sm.mcq_row(mega_row(answer="... \\boxed{B}")) is None                         # solution disagrees with reference
    assert sm.mcq_row(mega_row(answer="... \\boxed{B}"), require_agreement=False)["agrees"] is False
    assert sm.mcq_row(mega_row(reference="no letter")) is None
    ten = "Stem?\n" + "\n".join(f"{c}: opt {c}" for c in "ABCDEFGHIJ")
    assert sm.mcq_row(mega_row(question=ten, answer="\\boxed{I}", reference="\\boxed{I}")) is None   # not 4 options
    assert sm.mcq_row(mega_row(question="Open question?")) is None


def test_question_bucket_is_stable_and_splits():
    assert sm.question_bucket("What  is X?") == sm.question_bucket("what is x?")
    rows = [{"bucket": sm.question_bucket(f"question {i}"), "i": i} for i in range(2000)]
    parts = sm.split_mcq(rows, sft_share=0.55, val_share=0.10)
    assert sum(len(v) for v in parts.values()) == 2000
    assert 1000 < len(parts["sft"]) < 1200 and 600 < len(parts["rl"]) < 800 and 140 < len(parts["val"]) < 260
    ids = [{r["i"] for r in parts[k]} for k in ("sft", "rl", "val")]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_balance_letters_caps_frequent_letters():
    rows = [{"answer": l, "i": i} for i, l in enumerate("A" * 40 + "B" * 60 + "C" * 20 + "D" * 24)]
    out = sm.balance_letters(rows, ratio=1.25, seed=0)
    assert Counter(r["answer"] for r in out) == {"A": 25, "B": 25, "C": 20, "D": 24}
    assert [r["i"] for r in out] == sorted(r["i"] for r in out)       # order kept
    assert sm.balance_letters(rows[:40]) == rows[:40]                  # a single letter: nothing to balance


def test_qa_row_and_message_format():
    r = sm.qa_row(mega_row(question="Why is the sky blue?", answer="Rayleigh scattering ...", source="textbook_reasoning",
                           subject="physics"), "textbook_physics")
    assert r["source"] == "megascience/textbook_reasoning/physics" and r["part"] == "textbook_physics"
    m = sm.to_messages(r)
    assert [x["role"] for x in m["messages"]] == ["user", "assistant"] and m["part"] == "textbook_physics"
    assert sm.qa_row(mega_row(answer=" "), "x") is None


def test_pool_row_has_no_solution(tmp_path):
    r = sm.mcq_row(mega_row())
    p = sm.pool_row(r)
    assert set(p) == {"prompt", "answer", "answer_type", "source"} and p["prompt"][0]["role"] == "user"
    f = tmp_path / "rl_mcq.jsonl"
    f.write_text("".join(json.dumps(dict(p, answer=l)) + "\n" for l in "ABCD"))
    assert len(sm.load_mcq_pool(str(f))) == 4 and len(sm.load_mcq_pool(str(f), n=2)) == 2
    assert sm.load_mcq_pool(str(f), seed=1) == sm.load_mcq_pool(str(f), seed=1)


def test_parse_counts():
    assert sm.parse_counts(["physics=20000", "math=8000"]) == {"physics": 20000, "math": 8000}


def test_dpo_loss_kwargs_across_trl_versions(capsys):
    @dataclasses.dataclass
    class NewCfg:
        """loss types: 'sigmoid', 'sft', 'sigmoid_norm'."""
        loss_type: list = None
        loss_weights: list = None
        beta: float = 0.1

    @dataclasses.dataclass
    class OldCfg:
        """loss types: 'sigmoid', 'ipo'."""
        loss_type: str = "sigmoid"
        rpo_alpha: float = None
        beta: float = 0.1

    assert ru.dpo_loss_kwargs(NewCfg) == {"loss_type": "sigmoid_norm", "beta": 5.0}
    assert ru.dpo_loss_kwargs(NewCfg, sft_weight=1.0) == {"loss_type": ["sigmoid_norm", "sft"], "loss_weights": [1.0, 1.0],
                                                           "beta": 5.0}
    assert ru.dpo_loss_kwargs(NewCfg, "sigmoid", beta=0.2) == {"loss_type": "sigmoid", "beta": 0.2}
    assert ru.dpo_loss_kwargs(OldCfg, sft_weight=0.5) == {"loss_type": "sigmoid", "beta": 0.1, "rpo_alpha": 0.5}
    assert "no 'sigmoid_norm'" in capsys.readouterr().out


def test_select_per_part_respects_targets_and_prefers_new_content():
    import select_coverage as sc

    def row(part, text, n=10):
        return {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": text}], "part": part, "n_tokens": n}

    dup = "the quick brown fox jumps over the lazy dog again and again"
    rows = [row("a", dup), row("a", dup), row("a", "an entirely different sentence about enzyme kinetics and rate laws"),
            row("b", "kept whole one"), row("b", "kept whole two"), row("c", "no target for this part")]
    out, info = sc.select_per_part(rows, {"a": 2, "b": 5}, method="coverage", norm="none")
    texts = [r["messages"][1]["content"] for r in out if r["part"] == "a"]
    assert len(texts) == 2 and len(set(texts)) == 2                     # the duplicate is not picked twice
    assert sum(r["part"] == "b" for r in out) == 2                      # fewer candidates than the target: all kept
    assert all(r["part"] != "c" for r in out)
    assert info["a"] == {"pool": 3, "target": 2, "selected": 2, "tokens": 20} and info["b"]["selected"] == 2
    rnd, _ = sc.select_per_part(rows, {"a": 2}, method="random", seed=1)
    assert len(rnd) == 2
