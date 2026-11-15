"""CPU-only tests for scripts/recipe_utils.py (shared by notebooks 12-14). No GPQA data is used."""

from __future__ import annotations

import dataclasses
import json
import os
import random
import sys
from pathlib import Path

import pytest

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
                 "acc_answered": 50.0, "acc_stopped": 100.0, "acc_truncated": 0.0, "truncation_rate": 50.0, "mean_tokens": 6, "repeat_rate": 0.0, "answer_then_more": 0.0,
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


# ---------------------------------------------------------------- avg@k eval, paired comparison, GRPO diagnostics


def _out(*texts, finish="stop", n_tokens=10):
    """Fake vLLM RequestOutput with one completion per text."""
    from types import SimpleNamespace

    return SimpleNamespace(outputs=[SimpleNamespace(text=t, finish_reason=finish, token_ids=[0] * n_tokens) for t in texts])


def test_score_vllm_outputs_averages_all_samples():
    outs = [_out("The answer is (A).", "The answer is (B)."), _out("The answer is (C).", "The answer is (C).")]
    agg, per_q = ru.score_vllm_outputs(outs, ["A", "C"], detail=True)
    assert per_q == [0.5, 1.0]
    assert agg["n"] == 2 and agg["samples"] == 2
    assert agg["acc_flex"] == 75.0                    # 3 of 4 samples
    assert agg["acc_flex_se"] == 17.7                 # pstdev([0.5, 1.0]) / sqrt(2) = 0.1768
    assert ru.score_vllm_outputs(outs, ["A", "C"])["acc_flex"] == 75.0


def test_paired_delta():
    d = ru.paired_delta([0.0, 0.5, 1.0, 0.5], [0.5, 0.5, 1.0, 1.0])
    assert d["delta"] == 25.0 and d["n"] == 4
    assert d["se"] == 14.4 and d["z"] == 1.73          # stdev([.5,0,0,.5]) / 2 = 0.1443
    assert ru.paired_delta([0.5, 0.5], [0.5, 0.5])["z"] == 0.0
    assert ru.paired_delta([1.0], [0.0]) == {}


def test_quick_eval_runs_compares_and_splits_main(capsys):
    sets = {
        "gpqa_diamond": [("p1", "A", "k1"), ("p2", "B", "k2")],
        "gpqa_main": [("p1m", "A", "k1"), ("p2m", "B", "k2"), ("p3m", "C", "k3"), ("p4m", "D", "k4")],
    }
    ev = ru.QuickEval(sets, max_tokens=64, samples=2, stop_ids=[1])
    seen = []

    def gen(answers):
        def generate(texts, kw):
            seen.append(kw)
            return [_out(*[f"The answer is ({a})."] * kw["n"]) for a in answers[: len(texts)]]
        return generate

    ev.run("1_sft", gen(["A", "B", "A", "A"]))        # main: 2 of 4 right (both Diamond questions)
    ev.run("2_grpo", gen(["A", "A", "C", "D"]))       # main: 3 of 4 right (one Diamond question, both others)
    assert seen[0]["n"] == 2 and seen[0]["max_tokens"] == 64 and seen[0]["stop_token_ids"] == [1]
    main = ev.results["1_sft"]["gpqa_main"]
    assert main["acc_flex"] == 50.0 and main["acc_flex_diamond_part"] == 100.0 and main["acc_flex_rest"] == 0.0
    assert ev.results["2_grpo"]["gpqa_main"]["acc_flex_diamond_part"] == 50.0
    assert "acc_flex_diamond_part" not in ev.results["1_sft"]["gpqa_diamond"]
    cmp = ev.compare_all()["1_sft -> 2_grpo"]
    assert cmp["gpqa_diamond"]["delta"] == -50.0 and cmp["gpqa_main"]["delta"] == 25.0
    out = capsys.readouterr().out
    assert "p1" not in out.replace("pass", "") and "The answer is" not in out   # aggregates only


def test_quick_eval_sample_override():
    ev = ru.QuickEval({"gpqa_diamond": [("p", "A", "k")]}, max_tokens=8, samples=4)
    assert ev.sampling_kwargs()["n"] == 4 and ev.sampling_kwargs(1)["n"] == 1


def test_pass_stats_by_type_and_zero_std():
    rows = [{"answer": "A", "answer_type": "mcq"}, {"answer": "B", "answer_type": "mcq"},
            {"answer": "2.5", "answer_type": "numeric"}]
    outs = [
        _out("(A)", "(A)", "(B)", "(A)"),                                           # 3 of 4
        _out("(A)", "(C)", "(D)", "(A)"),                                           # 0 of 4: no signal
        _out("\\boxed{2.5}", "\\boxed{2.51}", "\\boxed{3}", "no answer", n_tokens=40),  # 2 of 4
    ]
    outs[0].outputs[0].finish_reason = "length"       # right letter but cut off: not counted
    st = ru.pass_stats(outs, rows)
    assert st["correct_counts"] == [2, 0, 2]
    assert st["pass_by_type"] == {"mcq": 25.0, "numeric": 50.0} and st["pass_rate"] == 33.3
    assert st["hist"] == {0: 1, 2: 2} and st["zero_std"] == 33.3
    assert st["stop_rate"] == 91.7 and len(st["stopped_lengths"]) == 11


def test_suggest_completion_cap():
    lengths = [300] * 98 + [900, 1100]
    assert ru.suggest_completion_cap(lengths, 6144) == 2304          # 2 x p99 (1100) = 2200 -> next multiple of 256
    assert ru.suggest_completion_cap([100] * 50, 6144) == 1024        # floor
    assert ru.suggest_completion_cap([4000] * 50, 6144) == 6144       # preset cap
    assert ru.suggest_completion_cap([], 6144) == 6144                # nothing measured
    assert ru.suggest_completion_cap([100] * 50, 512) == 512          # floor never exceeds the preset


def test_question_key_ignores_case_and_spacing():
    assert ru.question_key("What  is\nX?") == ru.question_key("what is x?")
    assert ru.question_key("What is X?") != ru.question_key("What is Y?")


def test_lora_delta(tmp_path):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    a = {"l.lora_A.weight": torch.ones(2, 2), "l.lora_B.weight": torch.zeros(2, 2)}
    b = {"l.lora_A.weight": torch.ones(2, 2), "l.lora_B.weight": torch.full((2, 2), 0.5)}
    for name, sd in (("a", a), ("b", b)):
        (tmp_path / name).mkdir()
        st.save_file(sd, str(tmp_path / name / "adapter_model.safetensors"))
    d = ru.lora_delta(str(tmp_path / "a"), str(tmp_path / "b"))
    assert d["rel_change"] == 0.5 and d["max_abs_change"] == 0.5     # ||b-a|| = 1, ||a|| = 2


def test_sft_repo_name_and_auto_init_model():
    assert ru.sft_repo_name("alice", "unsloth/Qwen3-4B-Base") == "alice/qwen3-4b-s1-sft"
    assert ru.sft_repo_name("alice", "unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit") == "alice/qwen3-0.6b-s1-sft"
    assert ru.resolve_init_model("auto", "alice", "unsloth/Qwen3-4B-Base") == "alice/qwen3-4b-s1-sft"
    assert ru.resolve_init_model(None, "alice", "x/Qwen3-4B-Base") is None
    assert ru.resolve_init_model("/content/drive/m", "alice", "x/Qwen3-4B-Base") == "/content/drive/m"


def test_score_under_the_leaderboard_cap():
    from types import SimpleNamespace

    def out(text, n_tokens):
        return SimpleNamespace(outputs=[SimpleNamespace(text=text, finish_reason="stop", token_ids=list(range(n_tokens)))])

    # the long answer first mentions (B) and only reaches the right letter after the cap
    outs = [out("thinking ... (B) ... The answer is (A).", 30), out("The answer is (C).", 5)]
    decode = lambda ids: "thinking ... (B) ..." if len(ids) == 10 else "?"
    agg = ru.score_vllm_outputs(outs, ["A", "C"], cap_tokens=10, decode=decode)
    assert agg["acc_flex"] == 100.0 and agg["acc_flex_at_10"] == 50.0 and agg["over_10_rate"] == 50.0
    assert "acc_flex_at_10" not in ru.score_vllm_outputs(outs, ["A", "C"])


def test_quick_eval_only_caps_when_the_budget_is_larger():
    sets = {"gpqa_diamond": [("p", "A", "k")]}
    assert ru.QuickEval(sets, max_tokens=8192, cap_tokens=8192, decode=str).cap_tokens is None
    assert ru.QuickEval(sets, max_tokens=16384, cap_tokens=8192, decode=str).cap_tokens == 8192
    assert ru.LEADERBOARD_BUDGET == 8192 and ru.TABLE_COLS[-1] == "acc_flex_at_8192"


# ---------------------------------------------------------------- re-runnable / resumable notebook helpers


def test_load_guard_reuses_same_model_and_explains_a_changed_one():
    ns = {}
    assert ru.load_guard(ns, ("m", 12288, 64)) is True                      # nothing loaded yet
    ns.update(model=object(), _LOADED_KEY=("m", 12288, 64))
    assert ru.load_guard(ns, ("m", 12288, 64)) is False                     # same settings: reuse
    with pytest.raises(RuntimeError, match="Restart session"):
        ru.load_guard(ns, ("m", 20480, 64))                                 # different context: needs a restart
    assert ru.load_guard({"_LOADED_KEY": ("m",), "model": None}, ("x",)) is True   # load failed half-way last time


def test_stage_cache_roundtrip_and_resume_switch(tmp_path):
    c = ru.StageCache(str(tmp_path))
    assert not c.has("grpo_s0") and c.done() == []
    c.save("grpo_s0", {"steps": 60, "minutes": 31.5})
    assert c.has("grpo_s0") and c.load("grpo_s0") == {"steps": 60, "minutes": 31.5} and c.done() == ["grpo_s0"]
    assert not list(tmp_path.glob("*.tmp"))
    off = ru.StageCache(str(tmp_path), enabled=False)                       # RESUME=False ignores stored results
    assert not off.has("grpo_s0") and off.done() == []
    off.save("grpo_s0", {"steps": 1})
    assert c.load("grpo_s0") == {"steps": 1}


def test_quick_eval_reads_a_cached_eval_instead_of_generating(tmp_path, capsys):
    sets = {"gpqa_diamond": [("p1", "A", "k1"), ("p2", "B", "k2")]}
    cache = ru.StageCache(str(tmp_path))
    calls = []

    def generate(texts, kw):
        calls.append(kw["n"])
        return [_out(*["The answer is (A)."] * kw["n"]) for _ in texts]

    ev = ru.QuickEval(sets, max_tokens=64, samples=2)
    first = ev.run("1_sft", generate, cache=cache)
    ev2 = ru.QuickEval(sets, max_tokens=64, samples=2)                       # a new session after a disconnect
    again = ev2.run("1_sft", generate, cache=cache)
    assert calls == [2] and again == first and ev2.per_question["1_sft"]["gpqa_diamond"] == [1.0, 0.0]
    assert "(cached)" in capsys.readouterr().out
    ev3 = ru.QuickEval(sets, max_tokens=128, samples=2)                      # different budget: not the same eval
    ev3.run("1_sft", generate, cache=cache)
    assert calls == [2, 2]


def test_persist_dir_falls_back_to_local_outside_colab(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    path = ru.persist_dir("S1_quick", use_drive=True)
    assert path == "outputs/S1_quick" and (tmp_path / path).is_dir()
    assert "not on Colab" in capsys.readouterr().out
    assert ru.persist_dir("S1_quick", use_drive=False) == "outputs/S1_quick"


def test_reload_scripts_only_touches_loaded_modules():
    assert ru.reload_scripts(("recipe_utils", "module_that_is_not_loaded")) == ["recipe_utils"]


def test_restore_lora_resets_and_reloads(tmp_path):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    st = pytest.importorskip("safetensors.torch")

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

        def forward(self, x):
            return self.proj(x)

    model = peft.get_peft_model(Tiny(), peft.LoraConfig(r=2, lora_alpha=4, target_modules=["proj"]))
    lora_b = model.base_model.model.proj.lora_B["default"].weight
    with torch.no_grad():
        lora_b.fill_(0.5)                                                   # "trained" adapter
    saved = tmp_path / "adapter"
    saved.mkdir()
    st.save_file(peft.get_peft_model_state_dict(model), str(saved / "adapter_model.safetensors"))

    assert ru.restore_lora(model) == {"reset_layers": 1}
    assert float(lora_b.abs().sum()) == 0.0                                 # B is zero again: the adapter is a no-op
    info = ru.restore_lora(model, str(saved))
    assert info["loaded"] == str(saved) and float(lora_b.mean()) == 0.5     # the saved weights are back


def test_stage_cache_stops_resuming_when_the_settings_change(tmp_path, capsys):
    c = ru.StageCache(str(tmp_path))
    assert c.check_config({"init_model": "u/s1", "lr": 1e-5}) is True
    c.save("grpo_s0", {"steps": 60})
    same = ru.StageCache(str(tmp_path))
    assert same.check_config({"lr": 1e-5, "init_model": "u/s1"}) is True and same.has("grpo_s0")
    other = ru.StageCache(str(tmp_path))
    assert other.check_config({"init_model": "u/d1-dpo", "lr": 1e-5}) is False     # a different starting model
    assert not other.has("grpo_s0") and "init_model" in capsys.readouterr().out
    again = ru.StageCache(str(tmp_path))                                           # the new settings are now the stored ones
    assert again.check_config({"init_model": "u/d1-dpo", "lr": 1e-5}) is True


def test_read_jsonl_keeps_unicode_line_separators(tmp_path):
    rows = [{"text": "a b c\x85d"}, {"text": "plain"}]
    path = tmp_path / "x.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    assert ru.read_jsonl(path) == rows


class _FakeDrive:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    def mount(self, mountpoint):
        self.calls.append(mountpoint)
        if self.fail:
            raise ValueError("mount failed")


def test_mount_drive_moves_stale_files_and_raises_on_failure(tmp_path):
    mp = tmp_path / "drive"
    mp.mkdir()
    (mp / "leftover.txt").write_text("x")
    ok = _FakeDrive()
    ru._mount_drive(ok, str(mp))
    assert ok.calls == [str(mp)] and not mp.exists()
    assert any(p.name.startswith("drive_local_") for p in tmp_path.iterdir())
    with pytest.raises(RuntimeError, match="USE_DRIVE"):
        ru._mount_drive(_FakeDrive(fail=True), str(tmp_path / "drive2"))


def test_truncated_rollout_earns_no_correctness_or_format_reward():
    # R1: a rollout that never stopped but contained the right letter scored +2 and beat a finished wrong answer.
    text = "... so the answer is (D). And furthermore " * 3
    assert ru.correctness_score(text, "D") == 2.0
    assert ru.correctness_score(text, "D", truncated=True) == -1.0
    assert ru.correctness_score("\\boxed{8.85}", "8.85", "numeric", truncated=True) == -1.0
    assert ru.format_score("The answer is (D).", "mcq", truncated=True) == 0.0
    cap = 2048
    truncated_right = ru.correctness_score(text, "D", truncated=True) + ru.soft_overlong_penalty(cap, cap)
    finished_wrong = ru.correctness_score("The answer is (A).", "D") + ru.format_score("The answer is (A).", "mcq")
    finished_none = ru.correctness_score("I am not sure.", "D")
    assert truncated_right < finished_none < finished_wrong  # never stopping is now the worst outcome


def test_truncation_guard_trips_after_consecutive_bad_steps():
    guard = ru.TruncationGuard(threshold=0.15, patience=3)
    assert [guard.update(x) for x in (0.01, 0.2, 0.3, 0.05, 0.16, 0.2)] == [False] * 6  # the streak was broken
    assert guard.update(0.4) is True and guard.update(0.0) is True  # stays tripped
    assert ru.TruncationGuard().update(None) is False


def test_stage_accept_rejects_a_collapsed_or_worse_stage():
    good = {"pass_rate": 42.1, "stop_rate": 99.6, "vs_before_grpo": {"delta": 7.0, "z": 4.09}}
    assert ru.stage_accept(good) == (True, "ok")
    collapsed = {"pass_rate": 0.0, "stop_rate": 0.2, "vs_after_s0": {"delta": -42.1, "z": -30.0}}
    ok, why = ru.stage_accept(collapsed)
    assert not ok and "stop_rate" in why
    worse = {"pass_rate": 30.0, "stop_rate": 99.0, "vs_after_s0": {"delta": -12.1, "z": -4.2}}
    ok, why = ru.stage_accept(worse)
    assert not ok and "dropped" in why
    noise = {"pass_rate": 41.0, "stop_rate": 99.0, "vs_after_s0": {"delta": -1.1, "z": -0.6}}
    assert ru.stage_accept(noise)[0] and ru.stage_accept({})[0]


def test_handoff_repo_name_keeps_trial_runs_out_of_the_shared_repo():
    repo = "alice/qwen3-4b-s1-sft"
    assert ru.handoff_repo_name(repo, "a100", True) == repo
    assert ru.handoff_repo_name(repo, "a100", False) == repo + "-partial"
    assert ru.handoff_repo_name(repo, "quick", True) == repo + "-quick"
    assert ru.handoff_repo_name(repo, "medium", False) == repo + "-medium"


def test_handoff_meta_roundtrip_and_init_model_check(tmp_path):
    assert ru.read_handoff_meta(str(tmp_path)) is None
    assert "handoff.json" in ru.init_model_problem(None)
    ru.write_handoff_meta(str(tmp_path), {"mode": "quick", "steps": 60, "total_steps": 60, "finished": True})
    meta = ru.read_handoff_meta(str(tmp_path))
    assert meta["steps"] == 60 and "quick" in ru.init_model_problem(meta)
    assert "1600/3407" in ru.init_model_problem({"mode": "a100", "steps": 1600, "total_steps": 3407, "finished": False})
    assert ru.init_model_problem({"mode": "a100", "steps": 3407, "total_steps": 3407, "finished": True}) is None


def test_latest_grpo_checkpoint_adds_up_resumed_runs(tmp_path):
    assert ru.latest_grpo_checkpoint(str(tmp_path / "missing")) == (None, 0)
    for run, step, with_weights in (("from_0", 10, True), ("from_0", 20, True), ("from_20", 10, True), ("from_20", 20, False)):
        d = tmp_path / run / f"checkpoint-{step}"
        d.mkdir(parents=True)
        if with_weights:
            (d / "adapter_model.safetensors").write_bytes(b"x")
    path, step = ru.latest_grpo_checkpoint(str(tmp_path))
    assert step == 30 and path.endswith("from_20/checkpoint-10")  # the half-written checkpoint-20 is ignored


def test_reward_totals_keep_correctness_dominant_and_truncation_last():
    # Totals as notebook 11 adds them up (correctness + format + overlong); the cap is the length budget.
    cap, cache, w = 4096, 1024, 0.5

    def total(text, answer, n_tokens):
        cut = n_tokens >= cap
        return (ru.correctness_score(text, answer, truncated=cut) + ru.format_score(text, "mcq", w, truncated=cut)
                + ru.soft_overlong_penalty(n_tokens, cap, cache))

    right, wrong, none = "The answer is (D).", "The answer is (A).", "I am not sure."
    assert total(right, "D", 500) == total(right, "D", cap - cache) == 2.5  # no length pressure inside the budget
    assert total(right, "D", cap - 1) > total(wrong, "D", 100)  # a correct answer beats any wrong one, however long
    assert total(wrong, "D", 100) > total(none, "D", 100) > total(right, "D", cap) == -2.0


def test_emergency_checkpoint_is_found_by_resume_and_skips_bad_weights(tmp_path):
    class Model:
        def save_lora(self, path):
            os.makedirs(path)
            open(os.path.join(path, "adapter_model.safetensors"), "w").close()

    run = tmp_path / "grpo_ckpt_s0" / "from_10"
    (run / "checkpoint-5").mkdir(parents=True)
    (run / "checkpoint-5" / "adapter_model.safetensors").touch()
    ok = lambda m: True
    assert ru.save_emergency_checkpoint(Model(), str(run), 0, check=ok) is None  # no step finished
    assert ru.save_emergency_checkpoint(Model(), str(run), 8, check=lambda m: False) is None  # NaN weights
    assert ru.latest_grpo_checkpoint(str(tmp_path / "grpo_ckpt_s0"))[1] == 15
    path = ru.save_emergency_checkpoint(Model(), str(run), 8, check=ok)
    assert path.endswith("from_10/checkpoint-8") and not os.path.exists(path + ".tmp")
    assert ru.save_emergency_checkpoint(Model(), str(run), 8, check=lambda m: 1 / 0) == path  # already there: untouched
    # newest first, so a resume can fall back to the one before if the newest does not load
    assert [step for _, step in ru.list_grpo_checkpoints(str(tmp_path / "grpo_ckpt_s0"))] == [18, 15]


def test_stop_checks_are_relative_to_the_starting_model():
    val = {"stop_rate": 86.0}
    assert ru.stage_accept(val)[0] is False  # absolute floor of 90
    assert ru.stage_accept(val, baseline_stop_rate=99.9)[0] is False  # started at 99.9: a real drop
    assert ru.stage_accept(val, baseline_stop_rate=88.0)[0] is True  # started at 88: floor is 83
    assert ru.stage_accept({"stop_rate": 82.0}, baseline_stop_rate=88.0)[0] is False
    assert ru.clip_stop_threshold(0.15, None) == 0.15 and ru.clip_stop_threshold(0.15, 99.9) == 0.15
    assert ru.clip_stop_threshold(0.15, 88.0) == 0.22


def test_guarded_generate_drops_leftovers_of_an_interrupted_call():
    class Engine:  # mimics vLLM: an interrupted generate leaves its requests behind, the next call returns them first
        def __init__(self):
            self.pending, self.can_abort = [], True
            self.output_processor = self
            self.request_states, self.parent_requests = {}, {}

        def abort_request(self, ids, internal=False):
            if not self.can_abort:
                raise RuntimeError("different vLLM internals")
            self.pending, self.request_states = [], {}

    class Model:
        def __init__(self):
            self.vllm_engine = type("LLM", (), {})()
            self.vllm_engine.llm_engine = self.engine = Engine()

        def fast_generate(self, prompts, interrupt=False, **kwargs):
            self.engine.request_states = {p: 1 for p in prompts}
            if interrupt:
                self.engine.pending = ["stale:" + p for p in prompts]
                raise KeyboardInterrupt
            out, self.engine.pending, self.engine.request_states = self.engine.pending + ["out:" + p for p in prompts], [], {}
            return out

    m = Model()
    assert ru.guard_fast_generate(m) is True and ru.guard_fast_generate(m) is False
    with pytest.raises(KeyboardInterrupt):
        m.fast_generate(["a", "b", "c"], interrupt=True)
    assert m.engine.pending == []  # aborted when the call was interrupted
    assert m.fast_generate(["x", "y"]) == ["out:x", "out:y"]
    # engine internals that cannot be aborted: the leftovers finish, and are dropped from the result
    m.engine.can_abort = False
    with pytest.raises(KeyboardInterrupt):
        m.fast_generate(["a", "b", "c"], interrupt=True)
    assert m.fast_generate(["x", "y"]) == ["out:x", "out:y"]


def test_progress_file_and_when_the_starting_model_is_evaluated(tmp_path):
    from recipe_utils import init_eval_now, read_progress, write_progress
    assert read_progress(tmp_path) is None
    write_progress(tmp_path, "grpo_s0_train", step=5, of=60)
    write_progress(tmp_path, "grpo_s0_train", step=6, of=60)
    p = read_progress(tmp_path)
    assert (p["phase"], p["step"], p["of"]) == ("grpo_s0_train", 6, 60) and p["utc"]
    assert sorted(f.name for f in tmp_path.iterdir()) == ["progress.json"]
    write_progress(tmp_path / "missing" / "dir", "x")                 # must not raise

    # "after": nothing before training, the eval once training is done
    assert not init_eval_now("after", True, run_grpo=True, after_grpo=False)
    assert init_eval_now("after", True, run_grpo=True, after_grpo=True)
    # "after" without a GRPO stage in this run: up front, and not a second time
    assert init_eval_now("after", True, run_grpo=False, after_grpo=False)
    assert not init_eval_now("after", True, run_grpo=False, after_grpo=True)
    # True = before training only; False / no starting model = never
    assert init_eval_now(True, True, run_grpo=True, after_grpo=False)
    assert not init_eval_now(True, True, run_grpo=True, after_grpo=True)
    assert not any(init_eval_now(False, True, True, a) or init_eval_now("after", False, True, a) for a in (False, True))


def test_guard_callback_reports_every_step_and_survives_a_failing_hook():
    pytest.importorskip("transformers")
    from types import SimpleNamespace
    from recipe_utils import TruncationGuard, truncation_guard_callback
    seen = []
    cb = truncation_guard_callback(TruncationGuard(0.5, 2), on_step=lambda step, logs: seen.append((step, logs["reward"])))
    control = SimpleNamespace(should_training_stop=False)
    cb.on_log(None, SimpleNamespace(global_step=3), control, logs={"reward": 1.5, "completions/clipped_ratio": 0.1})
    assert seen == [(3, 1.5)] and not control.should_training_stop
    bad = truncation_guard_callback(TruncationGuard(0.5, 1), on_step=lambda step, logs: 1 / 0)
    bad.on_log(None, SimpleNamespace(global_step=1), control, logs={"completions/clipped_ratio": 0.9})
    assert control.should_training_stop


class _Out:
    """A vLLM RequestOutput stand-in: one completion per text; finish 'length' for texts ending in '...'."""

    def __init__(self, texts):
        from types import SimpleNamespace
        self.outputs = [SimpleNamespace(text=x, finish_reason="length" if x.endswith("...") else "stop",
                                        token_ids=list(range(len(x.split())))) for x in texts]


def test_eval_is_split_by_how_answers_end_and_by_subject():
    golds = ["A", "B", "C", "D"]
    groups = ["Physics", "Physics", "Chemistry", "Chemistry"]
    outs = [_Out(["The answer is (A).", "no letter here"]),         # physics: 1 of 2, one answer without a letter
            _Out(["The answer is (B).", "The answer is (B)."]),     # physics: 2 of 2
            _Out(["The answer is (D).", "thinking ..."]),           # chemistry: 0 of 2, one cut off
            _Out(["(D) and then on and on ...", "The answer is (A)."])]   # chemistry: 1 of 2 (cut off but right)
    r, per_q = ru.score_vllm_outputs(outs, golds, detail=True, groups=groups)
    assert per_q == [0.5, 1.0, 0.0, 0.5] and r["acc_flex"] == 50.0
    assert r["extract_rate"] == 75.0 and r["acc_answered"] == round(100 * 4 / 6, 1)     # 4 right of the 6 with a letter
    assert r["acc_stopped"] == 50.0 and r["acc_truncated"] == 50.0                      # 3 of 6, 1 of 2
    assert r["by_domain"] == {"Chemistry": {"n": 2, "acc_flex": 25.0, "extract_rate": 75.0, "truncation_rate": 50.0},
                              "Physics": {"n": 2, "acc_flex": 75.0, "extract_rate": 75.0, "truncation_rate": 0.0}}
    assert "by_domain" not in ru.score_vllm_outputs(outs, golds)                         # no labels: nothing added
    assert ru.aggregate([{"flex": False, "extracted": False, "truncated": False, "n_tokens": 1, "letter": "none"}])["acc_answered"] is None


def test_starting_eval_is_taken_from_the_sft_notebook_folder(tmp_path, capsys):
    import os
    sets = {"gpqa_diamond": [("p1", "A", "k1"), ("p2", "B", "k2"), ("p3", "C", "k3")]}
    groups = {"gpqa_diamond": ["Physics", "Physics", "Biology"]}
    meta = dict(notebook="15_sft_checkpoint", exp="S1", mode="a100", variant="s1", steps=3407, loss_last=0.5548, finished=True)
    folder = tmp_path / "S1_a100"
    folder.mkdir()
    saved = {"settings": {"samples": 4, "max_tokens": 8192, "temperature": 0.6, "n": {"gpqa_diamond": 3, "gpqa_main": 5}},
             "results": {"gpqa_diamond": {"n": 3, "acc_flex": 50.0, "samples": 4}, "gpqa_main": {"n": 5, "acc_flex": 40.0}},
             "per_question": {"gpqa_diamond": [1.0, 0.25, 0.25], "gpqa_main": [0.0] * 5}}
    (folder / "stage_sft.json").write_text(json.dumps({"steps": 3407, "loss_last": 0.5548, "finished": True}))
    (folder / "stage_eval_1_sft.json").write_text(json.dumps(saved))

    ev = ru.QuickEval(sets, max_tokens=8192, samples=8, groups=groups)
    assert ru.reuse_sft_eval(ev, "0_init", meta, str(tmp_path))
    assert ev.samples == 4                                         # later stages use the same number of samples
    r = ev.results["0_init"]
    assert list(r) == ["gpqa_diamond"] and r["gpqa_diamond"]["acc_flex"] == 50.0      # only this run's question sets
    assert r["gpqa_diamond"]["by_domain"] == {"Biology": {"n": 1, "acc_flex": 25.0}, "Physics": {"n": 2, "acc_flex": 62.5}}
    assert r["gpqa_diamond"]["reused_from"].endswith("stage_eval_1_sft.json")
    # a later stage is compared with it question by question, overall and per subject
    ev.run("2_grpo", lambda texts, kw: [_Out(["The answer is (A)."] * kw["n"]), _Out(["The answer is (B)."] * kw["n"]),
                                         _Out(["The answer is (A)."] * kw["n"])])
    cmp = ev.compare_all()["0_init -> 2_grpo"]["gpqa_diamond"]
    assert cmp["delta"] == round(100 * (0.0 + 0.75 - 0.25) / 3, 1)
    assert cmp["by_domain"]["Physics"]["delta"] == 37.5 and cmp["by_domain"]["Biology"] == {}   # one question: no SE
    assert "按学科" in capsys.readouterr().out

    keep = ru.QuickEval(sets, max_tokens=8192, samples=8)
    assert ru.reuse_sft_eval(keep, "0_init", meta, str(tmp_path), keep_samples=True) and keep.samples == 8

    # not reused: another budget, another question count, another training run, an eval older than the training
    assert not ru.reuse_sft_eval(ru.QuickEval(sets, max_tokens=16384, samples=8), "0_init", meta, str(tmp_path))
    assert not ru.reuse_sft_eval(ru.QuickEval({"gpqa_diamond": sets["gpqa_diamond"][:2]}, max_tokens=8192), "0_init", meta, str(tmp_path))
    assert ru.find_sft_eval(dict(meta, steps=60), str(tmp_path))[0] is None
    assert ru.find_sft_eval(dict(meta, exp="S9"), str(tmp_path))[0] is None
    assert ru.find_sft_eval(None, str(tmp_path))[0] is None
    old = (folder / "stage_sft.json").stat().st_mtime - 100
    os.utime(folder / "stage_eval_1_sft.json", (old, old))
    assert ru.find_sft_eval(meta, str(tmp_path))[0] is None
