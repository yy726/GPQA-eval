"""CPU-only tests for the submission-branch helpers (Dolci prep, sample analysis)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_samples  # noqa: E402
import prepare_dolci_sft  # noqa: E402


def test_prompt_completion_conversion():
    msgs = prepare_dolci_sft.get_messages(
        {"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "<think>x</think>hello"}]}
    )
    pc = prepare_dolci_sft.to_prompt_completion(msgs)
    assert pc["prompt"] == [{"role": "user", "content": "hi"}]
    assert pc["completion"][0]["role"] == "assistant"
    assert prepare_dolci_sft.keep_source("ai2-adapt/science_qa", ["science"], ["safety"])
    assert not prepare_dolci_sft.keep_source("wildguard_safety", [], ["safety"])


def test_strip_think():
    assert prepare_dolci_sft.strip_think("<think>long\nreasoning</think>\n\nThe answer is (B).") == "The answer is (B)."
    assert prepare_dolci_sft.strip_think("no think here") == "no think here"


def _sample(doc_id, filt, text, extracted, correct):
    return {
        "doc_id": doc_id,
        "doc": {"Question": "SECRET QUESTION TEXT"},
        "target": "(A)",
        "resps": [[text]],
        "filtered_resps": [extracted],
        "filter": filt,
        "exact_match": correct,
    }


def test_analyze_samples(tmp_path: Path):
    d = tmp_path / "seed0" / "model"
    d.mkdir(parents=True)
    rows = [
        _sample(0, "strict-match", "<think>x</think> The answer is (A).", "(A)", 1),
        _sample(0, "flexible-extract", "<think>x</think> The answer is (A).", "(A)", 1),
        _sample(1, "strict-match", "I think (C) maybe", "[invalid]", 0),
        _sample(1, "flexible-extract", "I think (C) maybe", "(C)", 0),
    ]
    (d / "samples_gpqa_diamond_cot_zeroshot_2026-11-15T10-00-00.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    rep = analyze_samples.analyze_dir(tmp_path)
    stats = rep["gpqa_diamond"]["seed0"]
    assert stats["strict-match"]["accuracy"] == 50.0
    assert stats["strict-match"]["extract_rate"] == 50.0
    assert stats["flexible-extract"]["extract_rate"] == 100.0
    assert stats["flexible-extract"]["pred_letter_dist"] == {"A": 1, "C": 1}
    assert stats["length"]["n_docs"] == 2
    assert stats["length"]["has_think_rate"] == 50.0
    assert "SECRET" not in json.dumps(rep)  # never leak question text


def test_chatml_fallback_template():
    jinja2 = __import__("pytest").importorskip("jinja2")
    out = jinja2.Template(prepare_dolci_sft.CHATML_TEMPLATE).render(
        messages=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
        add_generation_prompt=False,
    )
    assert out == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello<|im_end|>\n"

    class Tok:
        chat_template = None

        def apply_chat_template(self, msgs, tokenize=True):
            return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    tok = prepare_dolci_sft.ensure_chat_template(Tok())
    assert tok.chat_template == prepare_dolci_sft.CHATML_TEMPLATE
    assert prepare_dolci_sft.count_chat_tokens(tok, [{"role": "user", "content": "x"}]) == 3
