"""CPU-only tests for the helper scripts (no GPU, no network, no GPQA data)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import aggregate_results  # noqa: E402
import build_leaderboard  # noqa: E402
import decontam  # noqa: E402
import prepare_dolci_sft  # noqa: E402

FAKE_Q = (
    "A synthetic placeholder question about the spin state of a hypothetical particle "
    "trapped inside an imaginary harmonic potential well at zero temperature"
)


def test_ngrams_and_check(tmp_path: Path):
    eval_file = tmp_path / "gpqa.jsonl"
    eval_file.write_text(json.dumps({"Question": FAKE_Q, "Correct Answer": "x"}) + "\n")
    grams, n = decontam.load_gpqa_ngrams(eval_file)
    assert n == 1 and grams

    rows = [
        {"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]},
        {"messages": [{"role": "user", "content": "Q: " + FAKE_Q.upper() + "?"}]},  # case-insensitive hit
        {"prompt": [{"role": "user", "content": "unrelated"}], "completion": [{"role": "assistant", "content": "ok"}]},
    ]
    report = decontam.check(rows, grams)
    assert report["train_rows_checked"] == 3
    assert report["flagged_row_indices"] == [1]


def test_decontam_cli(tmp_path: Path, capsys):
    eval_file = tmp_path / "gpqa.jsonl"
    eval_file.write_text(json.dumps({"Question": FAKE_Q}) + "\n")
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"text": "nothing to see here"}) + "\n")
    assert decontam.main(["--train-jsonl", str(train), "--eval-jsonl", str(eval_file)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["train_rows_flagged"] == 0
    assert FAKE_Q not in json.dumps(out)  # never echo eval text


def _fake_lm_eval_output(root: Path, seed: int, diamond: float, main: float):
    d = root / f"seed{seed}" / "org__model"
    d.mkdir(parents=True)
    (d / "results_2026-11-15T10-00-00.json").write_text(
        json.dumps(
            {
                "results": {
                    "gpqa_diamond_cot_zeroshot": {"exact_match,flexible-extract": diamond, "exact_match,strict-match": 0.1},
                    "gpqa_main_cot_zeroshot": {"exact_match,flexible-extract": main, "exact_match,strict-match": 0.1},
                }
            }
        )
    )


def test_aggregate(tmp_path: Path):
    for seed, (d, m) in enumerate([(0.30, 0.28), (0.32, 0.30), (0.34, 0.29), (0.36, 0.31)]):
        _fake_lm_eval_output(tmp_path, seed, d, m)
    out = tmp_path / "sub.json"
    assert aggregate_results.main([str(tmp_path), "--model", "org/model", "--out", str(out)]) == 0
    sub = json.loads(out.read_text())
    assert sub["scores"]["gpqa_diamond"]["mean"] == 33.0
    assert sub["scores"]["gpqa_diamond"]["per_seed"] == {"0": 30.0, "1": 32.0, "2": 34.0, "3": 36.0}
    assert abs(sub["scores"]["gpqa_diamond"]["stderr"] - 1.29) < 0.01
    assert sub["team"] == "org"


def test_example_and_reference_submissions_validate():
    schema = json.loads((ROOT / "submissions" / "schema.json").read_text())
    for path in (ROOT / "submissions").glob("*.json"):
        if path.name == "schema.json":
            continue
        assert build_leaderboard.validate(json.loads(path.read_text()), schema) == [], path.name


def test_bad_base_model_rejected():
    schema = json.loads((ROOT / "submissions" / "schema.json").read_text())
    sub = json.loads((ROOT / "submissions" / "_example.json").read_text())
    sub["base_model"] = "Qwen/Qwen3-4B"  # starting from the post-trained model is not allowed
    assert build_leaderboard.validate(sub, schema)


def test_render_orders_by_diamond():
    base = json.loads((ROOT / "submissions" / "_example.json").read_text())
    a = dict(base, team="a", scores={"gpqa_diamond": {"mean": 30.0, "stderr": 1.0}})
    b = dict(base, team="b", scores={"gpqa_diamond": {"mean": 40.0, "stderr": 1.0}})
    md = build_leaderboard.render([a, b])
    assert md.index("| 1 | b |") < md.index("| 2 | a |")


def test_prompt_completion_conversion():
    msgs = prepare_dolci_sft.get_messages(
        {"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "<think>x</think>hello"}]}
    )
    pc = prepare_dolci_sft.to_prompt_completion(msgs)
    assert pc["prompt"] == [{"role": "user", "content": "hi"}]
    assert pc["completion"][0]["role"] == "assistant"
    assert prepare_dolci_sft.keep_source("ai2-adapt/science_qa", ["science"], ["safety"])
    assert not prepare_dolci_sft.keep_source("wildguard_safety", [], ["safety"])
