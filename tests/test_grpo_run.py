"""The GRPO stage loop of notebook 11 (scripts/grpo_run.py) with a fake model and a fake trainer: no GPU,
no downloads. Checks the bookkeeping that cannot be seen until a Colab run is hours in: what is written to
the persistent directory, what a second run reads back, where an interrupted stage continues, and what
happens to a stage that fails the acceptance check."""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

pytest.importorskip("datasets")

import grpo_run  # noqa: E402
import recipe_utils as ru  # noqa: E402


class FakeModel:
    """fast_generate answers every prompt k times: right on the first `right` samples, wrong on the rest;
    samples of prompts listed in `runaway` never stop."""

    def __init__(self):
        self.right, self.runaway, self.calls, self.loaded = 2, set(), [], []

    def fast_generate(self, prompts, sampling_params=None, lora_request=None):
        self.calls.append((len(prompts), sampling_params["n"], lora_request))
        outs = []
        for p in prompts:
            gold = p.split("gold=")[1][0]
            comps = []
            for i in range(sampling_params["n"]):
                if p in self.runaway:
                    comps.append(SimpleNamespace(text="and so on", finish_reason="length", token_ids=[0] * 50))
                else:
                    letter = gold if i < self.right else ("A" if gold != "A" else "B")
                    comps.append(SimpleNamespace(text=f"The answer is ({letter}).", finish_reason="stop", token_ids=[0] * 20))
            outs.append(SimpleNamespace(outputs=comps))
        return outs

    def named_parameters(self):
        return []                      # no LoRA tensors to check for NaN

    def load_lora(self, path):
        self.loaded.append(path)
        return f"req:{path}"

    def save_lora(self, path):
        os.makedirs(path, exist_ok=True)
        Path(path, "adapter_model.safetensors").write_text("weights")


class FakeTrainer:
    """Runs max_steps steps: calls the reward functions on one fake batch per step, logs, saves a checkpoint
    every save_steps. `fail_at` (class attribute) raises KeyboardInterrupt at that step of the next train()."""

    fail_at = None
    created = []

    def __init__(self, model, processing_class, reward_funcs, args, train_dataset):
        self.model, self.reward_funcs, self.args, self.ds = model, reward_funcs, args, train_dataset
        self.state = SimpleNamespace(global_step=0, log_history=[])
        self.callbacks = []
        FakeTrainer.created.append(self)

    def add_callback(self, cb):
        self.callbacks.append(cb)

    def train(self):
        a = self.args
        control = SimpleNamespace(should_training_stop=False)
        fail_at, FakeTrainer.fail_at = FakeTrainer.fail_at, None
        for step in range(1, a.max_steps + 1):
            if fail_at == step:
                raise KeyboardInterrupt()
            row = self.ds[(step - 1) * a.gradient_accumulation_steps]
            n = a.num_generations
            comps = [[{"role": "assistant", "content": f"The answer is ({row['answer'] if i % 2 else 'Z'})."}] for i in range(n)]
            kw = dict(prompts=[row["prompt"]] * n, completions=comps, answer=[row["answer"]] * n,
                      answer_type=[row["answer_type"]] * n, completion_ids=[[0] * 20] * n, log_metric=lambda k, v: None)
            total = [sum(x) for x in zip(*[f(**kw) for f in self.reward_funcs])]
            self.state.global_step = step
            logs = {"reward": sum(total) / n, "completions/clipped_ratio": 0.0}
            self.state.log_history.append(logs)
            for cb in self.callbacks:
                cb.on_log(a, self.state, control, logs=logs)
            if step % a.save_steps == 0:
                self.model.save_lora(os.path.join(a.output_dir, f"checkpoint-{step}"))
            if control.should_training_stop:
                break
        return SimpleNamespace(metrics={"train_runtime": 120.0})


@pytest.fixture
def nb(tmp_path, monkeypatch):
    """A notebook namespace with everything GrpoRun reads, on fakes."""
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(GRPOConfig=lambda **kw: SimpleNamespace(**kw),
                                                                 GRPOTrainer=FakeTrainer))
    FakeTrainer.created, FakeTrainer.fail_at = [], None
    letters = "ABCD"

    def load_mcq_rows(source, n, skip=0):
        return [{"prompt": f"{source} q{skip + i} gold={letters[(skip + i) % 4]}", "answer": letters[(skip + i) % 4],
                 "source": source} for i in range(n)]

    restored = []
    monkeypatch.setattr(ru, "load_mcq_rows", load_mcq_rows)
    monkeypatch.setattr(ru, "decontam_rows", lambda rows, to_record, name, **kw: (rows, 0))
    monkeypatch.setattr(ru, "n_chat_tokens", lambda tok, messages, add_generation_prompt=False: 30)
    monkeypatch.setattr(ru, "chat_prompt", lambda tok, text: text)
    monkeypatch.setattr(ru, "restore_lora", lambda model, path=None: restored.append(path) or {"restored": path})
    monkeypatch.setattr(ru, "lora_delta", lambda a, b: {"rel_change": 0.1})
    persist = tmp_path / "gpqa_ckpt" / "R9_test"
    cache = ru.StageCache(str(persist))
    evals = []
    ns = dict(
        model=FakeModel(), tokenizer=object(), CACHE=cache, PERSIST=str(persist), STOP_IDS=[1], MAX_SEQ_LEN=4096,
        RUN_NAME="R9-test", wandb=None, INIT_MODEL="u/s1-sft", RUN_GRPO=True,
        Q=dict(grpo_steps=8, grpo_prompts_per_step=2, num_gen=4, grpo_max_completion=2048, filter_k=4,
               filter_max_tokens=512, val_n=6),
        GRPO_DATA="sciq", CURRICULUM_ROUNDS=2, CKPT_EVERY=2, RUN_INIT_EVAL=False,
        quick_eval=lambda tag, lora_request=None, samples=None: evals.append((tag, lora_request)),
        diagnose=lambda tag, lora_request=None: None,
    )
    ns["_restored"], ns["_evals"] = restored, evals
    return ns


def _files(ns):
    return sorted(p.name for p in Path(ns["PERSIST"]).iterdir())


def test_two_stages_then_a_second_run_reads_everything_back(nb):
    run = grpo_run.GrpoRun(nb)
    info = run.prepare()
    assert info["steps_per_stage"] == 4 and info["prompts_per_stage"] == 8 and info["val_prompts"] == 6
    out = run.train()

    assert out["final_adapter"] == "grpo_lora_s1" and run.final_adapter == "grpo_lora_s1"
    assert [s["steps"] for s in out["stages"]] == [4, 4] and not any("rejected" in s for s in out["stages"])
    assert list(out["rl_val"]) == ["before_grpo", "after_s0", "after_s1"]
    assert out["rl_val"]["after_s1"]["vs_before_grpo"]["n"] == 6          # paired with the starting model
    assert nb["_evals"] == [("2_grpo_s1", f"req:{run.adir('grpo_lora_s1')}")]   # final eval with the final adapter
    # reward functions keep the names TRL logs them under, and the summary picked the reward up from the log
    assert [f.__name__ for f in run.reward_funcs()] == ["format_reward", "correctness_reward", "overlong_reward"]
    assert out["stages"][0]["reward"]["first"] == pytest.approx(0.75)       # half right (+2.5), half without a valid answer (-1)
    # stage 2 trains with the lower learning rate, each stage writes checkpoints under from_0
    lrs = [t.args.learning_rate for t in FakeTrainer.created]
    assert lrs == [1e-5, 0.5e-5] and FakeTrainer.created[0].args.output_dir.endswith("grpo_ckpt_s0/from_0")
    assert FakeTrainer.created[0].args.mask_truncated_completions is True and FakeTrainer.created[0].args.scale_rewards == "batch"
    # training prompts never include a validation prompt; the two stages use different prompts
    val = {r["prompt"][0]["content"] for r in run.val_rows}
    used = [{r["prompt"][0]["content"] for r in t.ds} for t in FakeTrainer.created]
    assert not (used[0] & val) and not (used[1] & val) and not (used[0] & used[1])
    # on disk: adapters and results of both stages; checkpoints and prompt lists of finished stages are gone
    files = _files(nb)
    assert {"grpo_lora_s0", "grpo_lora_s1", "stage_grpo_s0.json", "stage_grpo_s1.json", "stage_rl_val_before_grpo.json",
            "grpo_diag_s0.jsonl", "progress.json"} <= set(files)
    assert not [f for f in files if f.startswith(("grpo_ckpt", "grpo_plan"))]
    assert ru.read_progress(nb["PERSIST"])["phase"] == "grpo_done"
    assert json.loads(Path(nb["PERSIST"], "grpo_diag_s0.jsonl").read_text().splitlines()[0])["diag/correct_and_stopped"] == 0.5

    # second run (after a disconnect): nothing is trained or sampled again
    n_trainers, n_calls = len(FakeTrainer.created), len(nb["model"].calls)
    again = grpo_run.GrpoRun(nb)
    out2 = again.train()
    assert out2["final_adapter"] == "grpo_lora_s1" and len(FakeTrainer.created) == n_trainers
    assert len(nb["model"].calls) == n_calls and [s["steps"] for s in out2["stages"]] == [4, 4]


def test_an_interrupted_stage_saves_and_the_next_run_continues_from_that_step(nb, capsys):
    run = grpo_run.GrpoRun(nb)
    FakeTrainer.fail_at = 4                                    # stop in step 4: 3 steps done, last periodic save at step 2
    with pytest.raises(KeyboardInterrupt):
        run.train()
    ckpts = ru.list_grpo_checkpoints(run.adir("grpo_ckpt_s0"))
    assert [step for _, step in ckpts] == [3, 2]               # the emergency save of step 3 is the newest
    assert "停下" in capsys.readouterr().out
    assert ru.read_progress(nb["PERSIST"])["phase"] == "grpo_s0_stopped"
    plan = json.load(open(run._plan_path(0)))
    first_prompts = [r["prompt"][0]["content"] for r in FakeTrainer.created[0].ds]

    again = grpo_run.GrpoRun(nb)
    out = again.train()
    resumed = FakeTrainer.created[1]
    assert resumed.args.max_steps == 1 and resumed.args.output_dir.endswith("grpo_ckpt_s0/from_3")
    assert ckpts[0][0] in nb["_restored"]                      # weights of step 3 were loaded
    # same prompt list as before the interruption; the resumed run starts after the prompts of the 3 finished steps
    assert [r["prompt"][0]["content"] for r in resumed.ds] == first_prompts[3 * 2:] and len(plan["rows"]) == 8
    assert out["stages"][0]["steps"] == 4 and out["stages"][0]["resumed_from"] == 3
    assert out["final_adapter"] == "grpo_lora_s1"


def test_a_stage_that_stops_finishing_is_set_aside_and_nothing_is_evaluated(nb):
    run = grpo_run.GrpoRun(nb)
    run.prepare()
    model = nb["model"]
    real = model.fast_generate

    def generate(prompts, sampling_params=None, lora_request=None):
        # with the stage-0 adapter, most validation answers run into the limit
        model.runaway = set(prompts[:4]) if lora_request and "grpo_lora_s0" in lora_request else set()
        return real(prompts, sampling_params, lora_request)

    model.fast_generate = generate
    out = run.train()
    assert out["final_adapter"] is None and len(out["stages"]) == 1 and "rejected" in out["stages"][0]
    assert nb["_evals"] == [] and nb["_restored"][-1] is None            # back to the starting model, no final eval
    files = _files(nb)
    assert not Path(run.adir("grpo_lora_s0")).exists()
    assert [f for f in files if f.startswith("grpo_lora_s0_rejected_")] and [f for f in files if f.startswith("rejected_")]
    assert len(FakeTrainer.created) == 1                                 # stage 1 is not trained on top of a rejected stage


def test_starting_eval_is_reused_or_run_after_training(nb, tmp_path):
    sets = {"gpqa_diamond": [("p1", "A", "k1"), ("p2", "B", "k2")]}
    ev = ru.QuickEval(sets, max_tokens=8192, samples=8)
    nb.update(EVAL=ev, RUN_INIT_EVAL="after", Q=dict(nb["Q"], eval_samples=8))
    nb["quick_eval"] = lambda tag, lora_request=None, samples=None: (nb["_evals"].append((tag, lora_request)),
                                                                     ev.results.setdefault(tag, {}))
    # nothing to reuse: not before training, once after it, and not a second time
    assert grpo_run.init_eval(nb) == "later" and nb["_evals"] == []
    assert grpo_run.init_eval(nb, after_grpo=True) == "evaluated" and nb["_evals"] == [("0_init", None)]
    assert grpo_run.init_eval(nb, after_grpo=True) == "done"

    # the SFT notebook's result is in the neighbouring folder: used straight away, nothing generated
    ev.results.clear()
    nb["_evals"].clear()
    folder = Path(nb["PERSIST"]).parent / "S1_a100"
    folder.mkdir(parents=True)
    (folder / "stage_sft.json").write_text(json.dumps({"steps": 3407, "loss_last": 0.55}))
    (folder / "stage_eval_1_sft.json").write_text(json.dumps({
        "settings": {"samples": 4, "max_tokens": 8192, "temperature": 0.6, "n": {"gpqa_diamond": 2}},
        "results": {"gpqa_diamond": {"n": 2, "acc_flex": 37.5}}, "per_question": {"gpqa_diamond": [0.5, 0.25]}}))
    nb["_init_meta"] = dict(exp="S1", mode="a100", steps=3407, loss_last=0.55)
    assert grpo_run.init_eval(nb) == "reused" and nb["_evals"] == []
    assert ev.results["0_init"]["gpqa_diamond"]["acc_flex"] == 37.5 and ev.samples == 4 and nb["Q"]["eval_samples"] == 4
    assert grpo_run.init_eval(nb, after_grpo=True) == "done" and nb["_evals"] == []
    # switched off: evaluated as before
    ev.results.clear()
    nb["REUSE_INIT_EVAL"] = False
    assert grpo_run.init_eval(nb) == "later"


def test_missing_notebook_state_is_reported_by_name(nb):
    del nb["tokenizer"], nb["Q"]
    with pytest.raises(RuntimeError, match="tokenizer.*Q"):
        grpo_run.GrpoRun(nb)
