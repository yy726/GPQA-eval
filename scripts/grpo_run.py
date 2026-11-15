"""Section 4 of notebook 11 (GRPO) as a module: prompt pools, held-out validation, pre-filter, reward
functions, the staged training loop with checkpoints / resume / acceptance checks, and the evaluation of
the starting model.

Why it lives here and not in notebook cells: the notebook's cell 3 runs `git pull` on every start, so this
file is always the current version, while a notebook tab that was opened earlier keeps (and, when saved to
GitHub, puts back) its old cells. Fixes made here cannot be undone that way. The notebook keeps only the
switches; this module reads them from the notebook's namespace (`GrpoRun(globals())`) at the moment they
are used, with the defaults below for switches an older notebook copy does not define.

Nothing here needs a GPU to import: vllm / trl / torch / datasets are imported inside the methods.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import statistics
import time
from collections import Counter

import recipe_utils as ru

# Switches read from the notebook (name -> value used when the notebook does not define it).
DEFAULTS = dict(
    RUN_SFT=False, RESUME=True, RUN_GRPO=True, INIT_MODEL=None, REPORT_TO="none",
    RUN_INIT_EVAL="after", REUSE_INIT_EVAL=True, EVAL_SAMPLES=None, RUN_BASE_EVAL=False,
    GRPO_DATA="science", GRPO_FILTER=True, CURRICULUM_ROUNDS=2, DRSCI_SUBJECTS=["physics", "chemistry", "biology"],
    GRPO_LR=1e-5, GRPO_MAX_COMPLETION="auto", EPSILON_HIGH=0.28, FORMAT_WEIGHT=0.5, OVERLONG_CACHE=1024,
    MAX_PROMPT_LEN=1024,   # prompts longer than this are dropped (TRL has no max_prompt_length)
    CKPT_EVERY=5, CKPT_KEEP=3, CLIP_PATIENCE=3, LATER_STAGE_LR=0.5, GRPO_BETA=0.0, SCALE_REWARDS="batch",
    MASK_TRUNCATED=True, MAX_GRAD_NORM=1.0, MIN_STOP_RATE=90.0, CLIP_STOP=0.15,
)
REQUIRED = ("model", "tokenizer", "Q", "CACHE", "PERSIST", "STOP_IDS", "MAX_SEQ_LEN", "RUN_NAME", "quick_eval", "diagnose")


def init_eval(ns: dict, after_grpo: bool = False) -> str:
    """GPQA score of the starting model (stage "0_init"); called before training and again after it.

    1. If the SFT notebook already evaluated this checkpoint (its results are in the persistent directory,
       next to this run's folder), that result is used and nothing is generated (REUSE_INIT_EVAL).
    2. Otherwise RUN_INIT_EVAL decides: "after" = evaluate once GRPO training is done and its adapters are on
       disk, True = before training, False = never.
    Returns what happened: "reused" / "evaluated" / "later" / "skipped" / "base" / "done".
    """
    def k(name):
        return ns.get(name, DEFAULTS[name])

    ev, init_model = ns["EVAL"], k("INIT_MODEL")
    if "0_init" in ev.results:
        return "done"
    if not after_grpo and init_model and not k("RUN_SFT") and k("REUSE_INIT_EVAL"):
        root = os.path.dirname(str(ns["PERSIST"]).rstrip("/"))
        if ru.reuse_sft_eval(ev, "0_init", ns.get("_init_meta"), root, keep_samples=bool(k("EVAL_SAMPLES"))):
            if isinstance(ns.get("Q"), dict):
                ns["Q"]["eval_samples"] = ev.samples
            if ns.get("wandb") is not None:
                ru.log_eval(ns["wandb"], ev.results["0_init"], "0_init")
            return "reused"
    if ru.init_eval_now(k("RUN_INIT_EVAL"), bool(init_model), bool(k("RUN_GRPO")), after_grpo):
        ns["quick_eval"]("0_init")
        return "evaluated"
    if after_grpo:
        return "skipped"
    if init_model and k("RUN_INIT_EVAL"):
        print('[i] RUN_INIT_EVAL="after"：起点的 GPQA 评测放到 GRPO 训完之后做（第 4 节末尾）。')
        return "later"
    if init_model:
        print("[i] RUN_INIT_EVAL=False：跳过起点的 GPQA 评测；第 5 节不会有起点 → GRPO 的配对比较（留出验证集的比较不受影响）。")
        return "skipped"
    if k("RUN_BASE_EVAL"):
        ns["quick_eval"]("0_base", samples=1)   # the base model never stops: one sample per question
        return "base"
    return "skipped"


class GrpoRun:
    """prepare() builds the pools and the held-out set; train() runs the stages and returns the summary."""

    def __init__(self, ns: dict):
        missing = [name for name in REQUIRED if name not in ns]
        if missing:
            raise RuntimeError(f"先运行前面的格：notebook 里还没有 {missing}")
        self.ns = ns
        self.model, self.tokenizer = ns["model"], ns["tokenizer"]
        self.Q, self.cache, self.persist = ns["Q"], ns["CACHE"], str(ns["PERSIST"])
        self.stop_ids, self.max_seq_len = ns["STOP_IDS"], ns["MAX_SEQ_LEN"]
        self.rl_val_results: dict = {}    # tag -> held-out result (the notebook's RL_VAL)
        self._rl_val_pq: dict = {}
        self.diag = {"stage": None, "call": 0, "samples_per_step": 2, "tail_chars": 400}
        self.val_rows: list = []
        self.cap: dict = {}
        self.final_adapter = None
        self.lora_req = None
        self._drsci = None
        self._prepared = False

    # ------------------------------------------------------------------ small helpers

    def k(self, name):
        """A notebook switch, read now (so editing the switch cell and re-running it is enough)."""
        return self.ns.get(name, DEFAULTS[name])

    def adir(self, name: str) -> str:
        return f"{self.persist}/{name}"

    def progress(self, phase: str, **info) -> None:
        ru.write_progress(self.persist, phase, **info)

    def _sampling_params(self, **kw):
        from vllm import SamplingParams

        return SamplingParams(**kw)

    # ------------------------------------------------------------------ prompt pools

    def _mcq_rows(self, source, n, skip=0):
        return [{"prompt": [{"role": "user", "content": r["prompt"]}], "answer": r["answer"], "answer_type": "mcq",
                 "source": r["source"]} for r in ru.load_mcq_rows(source, n, skip=skip)]

    def _drsci_rows(self, n, rng, start_frac=0.0):
        if self._drsci is None:
            from datasets import load_dataset

            self._drsci = load_dataset("MiniByte-666/Dr.SCI", data_files="Dr_SCI_verifiable.parquet",
                                       split="train").shuffle(seed=0)
        ds, out = self._drsci, []
        for row in ds.select(range(int(start_frac * len(ds)), len(ds))):
            r = ru.drsci_numeric_row(row, subjects=self.k("DRSCI_SUBJECTS"), difficulty=rng)
            if r:
                out.append(r)
                if len(out) >= n:
                    break
        return out

    def _fits(self, r) -> bool:
        return ru.n_chat_tokens(self.tokenizer, r["prompt"], add_generation_prompt=True) <= self.max_prompt_len

    def val_pool(self, n: int) -> list:
        """Held-out validation rows: same distribution as the training prompts, never trained on
        (MMLU-Pro from the far end of the shuffled set, Dr.SCI from its last 25%)."""
        if n <= 0:
            return []
        data = self.k("GRPO_DATA")
        if data == "science":
            rows = (self.sci_val[: n // 2] + self._mcq_rows("mmlu_pro_stem", n // 5, skip=10_000)
                    + self._drsci_rows(n - n // 2 - n // 5, (0.25, 0.875), start_frac=0.75))
        elif data == "sciq":
            rows = self._mcq_rows("sciq", n, skip=11_000)
        elif data == "drsci":
            rows = self._drsci_rows(n, (0.25, 0.875), start_frac=0.75)
        elif data == "mix":
            rows = (self._mcq_rows("mmlu_pro_stem", n // 2, skip=10_000)
                    + self._drsci_rows(n - n // 2, (0.25, 0.875), start_frac=0.75))
        else:
            rows = self._mcq_rows("mmlu_pro_stem", n, skip=10_000)
        rows, _ = ru.decontam_rows(rows, lambda r: {"prompt": r["prompt"]}, "grpo_val")
        return [r for r in rows if self._fits(r)]

    def stage_pool(self, stage: int) -> list:
        n, data = self.cand_per_stage, self.k("GRPO_DATA")
        rng = self.drsci_ranges[min(stage, len(self.drsci_ranges) - 1)]
        if data == "science":
            n_sci, n_pro = n // 2, n // 5
            rows = (self.sci_pool[stage * n_sci:(stage + 1) * n_sci]
                    + self._mcq_rows("mmlu_pro_stem", n_pro * (stage + 1))[-n_pro:]
                    + self._drsci_rows(n - n_sci - n_pro, rng))
        elif data == "mix":
            rows = self._mcq_rows("mmlu_pro_stem", n // 2 * (stage + 1))[-(n // 2):] + self._drsci_rows(n - n // 2, rng)
        elif data == "drsci":
            rows = self._drsci_rows(n, rng)
        elif data == "sciq":
            rows = self._mcq_rows("sciq", n * (stage + 1), skip=4000)[-n:]      # the start of SciQ is used for SFT format samples
        else:
            rows = self._mcq_rows("mmlu_pro_stem", n * (stage + 1))[-n:]
        rows = [r for r in rows if r["prompt"][0]["content"] not in self._val_prompts]   # never a validation prompt
        rows, _dropped = ru.decontam_rows(rows, lambda r: {"prompt": r["prompt"]}, f"grpo_pool_s{stage}")
        random.Random(stage).shuffle(rows)
        return rows

    def prepare(self) -> dict:
        """Sizes, completion caps, the MegaScience pools and the held-out validation rows."""
        Q, rounds = self.Q, self.k("CURRICULUM_ROUNDS")
        self.steps_per_stage = max(1, Q["grpo_steps"] // max(1, rounds))
        self.need_per_stage = self.steps_per_stage * Q["grpo_prompts_per_step"]
        # X1b: the pre-filter keeps about 55%; a shortfall is topped up with unfiltered prompts
        self.cand_per_stage = int(self.need_per_stage * (1.7 if self.k("GRPO_FILTER") else 1))
        self.max_prompt_len = self.k("MAX_PROMPT_LEN")
        self.preset_cap = min(Q["grpo_max_completion"], self.max_seq_len - self.max_prompt_len)
        # updated before every stage from the lengths the model stops at by itself
        self.cap = {"max": self.preset_cap, "cache": min(self.k("OVERLONG_CACHE"), self.preset_cap // 4)}
        # MegaScience 4-choice science questions held out from SFT (same script and question-hash split as
        # notebook 15, so the SFT model has not seen their solutions)
        self.sci_pool, self.sci_val = [], []
        if self.k("GRPO_DATA") == "science":
            from science_mix import load_mcq_pool

            if not os.path.exists("data/s1/rl_mcq.jsonl"):   # downloads MegaScience (about 1.9 GB), about 3 minutes
                ru.run_script(["scripts/science_mix.py", "--out-dir", "data/s1", "--pools-only"], show_tail=40)
            self.sci_pool, self.sci_val = load_mcq_pool("data/s1/rl_mcq.jsonl"), load_mcq_pool("data/s1/val_mcq.jsonl")
            print("science MCQ pool:", len(self.sci_pool), "| validation:", len(self.sci_val))
        self.drsci_ranges = [(0.5, 0.875), (0.25, 0.375)] if rounds > 1 else [(0.25, 0.875)]
        self.val_rows = self.val_pool(Q["val_n"])
        self._val_prompts = {r["prompt"][0]["content"] for r in self.val_rows}
        self._prepared = True
        info = dict(stages=rounds, steps_per_stage=self.steps_per_stage, prompts_per_step=Q["grpo_prompts_per_step"],
                    prompts_per_stage=self.need_per_stage, candidates_per_stage=self.cand_per_stage,
                    rollouts_total=rounds * self.need_per_stage * Q["num_gen"], preset_cap=self.preset_cap,
                    val_prompts=len(self.val_rows))
        print(info)
        return info

    # ------------------------------------------------------------------ sampling: validation and pre-filter

    def _sample(self, rows, lora_request, k, max_tokens, seed):
        sp = self._sampling_params(n=k, temperature=1.0, top_p=1.0, max_tokens=max_tokens, seed=seed,
                                   stop_token_ids=self.stop_ids, stop=ru.STOP_STRS)
        outs = self.model.fast_generate([ru.chat_prompt(self.tokenizer, r["prompt"][0]["content"]) for r in rows],
                                        sampling_params=sp, lora_request=lora_request)
        return ru.pass_stats(outs, rows)

    def rl_val(self, tag: str, lora_request=None) -> dict:
        """Pass rate (right and finished) on the held-out prompts, with the paired comparison against the
        previous measurement. A finished measurement is read back after a disconnect."""
        if not self.val_rows:
            return {}
        name = f"rl_val_{tag}"
        # "guard": only results measured with the leftover-request guard on model.fast_generate are read back;
        # older ones may contain the outputs of an interrupted eval and are measured again (about 3 minutes)
        if self.cache.has(name):
            saved = self.cache.load(name)
            if saved.get("prompts") == len(self.val_rows) and saved.get("guard"):
                self.rl_val_results[tag], self._rl_val_pq[tag] = saved["st"], saved["pq"]
                print("rl_val", tag, "(cached)", saved["st"])
                return saved["st"]
        t0 = time.time()
        self.progress(f"rl_val:{tag}", prompts=len(self.val_rows))
        st = self._sample(self.val_rows, lora_request, 4, self.Q["filter_max_tokens"], seed=11)
        pq = [c / st["samples"] for c in st.pop("correct_counts")]
        st.pop("stopped_lengths")
        st["minutes"] = round((time.time() - t0) / 60, 1)
        if self._rl_val_pq:
            prev = list(self._rl_val_pq)[-1]
            st["vs_" + prev] = ru.paired_delta(self._rl_val_pq[prev], pq)
            first = list(self._rl_val_pq)[0]
            if first != prev:
                st["vs_" + first] = ru.paired_delta(self._rl_val_pq[first], pq)
        self.rl_val_results[tag], self._rl_val_pq[tag] = st, pq
        self.cache.save(name, {"st": st, "pq": pq, "prompts": len(self.val_rows), "guard": 1})
        print("rl_val", tag, st)
        wandb = self.ns.get("wandb")
        if wandb is not None and getattr(wandb, "run", None) is not None:
            wandb.run.summary[f"rl_val/{tag}/pass_rate"] = st["pass_rate"]
        return st

    def pass_filter(self, rows, lora_request, need):
        """Sample every candidate k times with the current model and keep the prompts answered right 1..k-1
        times (a group that is all right or all wrong carries no learning signal)."""
        k = self.Q["filter_k"]
        t0 = time.time()
        self.progress("pass_filter", prompts=len(rows))
        st = self._sample(rows, lora_request, k, self.Q["filter_max_tokens"], seed=7)
        counts, lengths = st.pop("correct_counts"), st.pop("stopped_lengths")
        keep = [r for r, c in zip(rows, counts) if 0 < c < k]
        info = dict(st, candidates=len(rows), kept=len(keep), minutes=round((time.time() - t0) / 60, 1))
        print("pass filter:", info)
        if len(keep) < need:
            print(f"[!] only {len(keep)} prompts passed the filter (need {need}); topping up with unfiltered ones")
            kept_ids = {id(r) for r in keep}
            keep += [r for r in rows if id(r) not in kept_ids][: need - len(keep)]
        return keep[:need], info, lengths

    # ------------------------------------------------------------------ rewards and per-step diagnostics

    def _n_tokens(self, completions, completion_ids):
        if completion_ids is None:
            completion_ids = [self.tokenizer(c[0]["content"], add_special_tokens=False)["input_ids"] for c in completions]
        return [len(ids) for ids in completion_ids]

    def _diag(self, completions, answer, answer_type, lengths, cut, log_metric=None):
        """Per step: how answers end and which are right, to W&B (diag/...) and to grpo_diag_s<stage>.jsonl
        in the persistent directory. Prompts are public training data (not GPQA); answer tails stay on Drive."""
        diag, cap = self.diag, self.cap
        try:
            texts = [c[0]["content"] for c in completions]
            n = max(1, len(texts))
            raw = [ru.correctness_score(x, a, t) > 0 for x, a, t in zip(texts, answer, answer_type)]   # ignores how it ended
            has_answer = [ru.correctness_score(x, a, t) >= 0 for x, a, t in zip(texts, answer, answer_type)]
            stopped_len = sorted(length for length, x in zip(lengths, cut) if not x)
            m = {
                "diag/stopped_rate": sum(not x for x in cut) / n,
                "diag/correct_and_stopped": sum(r and not x for r, x in zip(raw, cut)) / n,
                "diag/truncated_but_correct": sum(r and x for r, x in zip(raw, cut)) / n,   # rising = learning not to stop
                "diag/no_answer_rate": sum(not h for h in has_answer) / n,
                "diag/mean_len_stopped": sum(stopped_len) / max(1, len(stopped_len)),
                "diag/p90_len_stopped": stopped_len[int(0.9 * (len(stopped_len) - 1))] if stopped_len else 0.0,
                "diag/max_len": max(lengths) if lengths else 0.0,
                "diag/cap": cap["max"],
            }
            # one group per prompt (num_gen consecutive rollouts): a group whose rollouts are all right or all
            # wrong has no correctness signal. With SCALE_REWARDS="batch" TRL's frac_reward_zero_std is no
            # longer per group; read this instead.
            g = self.Q["num_gen"]
            ok = [r and not x for r, x in zip(raw, cut)]
            groups = [ok[i:i + g] for i in range(0, len(ok), g)]
            m["diag/groups_without_signal"] = sum(len(set(grp)) == 1 for grp in groups) / max(1, len(groups))
            # these groups get no gradient under MASK_TRUNCATED
            m["diag/groups_all_truncated"] = sum(all(cut[i:i + g]) for i in range(0, len(cut), g)) / max(1, len(groups))
            for kind in ("mcq", "numeric"):
                idx = [i for i, t in enumerate(answer_type) if t == kind]
                if idx:
                    m[f"diag/{kind}_correct_and_stopped"] = sum(raw[i] and not cut[i] for i in idx) / len(idx)
            if callable(log_metric):
                for key, v in m.items():
                    log_metric(key, float(v))
            diag["call"] += 1
            # the longest rollouts show a model that stops finishing / starts repeating first
            order = sorted(range(len(texts)), key=lambda i: -lengths[i])[: diag["samples_per_step"]]
            rec = dict(stage=diag["stage"], call=diag["call"], time=time.strftime("%H:%M:%S"),
                       **{key: round(float(v), 4) for key, v in m.items()},
                       samples=[dict(tokens=lengths[i], truncated=bool(cut[i]), correct=bool(raw[i]),
                                     answer_type=answer_type[i], tail=texts[i][-diag["tail_chars"]:]) for i in order])
            with open(f"{self.persist}/grpo_diag_s{diag['stage']}.jsonl", "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:      # diagnostics must never affect training
            if diag["call"] < 3:
                print("diag skipped:", type(e).__name__, e)

    def reward_funcs(self) -> list:
        """[format_reward, correctness_reward, overlong_reward] as plain functions (TRL logs them by __name__).

        A rollout that fills the cap (cut off) scores as "no answer" for correctness, 0 for format, and gets
        the overlong penalty -1. In run R1 a cut-off rollout that had mentioned the right letter earned +1,
        more than a finished wrong answer (0), and the model learned not to stop.
        """
        run = self

        def format_reward(completions, answer_type, completion_ids=None, **kwargs):
            cut = [n >= run.cap["max"] for n in run._n_tokens(completions, completion_ids)]
            return [ru.format_score(c[0]["content"], t, run.k("FORMAT_WEIGHT"), truncated=x)
                    for c, t, x in zip(completions, answer_type, cut)]

        def correctness_reward(completions, answer, answer_type, completion_ids=None, log_metric=None, **kwargs):
            lengths = run._n_tokens(completions, completion_ids)
            cut = [n >= run.cap["max"] for n in lengths]
            run._diag(completions, answer, answer_type, lengths, cut, log_metric)
            return [ru.correctness_score(c[0]["content"], a, t, truncated=x)
                    for c, a, t, x in zip(completions, answer, answer_type, cut)]

        def overlong_reward(completions, completion_ids=None, **kwargs):
            return [ru.soft_overlong_penalty(n, run.cap["max"], run.cap["cache"])
                    for n in run._n_tokens(completions, completion_ids)]

        return [format_reward, correctness_reward, overlong_reward]

    # ------------------------------------------------------------------ the staged training loop

    def _plan_path(self, stage: int) -> str:
        return f"{self.persist}/grpo_plan_s{stage}.json"

    def _archive(self, stage: int, cur_dir: str) -> None:
        """A rejected stage: adapter, results, checkpoints and prompt list are renamed (kept, not deleted),
        so the stage is trained again next time and what went wrong can still be inspected."""
        tag = time.strftime("%m%d_%H%M%S")
        if os.path.isdir(cur_dir):
            os.rename(cur_dir, f"{cur_dir}_rejected_{tag}")
        for name in (f"grpo_s{stage}", f"rl_val_after_s{stage}"):
            p = self.cache.path(name)
            if p.exists():
                p.rename(p.with_name(f"rejected_{tag}_{p.name}"))
        ckpt_dir = self.adir(f"grpo_ckpt_s{stage}")
        if os.path.isdir(ckpt_dir):       # may hold the steps before a collapse
            os.rename(ckpt_dir, f"{ckpt_dir}_rejected_{tag}")
        if os.path.exists(self._plan_path(stage)):
            os.rename(self._plan_path(stage), f"{self.persist}/grpo_plan_s{stage}_rejected_{tag}.json")
        self.rl_val_results.pop(f"after_s{stage}", None)
        self._rl_val_pq.pop(f"after_s{stage}", None)

    @staticmethod
    def _avg(log, key, part):
        vals = [h[key] for h in log if key in h]
        n = max(1, len(vals) // 5)
        return round(float(statistics.fmean(vals[:n] if part == "first" else vals[-n:])), 3) if vals else None

    def train(self) -> dict:
        """All stages. Every call starts from the starting adapter; finished and accepted stages are read back
        from the persistent directory (RESUME), a stage that was cut short continues from its newest readable
        checkpoint. Returns the summary dict (the notebook's GRPO); self.final_adapter is the folder name of
        the last accepted stage (None when no stage was accepted)."""
        import torch
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer

        if not self._prepared:
            self.prepare()
        model, Q, k = self.model, self.Q, self.k
        cache, cap = self.cache, self.cap
        num_gen, pps = Q["num_gen"], Q["grpo_prompts_per_step"]
        min_stop_rate = k("MIN_STOP_RATE")
        vllm_sampling_params = self._sampling_params(min_p=0.1, top_p=1.0, top_k=-1, seed=3407,
                                                     stop_token_ids=self.stop_ids, stop=ru.STOP_STRS)
        # starting point: the SFT adapter of this notebook (RUN_SFT), otherwise a fresh LoRA (= INIT_MODEL itself)
        prev_dir = self.adir("sft_lora") if k("RUN_SFT") else None
        print("start adapter:", ru.restore_lora(model, prev_dir))
        lora_req = model.load_lora(prev_dir) if k("RUN_SFT") else None
        stages = []
        self.rl_val_results.clear()
        self._rl_val_pq.clear()
        self.rl_val("before_grpo", lora_req)
        # the starting model's own stop rate: acceptance and early stop are both relative to it
        base_stop = (self.rl_val_results.get("before_grpo") or {}).get("stop_rate")
        print("stop rate before GRPO:", base_stop, "| stage is rejected below:",
              min_stop_rate if base_stop is None else round(min(min_stop_rate, base_stop - 5.0), 1))
        # the cap only goes down: R1's stage 2 raised it from 3584 to 5632 and gave the divergence more room
        cap_limit = self.preset_cap

        for stage in range(k("CURRICULUM_ROUNDS")):
            cur_dir, ckpt_dir = self.adir(f"grpo_lora_s{stage}"), self.adir(f"grpo_ckpt_s{stage}")
            if cache.has(f"grpo_s{stage}") and os.path.exists(f"{cur_dir}/adapter_model.safetensors"):
                s = cache.load(f"grpo_s{stage}")
                print(f"[i] 第 {stage} 段已经完成，读回 adapter：", ru.restore_lora(model, cur_dir))
                cand_req = model.load_lora(cur_dir)
                val = self.rl_val(f"after_s{stage}", cand_req)
                ok, why = ru.stage_accept(val, min_stop_rate, baseline_stop_rate=base_stop)
                if ok:
                    s["rl_val"] = val
                    stages.append(s)
                    prev_dir, lora_req = cur_dir, cand_req
                    cap_limit = min(cap_limit, int(s.get("max_completion") or cap_limit))
                    continue
                print(f"[!] 已保存的第 {stage} 段不合格（{why}）：改名留档，用现在的设置重训这一段。")
                self._archive(stage, cur_dir)
                print("back to:", prev_dir or "起点", ru.restore_lora(model, prev_dir))

            # ---- this stage has to be trained: is there a checkpoint of an interrupted attempt?
            ckpt, done_steps, info = None, 0, None
            if k("RESUME") and os.path.exists(self._plan_path(stage)):
                # newest first: a checkpoint cut short by a dying runtime does not load, the one before it is used
                for path, step in ru.list_grpo_checkpoints(ckpt_dir):
                    try:
                        info = ru.restore_lora(model, path)
                        ckpt, done_steps = path, step
                        break
                    except Exception as e:
                        print(f"[!] 检查点 {path} 读不出来（{type(e).__name__}: {e}），试前一个")
                if not ckpt:   # no usable checkpoint: the stage starts over
                    print("back to:", prev_dir or "起点", ru.restore_lora(model, prev_dir))
            if ckpt:
                plan = json.load(open(self._plan_path(stage)))
                rows, filt, steps = plan["rows"], plan["filter"], plan["steps"]
                cap.update(plan["cap"])
                print(f"[i] 第 {stage} 段从检查点接着训：{ckpt}（已完成 {done_steps}/{steps} 步）", info)
            else:
                done_steps = 0
                rows = [r for r in self.stage_pool(stage) if self._fits(r)]
                filt, lengths = {}, []
                if k("GRPO_FILTER"):
                    rows, filt, lengths = self.pass_filter(rows, lora_req, self.need_per_stage)
                else:
                    rows = rows[:self.need_per_stage]
                # cap: 2 x the 99th percentile of the lengths the model stops at by itself (an answer that never
                # stops makes the whole step wait until it fills the cap), and never above the previous stage's cap
                if k("GRPO_MAX_COMPLETION") == "auto":
                    cap["max"] = min(cap_limit, ru.suggest_completion_cap(lengths, self.preset_cap))
                else:
                    cap["max"] = min(cap_limit, int(k("GRPO_MAX_COMPLETION")), self.max_seq_len - self.max_prompt_len)
                cap["cache"] = min(k("OVERLONG_CACHE"), cap["max"] // 4)
                steps = max(1, min(self.steps_per_stage, len(rows) // pps))
                rows = [{key: r[key] for key in ("prompt", "answer", "answer_type")} for r in rows]
                shutil.rmtree(ckpt_dir, ignore_errors=True)          # checkpoints of an earlier attempt belong to other prompts
                os.makedirs(ckpt_dir, exist_ok=True)
                json.dump(dict(rows=rows, filter=filt, steps=steps, cap=dict(cap)), open(self._plan_path(stage), "w"))
            left = steps - done_steps
            print(dict(stage=stage, prompts=len(rows), steps=steps, done=done_steps, max_completion=cap["max"],
                       overlong_cache=cap["cache"]))
            s = dict(stage=stage, steps=done_steps, prompts=len(rows), by_type=dict(Counter(r["answer_type"] for r in rows)),
                     filter=filt, max_completion=cap["max"], resumed_from=done_steps,
                     scale_rewards=k("SCALE_REWARDS"), mask_truncated=k("MASK_TRUNCATED"), beta=k("GRPO_BETA"))
            if left > 0:
                clip_stop = ru.clip_stop_threshold(k("CLIP_STOP"), filt.get("stop_rate", base_stop) if isinstance(filt, dict) else base_stop)
                print("stage stops early when clipped_ratio >=", clip_stop, "for", k("CLIP_PATIENCE"), "steps in a row")
                guard = ru.TruncationGuard(clip_stop, k("CLIP_PATIENCE"))
                self.diag["stage"] = stage
                ds = Dataset.from_list(rows[done_steps * pps:])        # shuffle_dataset=False: prompts in order, a resume skips the used ones
                args = GRPOConfig(**ru.config_kwargs(
                    GRPOConfig,
                    vllm_sampling_params=vllm_sampling_params,
                    generation_kwargs={"stop_token_ids": self.stop_ids},
                    temperature=1.0,
                    learning_rate=k("GRPO_LR") * (1.0 if stage == 0 else k("LATER_STAGE_LR")),
                    weight_decay=0.001,
                    warmup_steps=max(1, left // 10),                # every stage (and every resume) warms up: the optimizer is new
                    lr_scheduler_type="constant_with_warmup",      # OLMo 3 RL: constant learning rate
                    optim="adamw_8bit",
                    logging_steps=1,
                    per_device_train_batch_size=num_gen,            # one micro-batch = the num_gen rollouts of one prompt
                    gradient_accumulation_steps=pps,                # one step = this many prompts (one generation, one update)
                    num_generations=num_gen,
                    max_completion_length=cap["max"],
                    epsilon_high=k("EPSILON_HIGH"),                 # clip-higher (DAPO / OLMo 3); loss_type stays Unsloth's default
                    beta=k("GRPO_BETA"),
                    scale_rewards=k("SCALE_REWARDS"),
                    mask_truncated_completions=k("MASK_TRUNCATED"),
                    max_grad_norm=k("MAX_GRAD_NORM"),
                    max_steps=left,
                    shuffle_dataset=False,
                    save_strategy="steps",
                    save_steps=k("CKPT_EVERY"),                     # adapter to the persistent directory every N steps
                    save_total_limit=k("CKPT_KEEP"),
                    save_only_model=True,                           # adapter only (a resume uses a new optimizer + warmup)
                    log_completions=False,                          # generated text is never uploaded to W&B
                    report_to=k("REPORT_TO"),
                    run_name=f"{self.ns['RUN_NAME']}-grpo-s{stage}" + (f"-from{done_steps}" if done_steps else ""),
                    output_dir=f"{ckpt_dir}/from_{done_steps}",
                ))
                trainer = GRPOTrainer(model=model, processing_class=self.tokenizer, reward_funcs=self.reward_funcs(),
                                      args=args, train_dataset=ds)

                def on_step(step, logs, _stage=stage, _done=done_steps, _of=steps):
                    self.progress(f"grpo_s{_stage}_train", step=_done + step, of=_of, reward=logs.get("reward"),
                                  clipped_ratio=logs.get("completions/clipped_ratio"))

                trainer.add_callback(ru.truncation_guard_callback(guard, on_step=on_step))
                try:
                    stats = trainer.train()
                except BaseException as e:
                    # manual interrupt / out of memory / CUDA error: save the adapter in memory as a checkpoint,
                    # then stop; the next Run all continues from this step. (A Colab disconnect or used-up
                    # compute units never get here: the periodic saves cover that.)
                    gs = trainer.state.global_step
                    ru.abort_unfinished_requests(model)             # no half-generated rollouts left in the engine
                    try:
                        saved = ru.save_emergency_checkpoint(model, f"{ckpt_dir}/from_{done_steps}", gs)
                    except BaseException as e2:
                        saved = None
                        print(f"[!] 中断时存盘失败（{type(e2).__name__}: {e2}）")
                    last = ru.list_grpo_checkpoints(ckpt_dir)
                    print(f"[!] 第 {stage} 段在第 {done_steps + gs}/{steps} 步停下（{type(e).__name__}）。",
                          f"已存到 {saved}" if saved else "这一步没有存盘。",
                          f"Drive 上最新的检查点：{last[0] if last else '没有'}。",
                          "重启 runtime 后 Run all（RESUME=True）会从它接着训。")
                    self.progress(f"grpo_s{stage}_stopped", step=done_steps + gs, of=steps, error=type(e).__name__)
                    raise
                log = trainer.state.log_history
                s.update(steps=done_steps + trainer.state.global_step, minutes=round(stats.metrics["train_runtime"] / 60, 1),
                         stopped_early=guard.tripped)
                for key in ["reward", "rewards/format_reward/mean", "rewards/correctness_reward/mean",
                            "rewards/overlong_reward/mean", "completions/mean_length", "completions/clipped_ratio",
                            "frac_reward_zero_std"]:
                    if any(key in h for h in log):
                        s[key] = {"first": self._avg(log, key, "first"), "last": self._avg(log, key, "last")}
                del trainer, ds
                torch.cuda.empty_cache()
            model.save_lora(cur_dir)                               # straight into the persistent directory
            cand_req = model.load_lora(cur_dir)
            if prev_dir:
                try:
                    s["adapter_change"] = ru.lora_delta(prev_dir, cur_dir)   # how far this stage moved the LoRA weights
                except Exception as e:
                    print("adapter change skipped:", type(e).__name__, e)
            cache.save(f"grpo_s{stage}", s)                        # this stage is done: not retrained after a disconnect
            val = self.rl_val(f"after_s{stage}", cand_req)
            ok, why = ru.stage_accept(val, min_stop_rate, baseline_stop_rate=base_stop)
            s["rl_val"] = val
            if not ok:
                # this stage damaged the model: not used. Back to the previous adapter; later stages are not
                # trained either (they would start from this one).
                print(f"[!] 第 {stage} 段不合格（{why}）：改名留档，退回", prev_dir or "起点")
                self._archive(stage, cur_dir)
                print("restored:", ru.restore_lora(model, prev_dir))
                s["rejected"] = why
                stages.append(s)
                print(s)
                break
            prev_dir, lora_req = cur_dir, cand_req
            cap_limit = min(cap_limit, cap["max"])
            shutil.rmtree(ckpt_dir, ignore_errors=True)          # the adapter is in grpo_lora_s<stage>; checkpoints no longer needed
            if os.path.exists(self._plan_path(stage)):
                os.remove(self._plan_path(stage))
            stages.append(s)
            print(s)

        final = os.path.basename(prev_dir) if (prev_dir and os.path.basename(prev_dir).startswith("grpo_lora_s")) else None
        print("final adapter:", final, ru.restore_lora(model, prev_dir))   # weights in memory = the last accepted stage
        self.final_adapter, self.lora_req = final, lora_req
        summary = dict(stages=stages, final_adapter=final, grpo_data=k("GRPO_DATA"), prompts_per_step=pps, lr=k("GRPO_LR"),
                       later_stage_lr=k("LATER_STAGE_LR"), rl_val=self.rl_val_results)
        self.progress("grpo_done", final_adapter=final)
        if "EVAL" in self.ns:
            init_eval(self.ns, after_grpo=True)   # the starting model (no adapter), now that the training result is on disk
        if final:
            tag = "2_grpo_" + final.rsplit("_", 1)[-1]           # eval results are cached per adapter (2_grpo_s0 / 2_grpo_s1)
            self.ns["quick_eval"](tag, lora_req)
            self.ns["diagnose"](tag, lora_req)
        else:
            print("[!] 没有一个 GRPO 阶段合格：不做最终评测，第 6 节也不会推送。")
        return summary
