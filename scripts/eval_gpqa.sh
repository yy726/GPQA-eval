#!/usr/bin/env bash
# Locked GPQA evaluation for the T4 post-training hackathon.
# 锁定版 GPQA 评测脚本：所有队伍、所有复评都必须用这份配置。
#
# Usage / 用法:
#   bash scripts/eval_gpqa.sh <hf_model_id_or_local_path> [out_dir] [seeds] [backend]
#
# Examples:
#   bash scripts/eval_gpqa.sh Qwen/Qwen3-4B-Base
#   bash scripts/eval_gpqa.sh my-team/qwen3-4b-sft results/my-team "0 1 2 3" vllm
#   bash scripts/eval_gpqa.sh my-team/qwen3-4b-sft results/quick "0" hf   # HF fallback, slow
#   LIMIT=20 bash scripts/eval_gpqa.sh Qwen/Qwen3-4B-Base results/smoke "0"  # timing only; NOT valid for the leaderboard
#
# Notes:
# * GPQA is gated: accept the terms at https://huggingface.co/datasets/Idavidrein/gpqa
#   and run `hf auth login` (or set HF_TOKEN) first.
# * The GPQA terms ask that examples not be revealed online. `--log_samples` writes model
#   outputs that contain the questions; keep `results/` OUT of any public repo (it is in
#   .gitignore). Only the aggregated submission JSON is meant to be shared.
# * Chat template: the model's own default template is used (`--apply_chat_template`). For the
#   official Qwen/Qwen3-4B reference this means thinking mode is ON by default; with the locked
#   4096-token budget some long thoughts get cut off. That is intended: everyone gets the same budget.
# * Only the vLLM sampling seed changes between runs. lm-eval's own `--seed` (which controls
#   the answer-choice shuffle in the GPQA task) stays fixed at 0, so every team sees the same
#   prompts in the same order.
set -euo pipefail

MODEL="${1:?usage: eval_gpqa.sh <model> [out_dir] [seeds] [backend]}"
OUT_DIR="${2:-results/$(echo "$MODEL" | tr '/' '__')}"
SEEDS="${3:-${SEEDS:-0 1 2 3}}"
BACKEND="${4:-${BACKEND:-vllm}}"

# ---- Locked settings (do not change for leaderboard submissions) ----
TASKS="gpqa_diamond_cot_zeroshot,gpqa_main_cot_zeroshot"
MAX_MODEL_LEN=8192
MAX_GEN_TOKS=4096
TEMPERATURE=0.6
TOP_P=0.95
DTYPE=float16           # T4 (sm75) has no native bf16
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
# ---------------------------------------------------------------------

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  echo "WARNING: LIMIT=${LIMIT} -> partial eval for smoke testing only; scores are not valid for the leaderboard."
  LIMIT_ARGS=(--limit "$LIMIT")
fi

GEN_KWARGS="max_gen_toks=${MAX_GEN_TOKS},temperature=${TEMPERATURE},top_p=${TOP_P},do_sample=True"

mkdir -p "$OUT_DIR"
echo "Model:   $MODEL"
echo "Backend: $BACKEND"
echo "Seeds:   $SEEDS"
echo "Tasks:   $TASKS"
echo "Out:     $OUT_DIR"

for SEED in $SEEDS; do
  echo "=== seed ${SEED} ==="
  if [[ "$BACKEND" == "vllm" ]]; then
    MODEL_ARGS="pretrained=${MODEL},dtype=${DTYPE},max_model_len=${MAX_MODEL_LEN},max_gen_toks=${MAX_GEN_TOKS},gpu_memory_utilization=${GPU_MEM_UTIL},enforce_eager=True,seed=${SEED}"
    BATCH="auto"
  elif [[ "$BACKEND" == "hf" ]]; then
    # HF fallback: much slower. Sampling seed is set through lm-eval's torch seed.
    MODEL_ARGS="pretrained=${MODEL},dtype=${DTYPE},max_length=${MAX_MODEL_LEN}"
    BATCH="8"
  else
    echo "unknown backend: $BACKEND (use vllm or hf)" >&2
    exit 1
  fi

  # --seed python,numpy,torch,fewshot : python stays 0 so the choice shuffle is identical.
  lm_eval \
    --model "$BACKEND" \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASKS" \
    --apply_chat_template \
    --gen_kwargs "$GEN_KWARGS" \
    --batch_size "$BATCH" \
    --seed "0,0,${SEED},0" \
    --output_path "${OUT_DIR}/seed${SEED}" \
    --log_samples \
    "${LIMIT_ARGS[@]}"
done

echo
echo "Done. Aggregate with:"
echo "  python scripts/aggregate_results.py ${OUT_DIR} --model ${MODEL} --backend ${BACKEND} --out submission.json"
