"""CPU-only tests for scripts/teacher_answers.py (answer-level distillation data). No GPQA data is used."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import recipe_utils as ru  # noqa: E402
import teacher_answers as ta  # noqa: E402


def test_strip_think():
    assert ta.strip_think("<think>\nlong reasoning (B)\n</think>\n\nThe answer is (A).") == "The answer is (A)."
    assert ta.strip_think("reasoning without an opening tag</think>\nFinal: (C)") == "Final: (C)"
    assert ta.strip_think("  no think block  ") == "no think block"


def test_teacher_letter_reads_common_answer_styles():
    assert ta.teacher_letter("First (A) looks right, but actually (C).") == "C"           # last parenthesised letter
    assert ta.teacher_letter("So we get\n\n\\boxed{B}") == "B"
    assert ta.teacher_letter("Therefore the answer is D.") == "D"
    assert ta.teacher_letter("Reasoning...\n\n**Answer:** B") == "B"
    assert ta.teacher_letter("Reasoning...\nFinal Answer: **C**") == "C"
    assert ta.teacher_letter("Hence option D is correct.") == "D"
    assert ta.teacher_letter("The result is 42 joules.") is None


def test_finalize_keeps_only_gold_and_ends_with_the_eval_sentence():
    out = ta.finalize("<think>hmm</think>Step 1 ... Step 2 ...\n\n**Answer:** \\boxed{B}", "B")
    assert out == "Step 1 ... Step 2 ...\n\nThe answer is (B)."
    assert ta.finalize("Reasoning.\n\nThe answer is (B).", "B") == "Reasoning.\n\nThe answer is (B)."   # already in the format
    kept = ta.finalize("Reasoning shows option D is correct.", "D")
    assert kept.endswith("The answer is (D).") and kept.startswith("Reasoning shows")
    assert ta.finalize("Reasoning.\n\nThe answer is (B).", "C") is None                  # wrong letter: dropped
    assert ta.finalize("No letter at all.", "A") is None and ta.finalize("<think>only thoughts</think>", "A") is None
    score = ru.score_completion(kept, "D")
    assert score["strict"] and score["flex"]
