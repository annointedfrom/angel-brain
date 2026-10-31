"""Training renders grounded examples exactly like brain_voice serves them."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import grounding  # noqa: E402
from build_grounded_dataset import ABSTAIN_REPLIES, build, distractors  # noqa: E402

BRAIN_VOICE = ROOT.parents[1] / "NeuroLinked" / "brain-voice"


def test_plain_examples_render_as_before():
    msgs = grounding.chat_messages({"instruction": "Q", "input": "", "output": "A"})
    assert msgs == [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]


def test_grounded_example_has_system_and_memory_block():
    msgs = grounding.chat_messages({"instruction": "Q", "memories": ["m1"], "system": "S", "output": "A"})
    assert msgs[0] == {"role": "system", "content": "S"}
    assert msgs[1]["content"] == "=== Recalled memories ===\n- m1\n\n=== Question ===\nQ"


def test_empty_memories_placeholder():
    assert grounding.user_turn("Q", []).startswith("=== Recalled memories ===\n(no relevant memories found)")


@pytest.mark.skipif(not (BRAIN_VOICE / "brain_voice.py").is_file(), reason="brain_voice not present")
def test_training_format_matches_serving():
    sys.path.insert(0, str(BRAIN_VOICE))
    import brain_voice as bv
    mems = ["short", "x" * 5000, "y" * 9000]
    assert grounding.memory_block(mems) == bv.memory_block(mems)
    assert (grounding.MEMORY_CHARS_EACH, grounding.MEMORY_CHARS_TOTAL) == (bv.MEMORY_CHARS_EACH, bv.MEMORY_CHARS_TOTAL)


def test_distractors_avoid_target_vocabulary():
    import random
    pool = ["NAB Saint Joseph Edition for scripture", "Commits use type description format",
            "Sandbox holds running code", "NAB edition scripture citations spelled out"]
    picked = distractors("NAB Saint Joseph Edition for scripture", pool, 3, random.Random(0))
    assert "NAB edition scripture citations spelled out" not in picked


@pytest.mark.skipif(not (ROOT / "gold_identity.jsonl").is_file(), reason="private gold sets not present")
def test_build_pairs_positive_and_abstain():
    rows = build("S", seed=0, k=4)
    positives = [r for r in rows if any(r["output"] in m for m in r["memories"])]
    abstains = [r for r in rows if r["output"] in ABSTAIN_REPLIES]
    assert positives and abstains
    assert all(not any(r["output"] in m for m in r["memories"]) for r in abstains)
    assert all(r["system"] == "S" for r in rows)
