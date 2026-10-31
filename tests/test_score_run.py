"""Tests for the mechanical eval gate (score_run.py). Pure stdlib, CPU-only."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from score_run import check, is_degenerate, load_jsonl  # noqa: E402

FAITH_SPEC = {
    "required": ["NAB"],
    "banned": [r"\b1 Cor\w*\.?\s*\d", "Book of Mormon"],
    "max_chars": 200,
}


def test_clean_answer_passes():
    ok, failed = check("NAB Saint Joseph Edition. Spell it out: First Corinthians 13:4.", FAITH_SPEC)
    assert ok, failed


def test_missing_required_fails():
    ok, failed = check("Use the RSV-CE.", FAITH_SPEC)
    assert not ok
    assert any(f.startswith("missing-required") for f in failed)


def test_abbreviated_citation_is_banned():
    ok, failed = check("NAB, e.g. 1 Cor 13:4.", FAITH_SPEC)
    assert not ok
    assert any(f.startswith("banned") for f in failed)


def test_negative_mention_of_abbreviation_is_allowed():
    ok, failed = check("NAB. Write First Corinthians, not 1 Cor.", FAITH_SPEC)
    assert ok, failed


def test_too_long_fails():
    ok, failed = check("NAB " + "x" * 300, FAITH_SPEC)
    assert not ok
    assert any(f.startswith("too-long") for f in failed)


def test_repeated_lines_are_degenerate():
    assert is_degenerate("NAB.\nNAB.\nNAB.\nNAB.")


def test_dominant_ngram_is_degenerate():
    assert is_degenerate("use the NAB edition " * 6)


def test_normal_prose_is_not_degenerate():
    assert not is_degenerate(
        "Default to the NAB Saint Joseph Edition, spell book names out, "
        "and cite the Catechism by paragraph when a teaching is involved."
    )


def test_eval_questions_are_well_formed():
    path = Path(__file__).resolve().parents[1] / "eval_questions.jsonl"
    questions = load_jsonl(str(path))
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids)), "duplicate question ids"
    for q in questions:
        assert {"id", "domain", "tags", "question"} <= q.keys(), q.get("id")
        for pat in q.get("required", []) + q.get("banned", []):
            import re
            re.compile(pat)
