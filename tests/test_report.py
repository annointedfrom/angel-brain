"""Tests for repeated-run reporting (report.py) and grader calibration (calibrate_judge.py)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibrate_judge import agreement, cohens_kappa  # noqa: E402
from report import build_report  # noqa: E402
from score_run import load_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = {
    "a": {"id": "a", "domain": "x", "tags": ["fact"], "required": ["yes"]},
    "b": {"id": "b", "domain": "x", "tags": ["fact"], "required": ["yes"]},
    "c": {"id": "c", "domain": "y", "tags": ["voice"], "required": ["yes"]},
}


def write_answers(tmp_path: Path, name: str, passing: set[str]) -> str:
    path = tmp_path / name
    path.write_text("".join(json.dumps({"id": q, "answer": "yes" if q in passing else "no"}) + "\n"
                            for q in QUESTIONS), encoding="utf-8")
    return str(path)


def test_report_averages_runs_and_bounds_ci(tmp_path):
    runs = [write_answers(tmp_path, "r1.jsonl", {"a", "b"}), write_answers(tmp_path, "r2.jsonl", {"a"})]
    rep = build_report(QUESTIONS, runs, None, None, iters=500, seed=0)
    overall = rep["overall"]
    assert overall["runs"] == 2
    assert overall["pass_rate"] == 0.5          # a=1.0, b=0.5, c=0.0
    assert overall["run_spread"] == [0.333, 0.667]
    assert 0.0 <= overall["ci95"][0] <= overall["pass_rate"] <= overall["ci95"][1] <= 1.0
    assert rep["always_fail"] == ["c"]


def test_report_tag_filter_and_domains(tmp_path):
    runs = [write_answers(tmp_path, "r1.jsonl", {"a", "c"})]
    rep = build_report(QUESTIONS, runs, None, "fact", iters=200, seed=0)
    assert rep["overall"]["n_questions"] == 2
    assert set(rep["by_domain"]) == {"x"}


def test_report_domain_filter_and_empty_selection(tmp_path):
    import pytest
    runs = [write_answers(tmp_path, "r1.jsonl", {"c"})]
    assert build_report(QUESTIONS, runs, None, None, 200, 0, domain="y")["overall"]["pass_rate"] == 1.0
    with pytest.raises(ValueError):
        build_report(QUESTIONS, runs, None, "identity", 200, 0)


def test_paired_compare_detects_identical_systems(tmp_path):
    a = [write_answers(tmp_path, "a.jsonl", {"a"})]
    b = [write_answers(tmp_path, "b.jsonl", {"a"})]
    diff = build_report(QUESTIONS, a, b, None, iters=200, seed=0)["compare"]["diff_b_minus_a"]
    assert diff["diff"] == 0.0 and not diff["significant"]


def test_abstain_rates_split_answerable_and_adversarial(tmp_path):
    from report import abstain_rates
    qs = {"a": {"id": "a", "domain": "x", "tags": ["fact"]},
          "b": {"id": "b", "domain": "x", "tags": ["adversarial"]}}
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps({"id": "a", "answer": "I don't have that in memory."}) + "\n"
                    + json.dumps({"id": "b", "answer": "Sure, here it is."}) + "\n", encoding="utf-8")
    assert abstain_rates(qs, ["a", "b"], [str(path)]) == {"answerable": 1.0, "adversarial": 0.0}


def test_kappa_perfect_and_chance():
    assert cohens_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert abs(cohens_kappa([True, True, False, False], [True, False, True, False])) < 1e-9


def test_agreement_counts_false_passes():
    s = agreement([False, False, True], [True, False, True])
    assert s["grader_false_pass"] == 1 and s["grader_false_fail"] == 0
    assert s["raw_agreement"] == round(2 / 3, 3)


def test_v2_eval_set_is_well_formed_and_sized():
    questions = load_jsonl(str(ROOT / "eval_questions_v2.jsonl"))
    ids = [q["id"] for q in questions]
    assert len(ids) >= 120 and len(ids) == len(set(ids))
    for q in questions:
        assert {"id", "domain", "tags", "question", "set"} <= q.keys(), q["id"]
        for pat in q.get("required", []) + q.get("banned", []):
            re.compile(pat)
    assert sum("adversarial" in q["tags"] for q in questions) >= 15


def test_v2_references_pass_their_own_specs():
    """Each reference answer should satisfy its own required/banned spec, or the spec is wrong."""
    from score_run import check
    exempt = {"faith-7", "coding-5", "id-5", "faith-24", "adv-1"}  # voice-only or no-reveal specs
    bad = []
    for q in load_jsonl(str(ROOT / "eval_questions_v2.jsonl")):
        if q["id"] in exempt or not q.get("reference"):
            continue
        ok, failed = check(q["reference"], dict(q, max_chars=None))
        if not ok:
            bad.append((q["id"], failed))
    assert bad == [], bad
