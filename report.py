#!/usr/bin/env python3
"""report.py - pass rates with bootstrap confidence intervals over repeated eval runs.

Single-run counts on a small set hide noise (a one-question swing on 18 items is 5.6
points). This scores one or more answers files per system with score_run.check, then
reports, overall and per domain:
  - mean pass rate across runs, plus the min-max run spread
  - a 95% bootstrap CI that resamples QUESTIONS (each question's pass rate is averaged
    over runs first, so run variance and question sampling are both reflected)
  - with --compare: the paired difference B - A and its bootstrap CI over the same questions

Pure stdlib.

Usage:
  python report.py --questions eval_questions_v2.jsonl --answers a_r1.jsonl a_r2.jsonl a_r3.jsonl
  python report.py --questions eval_questions_v2.jsonl --answers a_r*.jsonl --compare b_r*.jsonl --json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from statistics import mean

from score_run import check, load_jsonl

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


ABSTAIN = re.compile(r"don't have|do not have|not in (my )?memory|isn't in my memory|can't know|cannot know"
                     r"|don't know|won't (guess|make it up)|notes don't cover", re.IGNORECASE)


def abstain_rates(questions: dict[str, dict], qids: list[str], answer_files: list[str]) -> dict:
    """How often the model says "not in memory" on answerable vs adversarial questions.

    Abstaining on a non-adversarial question is over-refusal (the v3 failure); on the
    adversarial set it is often the right call.
    """
    counts = {"answerable": [0, 0], "adversarial": [0, 0]}
    for path in answer_files:
        answers = {a["id"]: a.get("answer", "") for a in load_jsonl(path)}
        for qid in qids:
            bucket = "adversarial" if "adversarial" in questions[qid].get("tags", []) else "answerable"
            counts[bucket][1] += 1
            counts[bucket][0] += bool(ABSTAIN.search(answers.get(qid, "")))
    return {k: round(a / n, 3) if n else 0.0 for k, (a, n) in counts.items()}


def per_question_scores(questions: dict[str, dict], answer_files: list[str]) -> dict[str, list[int]]:
    """qid -> [pass(0/1) for each run]. A question missing from a run counts as a fail."""
    scores: dict[str, list[int]] = {qid: [] for qid in questions}
    for path in answer_files:
        answers = {a["id"]: a.get("answer", "") for a in load_jsonl(path)}
        for qid, spec in questions.items():
            ok = qid in answers and check(answers[qid], spec)[0]
            scores[qid].append(int(ok))
    return scores


def bootstrap_ci(values: list[float], iters: int, rng: random.Random) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    n = len(values)
    means = sorted(mean(rng.choices(values, k=n)) for _ in range(iters))
    return (means[int(0.025 * iters)], means[int(0.975 * iters) - 1])


def summarize(qids: list[str], scores: dict[str, list[int]], iters: int, seed: int) -> dict:
    rng = random.Random(seed)
    per_q = [mean(scores[q]) for q in qids]
    runs = len(next(iter(scores.values()), []))
    run_rates = [mean(scores[q][r] for q in qids) for r in range(runs)] if qids else []
    lo, hi = bootstrap_ci(per_q, iters, rng)
    return {
        "n_questions": len(qids),
        "runs": runs,
        "pass_rate": round(mean(per_q), 3) if per_q else 0.0,
        "ci95": [round(lo, 3), round(hi, 3)],
        "run_spread": [round(min(run_rates), 3), round(max(run_rates), 3)] if run_rates else [],
    }


def paired_diff(qids: list[str], a: dict, b: dict, iters: int, seed: int) -> dict:
    rng = random.Random(seed)
    diffs = [mean(b[q]) - mean(a[q]) for q in qids]
    lo, hi = bootstrap_ci(diffs, iters, rng)
    return {"diff": round(mean(diffs), 3) if diffs else 0.0, "ci95": [round(lo, 3), round(hi, 3)],
            "significant": lo > 0 or hi < 0}


def build_report(questions: dict[str, dict], answers: list[str], compare: list[str] | None,
                 tag: str | None, iters: int, seed: int, domain: str | None = None) -> dict:
    qids = [q for q, spec in questions.items()
            if (not tag or tag in spec.get("tags", [])) and (not domain or spec["domain"] == domain)]
    if not qids:
        raise ValueError(f"no questions match tag={tag!r} domain={domain!r}")
    a = per_question_scores(questions, answers)
    domains = sorted({questions[q]["domain"] for q in qids})
    report = {"tag": tag, "answers": answers,
              "overall": summarize(qids, a, iters, seed),
              "by_domain": {d: summarize([q for q in qids if questions[q]["domain"] == d], a, iters, seed)
                            for d in domains}}
    if compare:
        b = per_question_scores(questions, compare)
        report["compare"] = {"answers": compare, "overall": summarize(qids, b, iters, seed),
                             "diff_b_minus_a": paired_diff(qids, a, b, iters, seed)}
    report["abstain"] = abstain_rates(questions, qids, answers)
    if compare:
        report["compare"]["abstain"] = abstain_rates(questions, qids, compare)
    report["always_fail"] = sorted(q for q in qids if not any(a[q]))
    return report


def fmt(s: dict) -> str:
    spread = f"  runs {s['run_spread'][0]:.0%}-{s['run_spread'][1]:.0%}" if s["runs"] > 1 else ""
    return (f"{s['pass_rate']:.0%}  [95% CI {s['ci95'][0]:.0%}-{s['ci95'][1]:.0%}]"
            f"  n={s['n_questions']}{spread}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pass rates with bootstrap CIs over repeated runs.")
    ap.add_argument("--questions", default="eval_questions_v2.jsonl")
    ap.add_argument("--answers", nargs="+", required=True, help="Answers files for system A (one per run).")
    ap.add_argument("--compare", nargs="+", help="Answers files for system B (one per run).")
    ap.add_argument("--tag", help="Only questions with this tag (fact, voice, adversarial).")
    ap.add_argument("--domain", help="Only questions in this domain (faith, coding, identity, ...).")
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    questions = {q["id"]: q for q in load_jsonl(args.questions)}
    try:
        report = build_report(questions, args.answers, args.compare, args.tag, args.iters, args.seed, args.domain)
    except ValueError as e:
        sys.exit(str(e))

    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(f"\n== report :: A = {len(args.answers)} run(s)" + (f"  [tag={args.tag}]" if args.tag else ""))
    print(f"  overall   {fmt(report['overall'])}")
    for d, s in report["by_domain"].items():
        print(f"  {d:<11} {fmt(s)}")
    ab = report["abstain"]
    print(f"  abstains  answerable {ab['answerable']:.0%} (over-refusal)  adversarial {ab['adversarial']:.0%}")
    if "compare" in report:
        c = report["compare"]
        d = c["diff_b_minus_a"]
        print(f"\n  B overall {fmt(c['overall'])}")
        print(f"  B abstains answerable {c['abstain']['answerable']:.0%}  adversarial {c['abstain']['adversarial']:.0%}")
        print(f"  B - A     {d['diff']:+.1%}  [95% CI {d['ci95'][0]:+.1%} to {d['ci95'][1]:+.1%}]"
              f"  {'significant' if d['significant'] else 'not significant'}")
    if report["always_fail"]:
        print(f"\n  failed every run ({len(report['always_fail'])}): {', '.join(report['always_fail'])}")


if __name__ == "__main__":
    main()
