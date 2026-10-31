#!/usr/bin/env python3
"""calibrate_judge.py - check automated graders against human labels before trusting them.

An LLM jury (or a regex gate) is only evidence once it agrees with a human on a sample.
Two steps:

  1. sample  - draw N answers into a labeling sheet. Each row shows the question,
               reference, answer and the mechanical verdict; you fill "human_pass" with
               true/false (and an optional "note" naming the failure category).
  2. score   - compute agreement between your labels and (a) score_run's mechanical gate,
               (b) the council jury's per-item scores (council_judge.py --per-item), as raw
               agreement, Cohen's kappa and a confusion matrix.

Pure stdlib.

Usage:
  python calibrate_judge.py sample --questions eval_questions_v2.jsonl --answers a_r1.jsonl --n 50 --out labels.jsonl
  python calibrate_judge.py score --questions eval_questions_v2.jsonl --labels labels.jsonl [--jury jury_items.jsonl --jury-threshold 4]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

from score_run import check, load_jsonl

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def cohens_kappa(a: list[bool], b: list[bool]) -> float:
    """Agreement corrected for chance. 1 = perfect, 0 = chance level, < 0 = worse than chance."""
    n = len(a)
    if n == 0:
        return 0.0
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def confusion(human: list[bool], grader: list[bool]) -> dict[str, int]:
    pairs = list(zip(human, grader))
    return {"both_pass": pairs.count((True, True)), "both_fail": pairs.count((False, False)),
            "grader_false_pass": pairs.count((False, True)), "grader_false_fail": pairs.count((True, False))}


def agreement(human: list[bool], grader: list[bool]) -> dict:
    n = len(human)
    return {"n": n,
            "raw_agreement": round(sum(h == g for h, g in zip(human, grader)) / n, 3) if n else 0.0,
            "kappa": round(cohens_kappa(human, grader), 3),
            **confusion(human, grader)}


def cmd_sample(args) -> None:
    questions = {q["id"]: q for q in load_jsonl(args.questions)}
    rows = []
    for path in args.answers:
        candidate = re.sub(r"^answers_|\.jsonl$", "", os.path.basename(path))
        for a in load_jsonl(path):
            spec = questions.get(a["id"])
            if spec is None:
                continue
            ok, failed = check(a.get("answer", ""), spec)
            rows.append({"id": a["id"], "candidate": candidate, "question": spec["question"],
                         "reference": spec.get("reference", ""), "answer": a.get("answer", ""),
                         "mechanical_pass": ok, "mechanical_failed": failed,
                         "human_pass": None, "note": ""})
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.n]
    if os.path.exists(args.out) and not args.force:
        sys.exit(f"{args.out} exists (it may hold your labels); pass --force to overwrite")
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}; set human_pass to true/false on each")


def cmd_score(args) -> None:
    labels = [r for r in load_jsonl(args.labels) if isinstance(r.get("human_pass"), bool)]
    if not labels:
        sys.exit("no labeled rows (human_pass must be true/false)")
    human = [r["human_pass"] for r in labels]
    report = {"labeled": len(labels), "mechanical": agreement(human, [r["mechanical_pass"] for r in labels])}

    if args.jury:
        jury = {(j["id"], j["candidate"]): j for j in load_jsonl(args.jury)}
        matched = [(r["human_pass"], jury[(r["id"], r["candidate"])]) for r in labels
                   if (r["id"], r["candidate"]) in jury]
        if matched:
            report["jury"] = agreement([h for h, _ in matched],
                                       [j["fact"] >= args.jury_threshold for _, j in matched])
            report["jury"]["threshold"] = f"fact >= {args.jury_threshold}"

    notes: dict[str, int] = {}
    for r in labels:
        if r.get("note"):
            notes[r["note"]] = notes.get(r["note"], 0) + 1
    report["failure_categories"] = dict(sorted(notes.items(), key=lambda kv: -kv[1]))

    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(f"\n== calibration :: {len(labels)} human-labeled answers")
    for name in ("mechanical", "jury"):
        if name in report:
            s = report[name]
            print(f"  {name:<10} agreement {s['raw_agreement']:.0%}  kappa {s['kappa']:+.2f}  "
                  f"false-pass {s['grader_false_pass']}  false-fail {s['grader_false_fail']}  (n={s['n']})")
    if report["failure_categories"]:
        print("  failure categories: " + ", ".join(f"{k} {v}" for k, v in report["failure_categories"].items()))
    print("  rule of thumb: trust a grader for gating only at kappa >= 0.6")


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate graders against human labels.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="Draw a labeling sheet.")
    s.add_argument("--questions", default="eval_questions_v2.jsonl")
    s.add_argument("--answers", nargs="+", required=True)
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="labels.jsonl")
    s.add_argument("--force", action="store_true")
    c = sub.add_parser("score", help="Agreement of graders with your labels.")
    c.add_argument("--questions", default="eval_questions_v2.jsonl")
    c.add_argument("--labels", default="labels.jsonl")
    c.add_argument("--jury", help="Per-item jury scores from council_judge.py --per-item.")
    c.add_argument("--jury-threshold", type=float, default=4.0)
    c.add_argument("--json", action="store_true")
    args = ap.parse_args()
    cmd_sample(args) if args.cmd == "sample" else cmd_score(args)


if __name__ == "__main__":
    main()
