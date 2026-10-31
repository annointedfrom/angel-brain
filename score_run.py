#!/usr/bin/env python3
"""score_run.py - the reliable MECHANICAL gate for persona/adapter answers.

Pure stdlib, no model calls. Reads the held-out questions (with their embedded
required/banned/max_chars spec) and an answers file, and checks each answer for:
  - required : every regex MUST match (re.search, IGNORECASE)   -> fact rules present
  - banned   : no regex may match                                -> known failure modes absent
                 (e.g. "1 Cor" abbreviation, "Book of Mormon", NAB self-negation)
  - length   : len(answer) <= max_chars                          -> terseness (voice)
  - repeat   : no n-gram / line degeneration                     -> the v4 repetition-loop failure

Answers file: JSONL of {"id": <qid>, "answer": <text>}. Missing id = fail (missing).

Usage:
  python score_run.py --questions eval_questions.jsonl --answers answers_v2.jsonl [--tag fact] [--json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8-sig") as f:  # tolerate a BOM
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_degenerate(text: str) -> bool:
    """Detect repetition loops greedy decode exposes (the v4 failure).

    Two signals: a whole line repeated >=3x, or one 4-gram dominating the text.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        line_counts = Counter(lines)
        if max(line_counts.values()) >= 3 and len(lines) >= 4:
            return True
    words = re.findall(r"\w+", text.lower())
    if len(words) >= 12:
        grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
        if grams:
            top = Counter(grams).most_common(1)[0][1]
            if top >= 4 or top / len(grams) >= 0.12:
                return True
    return False


def check(answer: str, spec: dict) -> tuple[bool, list[str]]:
    failed: list[str] = []
    flags = re.IGNORECASE
    for pat in spec.get("required", []):
        if not re.search(pat, answer, flags):
            failed.append(f"missing-required:{pat}")
    for pat in spec.get("banned", []):
        if re.search(pat, answer, flags):
            failed.append(f"banned:{pat}")
    max_chars = spec.get("max_chars")
    if max_chars and len(answer) > max_chars:
        failed.append(f"too-long:{len(answer)}>{max_chars}")
    if is_degenerate(answer):
        failed.append("degenerate")
    return (not failed), failed


def main() -> None:
    ap = argparse.ArgumentParser(description="Mechanical scorer for adapter answers.")
    ap.add_argument("--questions", default="eval_questions.jsonl")
    ap.add_argument("--answers", required=True, help="JSONL of {id, answer}.")
    ap.add_argument("--tag", help="Only score questions carrying this tag (e.g. fact, voice).")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = ap.parse_args()

    questions = {q["id"]: q for q in load_jsonl(args.questions)}
    answers = {a["id"]: a.get("answer", "") for a in load_jsonl(args.answers)}

    qids = [qid for qid, q in questions.items()
            if not args.tag or args.tag in q.get("tags", [])]

    results = []
    for qid in qids:
        spec = questions[qid]
        if qid not in answers:
            results.append({"id": qid, "domain": spec["domain"], "pass": False,
                            "failed": ["missing-answer"]})
            continue
        ok, failed = check(answers[qid], spec)
        results.append({"id": qid, "domain": spec["domain"], "pass": ok, "failed": failed})

    passed = sum(r["pass"] for r in results)
    total = len(results)
    by_domain: dict[str, list[int]] = {}
    for r in results:
        d = by_domain.setdefault(r["domain"], [0, 0])
        d[1] += 1
        d[0] += int(r["pass"])

    summary = {
        "answers_file": args.answers,
        "tag": args.tag,
        "passed": passed,
        "total": total,
        "pass_rate": round(passed / total, 3) if total else 0.0,
        "by_domain": {d: f"{c[0]}/{c[1]}" for d, c in by_domain.items()},
        "results": results,
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    print(f"\n== score_run :: {args.answers}" + (f"  [tag={args.tag}]" if args.tag else ""))
    for r in results:
        mark = "PASS" if r["pass"] else "FAIL"
        detail = "" if r["pass"] else "  <- " + "; ".join(r["failed"])
        print(f"  [{mark}] {r['id']:<10}{detail}")
    print(f"\n  {passed}/{total} pass ({summary['pass_rate']:.0%})  by domain: " +
          ", ".join(f"{d} {v}" for d, v in summary["by_domain"].items()))
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
