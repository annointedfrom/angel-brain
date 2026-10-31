#!/usr/bin/env python3
"""council_judge.py - SECONDARY jury: the local LLM Council scores candidate answers.

Per the plan, this is a *corroborating* signal, not the decider (the Council's own
docs call 3-8B rankings noisy). The reliable gate is score_run.py (mechanical) +
Claude-as-judge. This reuses the Council's provider facade so the same roster
(backend/config.COUNCIL_MODELS) -- and optionally claude-cli/codex-cli -- judges.

Each judge scores a candidate answer 1-5 on voice / fact / terseness / coherence vs
the reference, strict-JSON with a regex fallback. Aggregates to a leaderboard across
one or more candidate answer files.

Run with the COUNCIL's venv (it has httpx/dotenv), e.g. (PowerShell):
  & "<vault>\Resources\vault-tools\llm-council\.venv\Scripts\python.exe" `
      council_judge.py --questions eval_questions.jsonl `
      --answers answers_persona_adapter_v2.jsonl answers_base-Qwen3-4B.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

DEFAULT_COUNCIL_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "llm-council")
)
DIMS = ("voice", "fact", "terseness", "coherence")
RUBRIC = (
    "You are a strict evaluator of an AI assistant's answer. Score the CANDIDATE answer "
    "against the REFERENCE on four axes, each an integer 1-5:\n"
    "  voice: terse, on-persona (concise, production-minded, Catholic-aware), no rambling.\n"
    "  fact: factually consistent with the reference (correct rules, citations, names).\n"
    "  terseness: appropriately short; penalize padding and markdown bloat.\n"
    "  coherence: well-formed, no repetition loops or contradictions.\n"
    'Reply with ONLY a JSON object: {"voice":N,"fact":N,"terseness":N,"coherence":N}'
)


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:  # tolerate a BOM from PowerShell-written files
        return [json.loads(ln) for ln in f if ln.strip()]


def import_council(council_dir: str):
    if not os.path.isdir(os.path.join(council_dir, "backend")):
        sys.exit(f"council backend not found in {council_dir} (use --council-dir).")
    sys.path.insert(0, council_dir)
    from backend.providers.facade import query_models_parallel  # noqa: E402
    from backend.config import COUNCIL_MODELS  # noqa: E402
    return query_models_parallel, COUNCIL_MODELS


def parse_scores(text: str) -> dict | None:
    if not text:
        return None
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if all(d in obj for d in DIMS):
                return {d: float(obj[d]) for d in DIMS}
        except (ValueError, TypeError):
            pass
    out = {}  # regex fallback: "voice": 4 / voice = 4
    for d in DIMS:
        mm = re.search(rf'"?{d}"?\s*[:=]\s*([1-5])', text, re.IGNORECASE)
        if mm:
            out[d] = float(mm.group(1))
    return out if len(out) == len(DIMS) else None


async def judge_answer(query_parallel, judges, question, reference, candidate) -> dict:
    msgs = [
        {"role": "system", "content": RUBRIC},
        {"role": "user", "content": (
            f"QUESTION:\n{question}\n\nREFERENCE:\n{reference}\n\n"
            f"CANDIDATE:\n{candidate}\n\nJSON only:")},
    ]
    responses = await query_parallel(judges, msgs)
    scores = {}
    for judge, resp in responses.items():
        parsed = parse_scores(resp.get("content") if resp else None)
        if parsed:
            scores[judge] = parsed
    return scores


async def run(query_parallel, judges, questions, answers_files, tag) -> list[dict]:
    qmap = {q["id"]: q for q in questions if not tag or tag in q.get("tags", [])}
    leaderboard = []
    for af in answers_files:
        label = re.sub(r"^answers_|\.jsonl$", "", os.path.basename(af))
        answers = {a["id"]: a.get("answer", "") for a in load_jsonl(af)}
        totals = {d: [] for d in DIMS}
        judged = 0
        for qid, q in qmap.items():
            if qid not in answers:
                continue
            scores = await judge_answer(query_parallel, judges, q["question"],
                                        q.get("reference", ""), answers[qid])
            if not scores:
                continue
            judged += 1
            for d in DIMS:  # mean across judges for this question
                vals = [s[d] for s in scores.values()]
                totals[d].append(sum(vals) / len(vals))
        means = {d: round(sum(v) / len(v), 2) if v else 0.0 for d, v in totals.items()}
        overall = round(sum(means.values()) / len(DIMS), 2)
        leaderboard.append({"candidate": label, "judged": judged,
                            "overall": overall, **means})
    leaderboard.sort(key=lambda r: r["overall"], reverse=True)
    return leaderboard


def main() -> None:
    ap = argparse.ArgumentParser(description="Council jury over candidate answer files.")
    ap.add_argument("--questions", default="eval_questions.jsonl")
    ap.add_argument("--answers", nargs="+", required=True, help="One or more answers_*.jsonl.")
    ap.add_argument("--tag", help="Only judge questions with this tag.")
    ap.add_argument("--judges", nargs="*", help="Override judge model ids (default COUNCIL_MODELS).")
    ap.add_argument("--council-dir", default=DEFAULT_COUNCIL_DIR)
    args = ap.parse_args()

    query_parallel, council_models = import_council(args.council_dir)
    judges = args.judges or council_models
    questions = load_jsonl(args.questions)
    print(f"Jury judges: {judges}\nCandidates: {len(args.answers)}  "
          f"questions tag={args.tag or 'all'}\n")

    board = asyncio.run(run(query_parallel, judges, questions, args.answers, args.tag))

    print(f"{'candidate':<32}{'overall':>8}{'voice':>7}{'fact':>6}{'terse':>7}{'coher':>7}{'n':>4}")
    print("-" * 71)
    for r in board:
        print(f"{r['candidate']:<32}{r['overall']:>8}{r['voice']:>7}{r['fact']:>6}"
              f"{r['terseness']:>7}{r['coherence']:>7}{r['judged']:>4}")
    print("\n(secondary signal — defer to score_run.py + Claude-as-judge)")


if __name__ == "__main__":
    main()
