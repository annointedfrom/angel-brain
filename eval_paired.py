#!/usr/bin/env python3
"""eval_paired.py - compare retriever arms with the noise designed out.

Separate eval_system.py runs carry two noise sources that look like arm effects: query
embeddings drift between calls (about 5e-4 cosine), and Ollama answers byte-identical
prompts differently even at temperature 0. So here all arms run in one process on one
embedding per query, and arms that recall the same memories share one set of samples.
A question whose recall is the same in every arm is a tie by construction and is never
sampled, so every win or loss below comes from a real recall change.

    python eval_paired.py --arms vector-nolog vector-nolog-floor vector-nolog-gate \
        --questions eval_questions_v2.jsonl --db retrieval_data/knowledge_snapshot.db \
        --vectors retrieval_data/vectors --runs 5 --out runs/paired.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import random

from eval_system import DEFAULT_BV_DIR, compose_at, load_brain_voice
from report import bootstrap_ci
from retrieval_arms import Snapshot, build_arms
from score_run import check


def group_by_recall(recalls: dict[str, list[str]]) -> dict[tuple[str, ...], list[str]]:
    """Distinct recalled-memory lists -> the arms that recalled each one."""
    groups: dict[tuple[str, ...], list[str]] = {}
    for arm, memories in recalls.items():
        groups.setdefault(tuple(memories), []).append(arm)
    return groups


def paired_stats(a: dict[str, float], b: dict[str, float], total_questions: int,
                 iters: int = 2000, seed: int = 7) -> dict:
    """B - A in points over all questions. Questions absent from a and b are exact ties."""
    diffs = {q: b[q] - a[q] for q in a}
    values = list(diffs.values()) + [0.0] * (total_questions - len(diffs))
    low, high = bootstrap_ci(values, iters, random.Random(seed))
    return {"diff": 100 * sum(values) / total_questions, "low": 100 * low, "high": 100 * high,
            "wins": sorted(q for q, d in diffs.items() if d > 0),
            "losses": sorted(q for q, d in diffs.items() if d < 0)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--questions", default="eval_questions_v2.jsonl")
    ap.add_argument("--db", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--model", default="angel-brain")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--brain-voice-dir", default=DEFAULT_BV_DIR)
    args = ap.parse_args()

    bv = load_brain_voice(args.brain_voice_dir)
    snap = Snapshot(args.db)
    built = build_arms(snap, bv, args.vectors)
    with open(args.questions, encoding="utf-8") as f:
        questions = {q["id"]: q for q in map(json.loads, filter(str.strip, f))}

    # Any arm's embedding cache serves all of them: embed_query is shared in this process.
    plans = {}
    for qid, q in questions.items():
        recalls = {a: [snap.text(i) for i in built[a](q["question"], args.limit)] for a in args.arms}
        groups = group_by_recall(recalls)
        if len(groups) > 1:
            plans[qid] = groups
    print(f"{len(plans)}/{len(questions)} questions differ between arms; the rest are exact ties")

    # Interleaved so slow drift in the serving stack falls evenly on every recall set.
    passes = {qid: {a: 0 for a in args.arms} for qid in plans}
    with open(args.out, "w", encoding="utf-8") as out:
        for run in range(1, args.runs + 1):
            for qid, groups in plans.items():
                q = questions[qid]
                for memories, arms in groups.items():
                    answer = compose_at(bv, q["question"], list(memories), args.model, args.temperature)
                    ok = check(answer, q)[0]
                    for arm in arms:
                        passes[qid][arm] += ok
                    out.write(json.dumps({"id": qid, "run": run, "arms": arms, "recalled": len(memories),
                                          "passed": ok, "answer": answer}, ensure_ascii=False) + "\n")
                    out.flush()
            print(f"run {run}/{args.runs} done", flush=True)

    rates = {a: {qid: passes[qid][a] / args.runs for qid in plans} for a in args.arms}
    print("\nquestion  " + "  ".join(args.arms))
    for qid in plans:
        print(f"{qid:10}" + "  ".join(f"{passes[qid][a]}/{args.runs}" for a in args.arms))
    for a, b in itertools.combinations(args.arms, 2):
        s = paired_stats(rates[a], rates[b], len(questions))
        print(f"\n{b} - {a}: {s['diff']:+.1f} pts [{s['low']:+.1f}, {s['high']:+.1f}]"
              f"\n  wins {s['wins']}\n  losses {s['losses']}")


if __name__ == "__main__":
    main()
