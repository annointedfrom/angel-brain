#!/usr/bin/env python3
"""score_retrieval.py - hit@5, MRR@10 and recall@5 per arm against silver labels.

Queries whose pooled candidates contained no relevant memory are excluded from the
metrics and listed separately. Check them against the snapshot before calling them
ingest gaps: if the fact is stored but no arm ranked it top-10, it is a miss shared by
every arm, and excluding it flatters all of them equally.

The labels are silver (local LLM judge + answer-term gate). A lenient judge favors
on-topic memories, which is what the vector arm returns, so a robustness table rescores
only the questions whose answer spec has discriminating terms.

    python score_retrieval.py --db retrieval_data/knowledge_snapshot.db \
        --vectors retrieval_data/vectors --gold retrieval_data/retrieval_gold.jsonl
"""
from __future__ import annotations

import argparse
import json

from retrieval_arms import Snapshot, build_arms
from retrieval_metrics import (bootstrap_ci, hit_at_k, paired_bootstrap_diff, recall_at_k,
                               reciprocal_rank)

# The 14 "retrieval miss, then confident fabrication" questions from ERROR-ANALYSIS.md.
RETRIEVAL_MISS_IDS = {"id-1", "id-3", "id-8", "id-10", "id-11", "id-12", "id-13", "id-14",
                      "id-21", "wf-3", "wf-13", "wf-15", "wf-16", "wf-21"}
# Answer specs that are empty or match generic on-topic text ("Catholic", "NAB", a bare
# negation), so the answer-term gate can't catch a lenient judge on these questions.
WEAK_SPEC_IDS = {"faith-1", "faith-3", "faith-4", "faith-7", "faith-24", "coding-5", "id-5",
                 "id-14", "wf-4", "wf-5", "wf-9", "wf-12"}
METRICS = {"hit@5": lambda r, g: hit_at_k(r, g, 5),
           "MRR@10": lambda r, g: reciprocal_rank(r, g, 10),
           "recall@5": lambda r, g: recall_at_k(r, g, 5)}


def score_arms(arms: dict, gold: list[dict], questions: dict[str, str]) -> dict[str, dict[str, list[float]]]:
    """Per arm, per metric: one value per gold query, aligned across arms."""
    out = {name: {m: [] for m in METRICS} for name in arms}
    for g in gold:
        rel = set(g["relevant"])
        for name, arm in arms.items():
            ranked = arm(questions[g["id"]], 10)
            for m, fn in METRICS.items():
                out[name][m].append(fn(ranked, rel))
    return out


def fmt(mean: float, lo: float, hi: float, signed: bool = False) -> str:
    s = "+" if signed else ""
    return f"{mean:{s}.2f} [{lo:{s}.2f}, {hi:{s}.2f}]"


def table(title: str, scores: dict, baseline: str = "production") -> None:
    n = len(next(iter(scores.values()))["hit@5"])
    print(f"\n### {title} (n={n})\n")
    print("| Arm | hit@5 | MRR@10 | recall@5 | Δ hit@5 vs production |")
    print("|---|---|---|---|---|")
    for name, s in scores.items():
        cells = [fmt(*bootstrap_ci(s[m])) for m in METRICS]
        if name == baseline:
            delta = "—"
        else:
            d, lo, hi = paired_bootstrap_diff(scores[baseline]["hit@5"], s["hit@5"])
            delta = fmt(d, lo, hi, signed=True) + (" *" if lo > 0 or hi < 0 else "")
        print(f"| {name} | " + " | ".join(cells) + f" | {delta} |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--questions", default="eval_questions_v2.jsonl")
    ap.add_argument("--brain-voice-dir")
    args = ap.parse_args()

    from eval_system import DEFAULT_BV_DIR, load_brain_voice
    bv = load_brain_voice(args.brain_voice_dir or DEFAULT_BV_DIR)
    snap = Snapshot(args.db)
    arms = build_arms(snap, bv, args.vectors)

    with open(args.questions, encoding="utf-8") as f:
        questions = {q["id"]: q["question"] for q in map(json.loads, f)}
    with open(args.gold, encoding="utf-8") as f:
        all_gold = [json.loads(ln) for ln in f]
    gold = [g for g in all_gold if g["relevant"]]
    gaps = sorted(g["id"] for g in all_gold if not g["relevant"])

    print(f"{len(all_gold)} labeled queries; {len(gaps)} have no relevant memory in any arm's "
          f"top 10 (excluded; check the snapshot before calling them ingest gaps): "
          f"{', '.join(gaps) or 'none'}")
    print("`*` = paired 95% CI excludes 0.")
    table("All answerable personal questions", score_arms(arms, gold, questions))
    strict = [g for g in gold if g["id"] not in WEAK_SPEC_IDS]
    table("Robustness: discriminating answer specs only", score_arms(arms, strict, questions))
    for domain in sorted({g["domain"] for g in gold}):
        sub = [g for g in gold if g["domain"] == domain]
        table(f"Domain: {domain}", score_arms(arms, sub, questions))
    miss = [g for g in gold if g["id"] in RETRIEVAL_MISS_IDS]
    if miss:
        table("ERROR-ANALYSIS category-1 retrieval misses", score_arms(arms, miss, questions))
    print(f"\nCategory-1 misses with no relevant memory in any arm's top 10: "
          f"{', '.join(i for i in gaps if i in RETRIEVAL_MISS_IDS) or 'none'}")


if __name__ == "__main__":
    main()
