#!/usr/bin/env python3
"""calibrate_floor.py - pick the vector arm's retrieval-confidence floor.

The floor drops recalled memories whose cosine falls below it, so a question with
no personal answer reaches the cortex with no memories instead of the nearest
off-topic note. It is calibrated on a private set kept apart from the eval
(Jaccard < 0.5 to every eval question), never on eval_questions_v2:

  answerable    personal questions whose fact is in the snapshot; `required` is a
                regex that marks the memories holding it
  unanswerable  general-knowledge questions with no personal answer

Rule, fixed before looking at scores: floor = the 2nd-lowest relevant-memory cosine
over answerable questions (about the 5th percentile at n=25), so real facts are
almost never dropped. The script also reports how many unanswerable top-1 scores
fall below that floor; if few do, one global floor can't separate the two sets.

    python calibrate_floor.py --db retrieval_data/knowledge_snapshot.db \
        --vectors retrieval_data/vectors --cal retrieval_data/floor_calibration.jsonl
"""
from __future__ import annotations

import argparse
import json
import re

from retrieval_arms import Snapshot, vector_arm


def best_relevant_sim(scored: list[tuple[int, float]], texts, pattern: str) -> float | None:
    """Cosine of the highest-ranked memory matching `pattern`, or None if none does."""
    for mem_id, sim in scored:
        if re.search(pattern, texts(mem_id), re.IGNORECASE):
            return sim
    return None


def pick_floor(relevant_sims: list[float]) -> float:
    ordered = sorted(relevant_sims)
    return ordered[1] if len(ordered) > 1 else ordered[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--cal", required=True)
    args = ap.parse_args()

    snap = Snapshot(args.db)
    arm = vector_arm(snap, args.vectors, log_penalty=float("inf"))
    with open(args.cal, encoding="utf-8") as f:
        cal = [json.loads(ln) for ln in f if ln.strip()]

    relevant, misses, general = [], [], []
    for q in cal:
        scored = arm.scored(q["question"], 100)
        if q["answerable"]:
            sim = best_relevant_sim(scored, snap.text, q["required"])
            (relevant.append(sim) if sim is not None else misses.append(q["id"]))
        else:
            general.append(scored[0][1])

    floor = pick_floor(relevant)
    below = sum(1 for s in general if s < floor)
    print(f"answerable relevant cosine (n={len(relevant)}): "
          f"{', '.join(f'{s:.3f}' for s in sorted(relevant))}")
    if misses:
        print(f"answerable with no matching memory in the top 100 (excluded): {', '.join(misses)}")
    print(f"unanswerable top-1 cosine (n={len(general)}): "
          f"{', '.join(f'{s:.3f}' for s in sorted(general))}")
    print(f"floor = {floor:.3f}; unanswerable top-1 below it: {below}/{len(general)}")


if __name__ == "__main__":
    main()
