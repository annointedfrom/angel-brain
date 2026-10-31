#!/usr/bin/env python3
"""assemble_dataset.py - merge gold/qa streams into a clean training set. Windows, stdlib.

Normalizes records to {instruction, input, output}, dedups, and -- critically --
drops any example whose question is too similar to a held-out eval question
(leakage guard: the eval only measures generalization if its phrasing is unseen).
Writes data/train.jsonl (+ optional internal val split). The real held-out gate
remains eval_questions.jsonl.

    python assemble_dataset.py gold_faith.jsonl gold_coding.jsonl gold_identity.jsonl \
        --eval eval_questions.jsonl --out-dir data
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def normalize(rec: dict) -> dict | None:
    instr = (rec.get("instruction") or rec.get("question") or "").strip()
    out = (rec.get("output") or rec.get("answer") or rec.get("text") or "").strip()
    if not instr or not out:
        return None
    norm = {"instruction": instr, "input": (rec.get("input") or "").strip(), "output": out}
    for key in ("memories", "system"):  # grounded examples (see grounding.py)
        if key in rec:
            norm[key] = rec[key]
    return norm


def norm_text(s: str) -> set[str]:
    """Bag of content words for fuzzy leakage comparison."""
    return {w for w in re.findall(r"\w+", s.lower()) if len(w) > 3}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge + clean + split training data.")
    ap.add_argument("inputs", nargs="+", help="gold_*.jsonl / qa jsonl files.")
    ap.add_argument("--eval", default="eval_questions.jsonl", help="Held-out questions to guard against.")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-frac", type=float, default=0.0, help="Internal val split (default 0; eval_questions is the real gate).")
    ap.add_argument("--leak-threshold", type=float, default=0.6, help="Jaccard >= this vs an eval Q => drop as leakage.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    eval_qs = [norm_text(q["question"]) for q in load_jsonl(args.eval)] if os.path.isfile(args.eval) else []

    seen: set[str] = set()
    rows: list[dict] = []
    dropped_dup = dropped_leak = 0
    for path in args.inputs:
        for rec in load_jsonl(path):
            n = normalize(rec)
            if not n:
                continue
            # A grounded question appears twice (memory present -> answer, absent -> abstain),
            # so the dedup key includes the target. A plain copy of a grounded example is a
            # different training signal (answer from weights), so grounding is part of the key.
            key = (n["instruction"] + "\x1f" + n["output"] + "\x1f" + str("memories" in n)).lower()
            if key in seen:
                dropped_dup += 1
                continue
            bag = norm_text(n["instruction"])
            if any(jaccard(bag, e) >= args.leak_threshold for e in eval_qs):
                dropped_leak += 1
                print(f"  [leak] dropped (~eval): {n['instruction'][:70]}")
                continue
            seen.add(key)
            rows.append(n)

    random.seed(args.seed)
    random.shuffle(rows)
    os.makedirs(args.out_dir, exist_ok=True)

    n_val = int(len(rows) * args.val_frac)
    val, train = rows[:n_val], rows[n_val:]
    train_path = os.path.join(args.out_dir, "train.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    msg = f"{len(train)} train"
    if n_val:
        val_path = os.path.join(args.out_dir, "val.jsonl")
        with open(val_path, "w", encoding="utf-8") as f:
            for r in val:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        msg += f", {len(val)} val"
    print(f"\nWrote {msg} -> {args.out_dir}/  "
          f"(dropped {dropped_dup} dup, {dropped_leak} leakage)")


if __name__ == "__main__":
    main()
