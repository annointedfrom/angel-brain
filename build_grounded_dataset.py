#!/usr/bin/env python3
"""build_grounded_dataset.py - turn voice demos into grounded examples that teach abstention.

The v2 eval found 52% of hard failures were confident fabrication after a retrieval miss
(ERROR-ANALYSIS.md). Each personal-fact demo becomes two examples in the serve-time
format (grounding.py):
  - positive: the true answer appears among the recalled memories -> answer from it
  - abstain : only unrelated memories are recalled            -> say it isn't in memory
Hand-written examples (gold_abstain.jsonl: unknowables, approval gates, commit types,
general knowledge) get unrelated memories, so the model learns those answers don't
depend on recall.

Distractor memories are other demos' answers, chosen to share few words with the target.

Usage:
  python build_grounded_dataset.py --system prompts/voice_v2_abstain.txt -o gold_grounded.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Facts that must come from memory: these get both a positive and an abstain example.
PERSONAL = ["gold_identity.jsonl", "gold_vault.jsonl", "gold_workflow.jsonl"]
# Rules the voice should also hold from weights: positive examples only.
RULES = ["gold_coding.jsonl", "gold_faith.jsonl", "gold_persona.jsonl", "gold_more.jsonl"]
HAND = ["gold_abstain.jsonl", "gold_general.jsonl"]
ABSTAIN_REPLIES = [
    "I don't have that in memory, so I won't guess.",
    "That isn't in my memory. You may want to check the vault.",
    "My notes don't cover that, and I'd rather not make it up.",
]


def load(name: str) -> list[dict]:
    path = HERE / name
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if len(w) > 3}


def distractors(target: str, pool: list[str], k: int, rng: random.Random) -> list[str]:
    """k memories that share little vocabulary with the target answer."""
    tw = words(target)
    far = [p for p in pool if p != target and len(words(p) & tw) <= max(1, len(tw) // 6)]
    return rng.sample(far, min(k, len(far)))


def build(system: str, seed: int, k: int, abstain_frac: float = 0.33) -> list[dict]:
    rng = random.Random(seed)
    personal = [r for name in PERSONAL for r in load(name)]
    rules = [r for name in RULES for r in load(name)]
    hand = [r for name in HAND for r in load(name)]
    pool = [r["output"] for r in personal + rules]

    def example(question: str, memories: list[str], output: str) -> dict:
        ex = {"instruction": question, "memories": memories, "output": output}
        if system:
            ex["system"] = system
        return ex

    def note(core: str) -> str:
        """Bury a fact inside a note-like memory, as real recall returns whole notes.

        v3 lesson: with the answer as an exact standalone memory, the model learned
        "copy an exact match, otherwise refuse" and then refused everything at serve time.
        """
        parts = [core] + rng.sample(pool, 2)
        rng.shuffle(parts)
        return " ".join(parts)

    out = []
    for r in personal + rules:
        mems = [note(m) for m in distractors(r["output"], pool, k - 1, rng)] + [note(r["output"])]
        rng.shuffle(mems)
        out.append(example(r["instruction"], mems, r["output"]))
    # Abstain on a minority only; v3 abstained on every personal question and over-refused.
    abstain_n = round(len(personal) * abstain_frac)
    for r in rng.sample(personal, abstain_n):
        mems = [note(m) for m in distractors(r["output"], pool, k, rng)]
        out.append(example(r["instruction"], mems, rng.choice(ABSTAIN_REPLIES)))
    for r in hand:  # unknowables, approval gates, commit types, general knowledge
        mems = [note(m) for m in distractors(r["output"], pool, k, rng)]
        out.append(example(r["instruction"], mems, r["output"]))
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--system", help="System prompt file to bake into every example (match serving).")
    ap.add_argument("-o", "--out", default="gold_grounded.jsonl")
    ap.add_argument("--memories", type=int, default=4, help="Memories per example (production recalls 5).")
    ap.add_argument("--abstain-frac", type=float, default=0.33, help="Share of personal questions given an abstain example.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    system = Path(args.system).read_text(encoding="utf-8").strip() if args.system else ""
    rows = build(system, args.seed, args.memories, args.abstain_frac)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    abstain = sum(r["output"] in ABSTAIN_REPLIES for r in rows)
    print(f"wrote {len(rows)} grounded examples ({abstain} abstain) to {args.out}")


if __name__ == "__main__":
    main()
