#!/usr/bin/env python3
"""eval_system.py - the SYSTEM (S) fact eval. Runs on Windows (brain :8000 + Ollama :11434).

This is the real acceptance path for FACTS in the voice-LoRA + retrieval design:
    recall(brain) -> reason(cortex LLM) -> answer
Facts come from the brain's recall, NOT the model's weights. Point --model at any
cortex (e.g. nemotron-3-nano:4b now, or angel-brain after Phase 4) and grade with
score_run.py --tag fact.

Faithful to production: imports brain_voice.recall + its SYSTEM_PROMPT/NUM_CTX and
reuses the identical prompt. The only deliberate divergence is temperature=0 (greedy)
so the gate is reproducible -- brain_voice's live path uses 0.7. Does NOT speak or
write back to the brain (no TTS, no /observe pollution).

Emits answers_system_<model>.jsonl of {id, answer} for score_run.py / council_judge.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

DEFAULT_BV_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "NeuroLinked", "brain-voice")
)


def load_brain_voice(bv_dir: str):
    if not os.path.isfile(os.path.join(bv_dir, "brain_voice.py")):
        sys.exit(f"brain_voice.py not found in {bv_dir} (use --brain-voice-dir).")
    sys.path.insert(0, bv_dir)
    import brain_voice as bv  # noqa: E402
    return bv


def load_questions(path: str, tag: str) -> list[dict]:
    qs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                q = json.loads(line)
                if tag in q.get("tags", []):
                    qs.append({"id": q["id"], "question": q["question"]})
    return qs


def compose_greedy(bv, query: str, memories: list[str], model: str) -> str:
    """The real production path (brain_voice.compose via /api/chat), pinned to temp 0
    for a reproducible gate. Using /api/chat applies the cortex's template + stops, which
    is essential for an instruct cortex like angel-brain (raw /api/generate makes it ramble)."""
    return bv.compose(query, memories, model, temperature=0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="System fact eval: brain recall -> cortex.")
    ap.add_argument("--questions", default="eval_questions.jsonl")
    ap.add_argument("--tag", default="fact", help="Question tag to evaluate (default fact).")
    ap.add_argument("--model", default=os.environ.get("BRAIN_VOICE_MODEL", "nemotron-3-nano:4b"))
    ap.add_argument("--limit", type=int, default=5, help="Memories to recall per question.")
    ap.add_argument("--brain-voice-dir", default=DEFAULT_BV_DIR)
    args = ap.parse_args()

    bv = load_brain_voice(args.brain_voice_dir)
    questions = load_questions(args.questions, args.tag)
    print(f"System eval :: model={args.model}  questions={len(questions)} [tag={args.tag}]\n")

    answers = []
    for q in questions:
        mems = bv.recall(q["question"], args.limit)
        ans = compose_greedy(bv, q["question"], mems, args.model)
        answers.append({"id": q["id"], "answer": ans})
        print(f"Q[{q['id']}] ({len(mems)} mem): {q['question']}\nA: {ans}\n")

    label = re.sub(r"[^A-Za-z0-9._-]+", "-", args.model)
    out_path = f"answers_system_{label}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in answers:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"-> wrote {len(answers)} answers to {out_path}")
    print(f"   score with: python score_run.py --answers {out_path} --tag {args.tag}")


if __name__ == "__main__":
    main()
