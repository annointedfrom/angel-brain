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
import time

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
                if tag == "all" or tag in q.get("tags", []):
                    qs.append({"id": q["id"], "question": q["question"]})
    return qs


def compose_at(bv, query: str, memories: list[str], model: str, temperature: float,
               system: str | None = None) -> str:
    """The real production path (brain_voice.compose via /api/chat). Temp 0 = reproducible
    gate; temp > 0 with --runs N measures run-to-run variance. Using /api/chat applies the
    cortex's template + stops, which is essential for an instruct cortex like angel-brain
    (raw /api/generate makes it ramble)."""
    return bv.compose(query, memories, model, temperature=temperature, system=system)


def main() -> None:
    ap = argparse.ArgumentParser(description="System fact eval: brain recall -> cortex.")
    ap.add_argument("--questions", default="eval_questions.jsonl")
    ap.add_argument("--tag", default="fact", help="Question tag to evaluate (default fact; 'all' = every question).")
    ap.add_argument("--model", default=os.environ.get("BRAIN_VOICE_MODEL", "nemotron-3-nano:4b"))
    ap.add_argument("--limit", type=int, default=5, help="Memories to recall per question.")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = reproducible gate; >0 with --runs measures variance.")
    ap.add_argument("--runs", type=int, default=1, help="Repeat the whole set N times (one answers file per run).")
    ap.add_argument("--system-prompt", help="File holding a system prompt to A/B against brain_voice's default.")
    ap.add_argument("--label", help="Extra label for output files (e.g. the prompt variant).")
    ap.add_argument("--brain-voice-dir", default=DEFAULT_BV_DIR)
    args = ap.parse_args()

    system = None
    if args.system_prompt:
        with open(args.system_prompt, encoding="utf-8") as f:
            system = f.read().strip()

    bv = load_brain_voice(args.brain_voice_dir)
    questions = load_questions(args.questions, args.tag)
    print(f"System eval :: model={args.model}  questions={len(questions)} [tag={args.tag}]"
          f"  temp={args.temperature}  runs={args.runs}\n")

    # Recall is deterministic, so fetch it once and reuse it across runs.
    recalled = {q["id"]: bv.recall(q["question"], args.limit) for q in questions}

    label = re.sub(r"[^A-Za-z0-9._-]+", "-", args.model + (f"_{args.label}" if args.label else ""))
    suffix = "" if args.runs == 1 and args.temperature == 0 else f"_t{args.temperature:g}"
    out_paths = []
    for run in range(1, args.runs + 1):
        run_suffix = suffix + (f"_r{run}" if args.runs > 1 else "")
        out_path = f"answers_system_{label}{run_suffix}.jsonl"
        errors = 0
        # Written row by row so a crash late in a long run keeps every finished answer.
        with open(out_path, "w", encoding="utf-8") as f:
            for q in questions:
                mems = recalled[q["id"]]
                start = time.perf_counter()
                row = {"id": q["id"], "recalled": len(mems), "memories": mems}
                try:
                    row["answer"] = compose_at(bv, q["question"], mems, args.model, args.temperature, system)
                except SystemExit as e:  # brain_voice.compose exits on Ollama errors
                    errors += 1
                    row["answer"], row["error"] = "", str(e)
                row["latency_s"] = round(time.perf_counter() - start, 2)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[run {run}] Q[{q['id']}] ({len(mems)} mem, {row['latency_s']}s): "
                      f"{q['question']}\nA: {row['answer'] or 'ERROR ' + row.get('error', '')}\n")
        out_paths.append(out_path)
        print(f"-> wrote {len(questions)} answers to {out_path} ({errors} errors, scored as fails)")

    print(f"   score with: python report.py --questions {args.questions} --answers {' '.join(out_paths)}")


if __name__ == "__main__":
    main()
