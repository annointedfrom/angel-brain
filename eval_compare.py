#!/usr/bin/env python3
"""eval_compare.py - honest VOICE eval (the V track). Runs in WSL.

Fixes the blind-eval trap: greedy decode (no sampling noise), Qwen3 thinking
DISABLED (so the budget produces an answer, not a truncated <think>), and BASE
vs ADAPTER side by side so any delta is attributable to the adapter.

Reads held-out questions from eval_questions.jsonl and WRITES answers_<label>.jsonl
for each target so score_run.py (mechanical gate) and council_judge.py (jury) can
consume them. In the voice-LoRA + retrieval design, factual errors in the raw
adapter weights are expected here; this eval grades STYLE / terseness / non-degeneration.
(Facts are graded by eval_system.py through the retrieval cortex.)
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def load_questions(path: str, tag: str | None) -> list[dict]:
    qs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            if not tag or tag in q.get("tags", []):
                qs.append({"id": q["id"], "question": q["question"]})
    return qs


def label_for(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip("/").split("/")[-1])


def load(name: str):
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        name, max_seq_length=2048, load_in_4bit=True, dtype=None
    )
    FastLanguageModel.for_inference(model)
    return model, tok


def ask(model, tok, q: str, max_new: int) -> str:
    msgs = [{"role": "user", "content": q}]
    try:  # Qwen3 supports enable_thinking; other bases may not
        ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt"
        ).to("cuda")
    except TypeError:
        ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
    out = model.generate(input_ids=ids, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def run_target(label: str, name: str, questions: list[dict], max_new: int) -> None:
    import torch
    print(f"\n{'=' * 64}\n{label} :: {name}\n{'=' * 64}")
    model, tok = load(name)
    answers = []
    for q in questions:
        a = ask(model, tok, q["question"], max_new)
        answers.append({"id": q["id"], "answer": a})
        print(f"\nQ[{q['id']}]: {q['question']}\nA: {a}")
    out_path = f"answers_{label}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in answers:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n-> wrote {len(answers)} answers to {out_path}")
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval_questions.jsonl")
    ap.add_argument("--tag", help="Only ask questions carrying this tag.")
    ap.add_argument("--adapter", default="persona_adapter")
    ap.add_argument("--base", default="unsloth/Qwen3-4B")
    ap.add_argument("--which", choices=("adapter", "base", "both"), default="both")
    ap.add_argument("--max-new", type=int, default=300)
    args = ap.parse_args()

    questions = load_questions(args.questions, args.tag)
    print(f"Loaded {len(questions)} questions" + (f" [tag={args.tag}]" if args.tag else ""))

    if args.which in ("base", "both"):
        run_target(f"base-{label_for(args.base)}", args.base, questions, args.max_new)
    if args.which in ("adapter", "both"):
        run_target(label_for(args.adapter), args.adapter, questions, args.max_new)


if __name__ == "__main__":
    main()
