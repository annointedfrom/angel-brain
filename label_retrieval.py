#!/usr/bin/env python3
"""label_retrieval.py - build silver relevance labels for the retrieval A/B.

For each answerable personal question (faith, coding, identity, workflow), pool the
top-N memories from every arm, then ask a LOCAL judge model whether each memory
contains the fact needed for the reference answer. Pooling across arms keeps the
labels from favoring whichever retriever generated them; a regex-match label would
favor the keyword arms.

Judgments are cached in <out>.cache.jsonl by (question id, memory id), so adding an
arm or re-running only judges new pairs. Memories are private: the judge must be a
local model, and the gold file stays out of the public release.

    python label_retrieval.py --db retrieval_data/knowledge_snapshot.db \
        --vectors retrieval_data/vectors --out retrieval_data/retrieval_gold.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request

from retrieval_arms import Snapshot, build_arms

DOMAINS = ("faith", "coding", "identity", "workflow")
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MEMORY_CHARS = 1500

JUDGE_PROMPT = """You are labeling search results for a personal-memory assistant.

Question: {question}
Correct answer: {reference}

Retrieved memory:
<<<
{memory}
>>>

Does this memory state the specific fact a reader would need to give the correct answer?
Rules:
- relevant only if the memory itself contains that fact (or a clear paraphrase of it).
- not relevant if it is merely on the same topic, mentions a keyword, or contains only part
  of a multi-part answer that leaves out the key fact.
Reply with JSON only: {{"relevant": true}} or {{"relevant": false}}"""


def parse_verdict(raw: str) -> bool | None:
    """Read the judge's JSON verdict; None if it can't be parsed."""
    m = re.search(r'"relevant"\s*:\s*(true|false)', raw, re.IGNORECASE)
    return m.group(1).lower() == "true" if m else None


def holds_answer_terms(memory: str, required: list[str]) -> bool:
    """True if the memory contains every `required` pattern of the question's answer spec.

    The 8B judge says yes to on-topic memories that lack the fact (a retraining note
    for "what test coverage do you aim for?"). Requiring the answer's own terms ("80",
    "Sandbox") removes those. These are answer terms, not query terms, so the check
    doesn't hand the keyword arms an advantage. Cost: a memory that states the fact in
    words the spec doesn't accept is labeled not relevant.
    """
    return all(re.search(p, memory, re.IGNORECASE) for p in required)


def is_relevant(verdict: bool | None, memory: str, required: list[str]) -> bool:
    return bool(verdict) and holds_answer_terms(memory, required)


def judge(question: str, reference: str, memory: str, model: str) -> bool | None:
    prompt = JUDGE_PROMPT.format(question=question, reference=reference,
                                 memory=memory[:MEMORY_CHARS])
    body = {"model": model, "stream": False, "think": False, "format": "json",
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0, "num_ctx": 4096}}
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return parse_verdict(json.loads(r.read())["message"]["content"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", required=True)
    ap.add_argument("--vectors", help="embed_corpus.py output prefix; omit to pool keyword arms only.")
    ap.add_argument("--questions", default="eval_questions_v2.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool-depth", type=int, default=10)
    ap.add_argument("--judge-model", default="qwen3:8b")
    ap.add_argument("--brain-voice-dir")
    args = ap.parse_args()

    from eval_system import DEFAULT_BV_DIR, load_brain_voice
    bv = load_brain_voice(args.brain_voice_dir or DEFAULT_BV_DIR)
    snap = Snapshot(args.db)
    arms = build_arms(snap, bv, args.vectors)

    with open(args.questions, encoding="utf-8") as f:
        questions = [q for q in map(json.loads, f) if q["domain"] in DOMAINS]

    cache_path = args.out + ".cache.jsonl"
    cache: dict[tuple[str, int], bool | None] = {}
    if os.path.isfile(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            for rec in map(json.loads, f):
                cache[(rec["qid"], rec["mem_id"])] = rec["relevant"]

    gold = []
    with open(cache_path, "a", encoding="utf-8") as cf:
        for n, q in enumerate(questions, 1):
            pool: list[int] = []
            for arm in arms.values():
                pool += [i for i in arm(q["question"], args.pool_depth) if i not in pool]
            new = 0
            for mem_id in pool:
                if (q["id"], mem_id) in cache:
                    continue
                verdict = judge(q["question"], q["reference"], snap.text(mem_id), args.judge_model)
                cache[(q["id"], mem_id)] = verdict
                cf.write(json.dumps({"qid": q["id"], "mem_id": mem_id, "relevant": verdict}) + "\n")
                cf.flush()
                new += 1
            req = q.get("required", [])
            relevant = sorted(i for i in pool if is_relevant(cache[(q["id"], i)], snap.text(i), req))
            unparsed = sum(1 for i in pool if cache[(q["id"], i)] is None)
            gold.append({"id": q["id"], "domain": q["domain"], "pooled": len(pool),
                         "relevant": relevant, "unparsed": unparsed})
            print(f"[{n}/{len(questions)}] {q['id']}: pool {len(pool)} ({new} judged), "
                  f"{len(relevant)} relevant", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g) + "\n")
    no_rel = sum(1 for g in gold if not g["relevant"])
    print(f"wrote {len(gold)} queries -> {args.out} ({no_rel} with no relevant memory in the pool)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
