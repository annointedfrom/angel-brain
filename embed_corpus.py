#!/usr/bin/env python3
"""embed_corpus.py - embed every active memory in a knowledge-DB snapshot via Ollama.

Writes <out>.npy (float16, L2-normalized, one row per memory) and <out>.ids.json
(the memory ids in row order). Resumable: rows already in <out>.partial.jsonl are
skipped, so an interrupted run picks up where it stopped.

    python embed_corpus.py --db retrieval_data/knowledge_snapshot.db --out retrieval_data/vectors
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request

import numpy as np

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MAX_CHARS = 1500  # ~400 tokens; long notes are truncated, not chunked


def embed(texts: list[str], model: str, num_ctx: int = 512) -> list[list[float]]:
    body = {"model": model, "input": texts, "truncate": True, "options": {"num_ctx": num_ctx}}
    req = urllib.request.Request(f"{OLLAMA}/api/embed", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())["embeddings"]


def clean(text: str | None) -> str:
    """Truncate and drop characters Ollama rejects (lone surrogates, NULs)."""
    t = (text or "").replace("\x00", " ")[:MAX_CHARS]
    t = t.encode("utf-8", errors="replace").decode("utf-8")
    return t if t.strip() else " "


def embed_one_by_one(texts: list[str], model: str) -> list[list[float]]:
    """Fallback when a batch is rejected: a text the server still refuses gets a zero
    vector (normalizes to zero, so it never ranks) instead of killing the run."""
    vecs: list[list[float] | None] = []
    for t in texts:
        try:
            vecs.append(embed([t], model)[0])
        except urllib.error.HTTPError:
            print(f"  [skip] server rejected a text ({len(t)} chars); zero vector", flush=True)
            vecs.append(None)
    dim = len(next(v for v in vecs if v is not None)) if any(vecs) else 4096
    return [v if v is not None else [0.0] * dim for v in vecs]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True, help="Output prefix (no extension).")
    ap.add_argument("--model", default="qwen3-embedding")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows = conn.execute("SELECT id, text FROM knowledge WHERE status='active' ORDER BY id").fetchall()
    partial = args.out + ".partial.jsonl"
    done: dict[int, list[float]] = {}
    if os.path.isfile(partial):
        with open(partial, encoding="utf-8") as f:
            for ln in f:
                rec = json.loads(ln)
                done[rec["id"]] = rec["v"]
    todo = [(i, t) for i, t in rows if i not in done]
    print(f"{len(rows)} memories, {len(done)} cached, {len(todo)} to embed", flush=True)

    t0 = time.time()
    with open(partial, "a", encoding="utf-8") as f:
        for b in range(0, len(todo), args.batch):
            chunk = todo[b:b + args.batch]
            texts = [clean(t) for _, t in chunk]
            try:
                vecs = embed(texts, args.model)
            except urllib.error.HTTPError:
                vecs = embed_one_by_one(texts, args.model)
            for (i, _), v in zip(chunk, vecs):
                done[i] = v
                f.write(json.dumps({"id": i, "v": v}) + "\n")
            n = b + len(chunk)
            rate = n / max(time.time() - t0, 1e-6)
            print(f"  {n}/{len(todo)}  {rate:.1f}/s  eta {(len(todo) - n) / rate / 60:.1f} min", flush=True)

    ids = [i for i, _ in rows]
    mat = np.asarray([done[i] for i in ids], dtype=np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    np.save(args.out + ".npy", mat.astype(np.float16))
    with open(args.out + ".ids.json", "w", encoding="utf-8") as f:
        json.dump(ids, f)
    print(f"wrote {mat.shape} -> {args.out}.npy", file=sys.stderr)


if __name__ == "__main__":
    main()
