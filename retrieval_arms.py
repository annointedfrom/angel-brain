"""retrieval_arms.py - the retrievers compared in the A3 retrieval A/B.

Every arm runs against one frozen knowledge-DB snapshot and returns memory ids,
best first. All arms share the signature  retrieve(query, k) -> list[int]  so any
of them can be swapped into the serving path or eval_system.py.

  production  brain_voice.recall's logic: per-keyword FTS AND-search (as the live
              /api/claude/search does), OR-merged, ranked by keyword-match count
              plus a bonus for curated sources
  bm25        one FTS5 query with the keywords OR-ed, ranked by bm25()
  vector      cosine similarity over Ollama embeddings (embed_corpus.py)
  hybrid      reciprocal-rank fusion of bm25 and vector
"""
from __future__ import annotations

import json
import re
import sqlite3
import urllib.request
from collections.abc import Callable

from retrieval_metrics import rrf

Retriever = Callable[[str, int], list[int]]

# Qwen3-Embedding expects an instruction on queries only; documents are embedded bare.
QUERY_INSTRUCTION = ("Instruct: Given a question about the user, find the stored memory "
                     "that answers it\nQuery:")
_FTS_SPECIAL = re.compile(r'["\'\(\)\-\+\*\^\~\{\}\[\]:;!@#\$%&=<>,\./\\]')


def fts_token(word: str) -> str:
    """Prefix token exactly as KnowledgeStore._sanitize_fts_query builds it."""
    return f'"{word}"*'


def fts_and_query(text: str) -> str:
    words = _FTS_SPECIAL.sub(" ", text).split()
    return " ".join(fts_token(w) for w in words)


def fts_or_query(words: list[str]) -> str:
    cleaned = [w for w in (_FTS_SPECIAL.sub(" ", w).strip() for w in words) if w]
    return " OR ".join(fts_token(w) for w in cleaned)


class Snapshot:
    """Read-only view of a knowledge.db snapshot."""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self.source = dict(self.conn.execute(
            "SELECT id, lower(coalesce(source,'')) FROM knowledge WHERE status='active'"))

    def text(self, mem_id: int) -> str:
        row = self.conn.execute("SELECT text FROM knowledge WHERE id=?", (mem_id,)).fetchone()
        return row[0] if row else ""

    def fts(self, match: str, n: int, order: str = "rank") -> list[int]:
        if not match:
            return []
        try:
            rows = self.conn.execute(
                f"SELECT k.id FROM knowledge k JOIN knowledge_fts f ON k.id = f.rowid "
                f"WHERE knowledge_fts MATCH ? AND k.status='active' ORDER BY {order} LIMIT ?",
                (match, n)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [r[0] for r in rows]


def production_arm(snap: Snapshot, bv) -> Retriever:
    """brain_voice.recall, re-pointed from REST at the snapshot. bv is the imported module."""

    def retrieve(query: str, k: int) -> list[int]:
        keywords = bv._salient_keywords(query)
        tally: dict[int, list] = {}  # id -> [match_count, first_keyword_rank]
        for rank, kw in enumerate(keywords):
            hits = [i for i in snap.fts(fts_and_query(kw), k * 2)
                    if snap.source.get(i) not in bv._RECALL_EXCLUDE_SOURCES]
            for i in hits:
                if i in tally:
                    tally[i][0] += 1
                else:
                    tally[i] = [1, rank]

        def score(item):
            mem_id, (count, rank) = item
            bonus = bv._CURATED_BONUS if snap.source.get(mem_id) in bv._CURATED_SOURCES else 0
            return (-(count + bonus), rank)

        return [i for i, _ in sorted(tally.items(), key=score)[:k]]

    return retrieve


def bm25_arm(snap: Snapshot, bv) -> Retriever:
    def retrieve(query: str, k: int) -> list[int]:
        words = [w for w in bv._distill(query).split() if len(w) >= 3]
        hits = snap.fts(fts_or_query(words), k * 3, order="bm25(knowledge_fts)")
        return [i for i in hits if snap.source.get(i) not in bv._RECALL_EXCLUDE_SOURCES][:k]

    return retrieve


def embed_query(text: str, model: str, url: str = "http://localhost:11434") -> list[float]:
    body = {"model": model, "input": [QUERY_INSTRUCTION + text], "options": {"num_ctx": 512}}
    req = urllib.request.Request(f"{url}/api/embed", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["embeddings"][0]


def vector_arm(snap: Snapshot, vectors_prefix: str, model: str = "qwen3-embedding",
               exclude: frozenset[str] = frozenset({"brain_voice"})) -> Retriever:
    import numpy as np  # only the vector arm needs numpy

    mat = np.load(vectors_prefix + ".npy").astype(np.float32)
    with open(vectors_prefix + ".ids.json", encoding="utf-8") as f:
        ids = json.load(f)
    keep = np.array([snap.source.get(i) not in exclude for i in ids])
    cache: dict[str, list[int]] = {}

    def retrieve(query: str, k: int) -> list[int]:
        if query not in cache:
            q = np.asarray(embed_query(query, model), dtype=np.float32)
            q /= np.linalg.norm(q) + 1e-12
            sims = np.where(keep, mat @ q, -np.inf)
            top = np.argpartition(-sims, 100)[:100]
            cache[query] = [ids[j] for j in top[np.argsort(-sims[top])]]
        return cache[query][:k]

    return retrieve


def hybrid_arm(*arms: Retriever, depth: int = 50, k_rrf: int = 60) -> Retriever:
    def retrieve(query: str, k: int) -> list[int]:
        return rrf([arm(query, depth) for arm in arms], k=k_rrf, limit=k)

    return retrieve
