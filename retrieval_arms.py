"""retrieval_arms.py - the retrievers compared in the A3 retrieval A/B.

Every arm runs against one frozen knowledge-DB snapshot and returns memory ids,
best first. All arms share the signature  retrieve(query, k) -> list[int]  so any
of them can be swapped into the serving path or eval_system.py.

  production  brain_voice.recall's logic: per-keyword FTS AND-search (as the live
              /api/claude/search does), OR-merged, ranked by keyword-match count
              plus a bonus for curated sources
  bm25        one FTS5 query with the keywords OR-ed, ranked by bm25()
  vector      cosine similarity over Ollama embeddings (embed_corpus.py)
  vector-nolog    vector with the automated log sources dropped
  vector-downlog  vector with log sources' cosine lowered by LOG_PENALTY
  vector-nolog-floor  vector-nolog without memories below MIN_SIM (may return none)
  vector-nolog-gate   vector-nolog, but nothing at all if the top-1 is below MIN_SIM
  hybrid      reciprocal-rank fusion of bm25 and vector
"""
from __future__ import annotations

import functools
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


@functools.lru_cache(maxsize=512)  # the vector arms share one embedding per query
def embed_query(text: str, model: str, url: str = "http://localhost:11434") -> tuple[float, ...]:
    body = {"model": model, "input": [QUERY_INSTRUCTION + text], "options": {"num_ctx": 512}}
    req = urllib.request.Request(f"{url}/api/embed", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return tuple(json.loads(r.read())["embeddings"][0])


_LOG_SOURCES = frozenset({"spector", "post-analysis", "terminal-monitor", "tiktok-enrich",
                          "brain-intelligence", "brain-intelligence-digest", "robinhood-bridge"})

# Fixed before scoring, not tuned on the eval: 87 labeled queries are too few to tune on.
LOG_PENALTY = 0.05


def is_log_source(source: str) -> bool:
    """High-volume automated feeds (agent/scanner logs), ~85% of the snapshot. Fact
    memories come from curated and vault sources, which these bury by sheer count."""
    return source.startswith("socialhub") or source in _LOG_SOURCES


@functools.lru_cache(maxsize=2)  # ~420 MB as float32; load once for all vector arms
def _load_vectors(prefix: str):
    import numpy as np

    mat = np.load(prefix + ".npy").astype(np.float32)
    with open(prefix + ".ids.json", encoding="utf-8") as f:
        return mat, json.load(f)


def vector_arm(snap: Snapshot, vectors_prefix: str, model: str = "qwen3-embedding",
               exclude: frozenset[str] = frozenset({"brain_voice"}),
               log_penalty: float = 0.0, min_sim: float | None = None,
               gate: float | None = None) -> Retriever:
    """log_penalty is subtracted from the cosine of log-source memories; inf drops them.
    min_sim drops memories scoring below it, so a query may get fewer than k (or none).
    gate is all-or-nothing: no memories if the top-1 scores below it, else the full top k.
    retrieve.scored(query, k) returns (id, score) pairs for calibration."""
    import numpy as np  # only the vector arm needs numpy

    mat, ids = _load_vectors(vectors_prefix)
    keep = np.array([snap.source.get(i) not in exclude for i in ids])
    is_log = np.array([is_log_source(snap.source.get(i, "")) for i in ids])
    if log_penalty == float("inf"):
        keep &= ~is_log
    offset = np.where(is_log & keep, np.float32(min(log_penalty, 1e9)), np.float32(0))
    cache: dict[str, list[tuple[int, float]]] = {}

    def scored(query: str, k: int) -> list[tuple[int, float]]:
        if query not in cache:
            q = np.asarray(embed_query(query, model), dtype=np.float32)
            q /= np.linalg.norm(q) + 1e-12
            sims = np.where(keep, mat @ q - offset, -np.inf)
            depth = min(100, len(ids))
            top = np.argpartition(-sims, depth - 1)[:depth]
            cache[query] = [(ids[j], float(sims[j])) for j in top[np.argsort(-sims[top])]
                            if np.isfinite(sims[j])]
        hits = cache[query]
        if gate is not None and (not hits or hits[0][1] < gate):
            return []
        if min_sim is not None:
            hits = [h for h in hits if h[1] >= min_sim]
        return hits[:k]

    def retrieve(query: str, k: int) -> list[int]:
        return [i for i, _ in scored(query, k)]

    retrieve.scored = scored
    return retrieve


def hybrid_arm(*arms: Retriever, depth: int = 50, k_rrf: int = 60) -> Retriever:
    def retrieve(query: str, k: int) -> list[int]:
        return rrf([arm(query, depth) for arm in arms], k=k_rrf, limit=k)

    return retrieve


# From calibrate_floor.py on a private set kept apart from the eval: the 2nd-lowest
# relevant-memory cosine. Specific to qwen3-embedding + QUERY_INSTRUCTION; recalibrate
# for any other embedder or instruction.
MIN_SIM = 0.508

ARM_NAMES = ("production", "bm25", "vector", "vector-nolog", "vector-downlog",
             "vector-nolog-floor", "vector-nolog-gate", "hybrid")


def build_arms(snap: Snapshot, bv, vectors_prefix: str | None = None) -> dict[str, Retriever]:
    """Every arm by name; the vector-based ones only when vectors are given."""
    arms = {"production": production_arm(snap, bv), "bm25": bm25_arm(snap, bv)}
    if vectors_prefix:
        arms["vector"] = vector_arm(snap, vectors_prefix)
        arms["vector-nolog"] = vector_arm(snap, vectors_prefix, log_penalty=float("inf"))
        arms["vector-downlog"] = vector_arm(snap, vectors_prefix, log_penalty=LOG_PENALTY)
        arms["vector-nolog-floor"] = vector_arm(snap, vectors_prefix, log_penalty=float("inf"),
                                                min_sim=MIN_SIM)
        arms["vector-nolog-gate"] = vector_arm(snap, vectors_prefix, log_penalty=float("inf"),
                                               gate=MIN_SIM)
        arms["hybrid"] = hybrid_arm(arms["bm25"], arms["vector"])
    return arms
