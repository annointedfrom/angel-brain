"""retrieval_metrics.py - ranking metrics and fusion for the retrieval A/B. Stdlib only.

Each query's gold is a *set* of relevant memory ids, often more than one, so the
headline metrics are hit@k (did any relevant memory reach the top k?) and MRR
(1 / rank of the first relevant one). hit@k is not recall@k when a query has
several relevant memories; recall_at_k is reported alongside for that reason.
"""
from __future__ import annotations

import random
from collections.abc import Iterable, Sequence


def hit_at_k(ranked: Sequence[int], relevant: set[int], k: int = 5) -> float:
    return 1.0 if any(r in relevant for r in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int = 5) -> float:
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def reciprocal_rank(ranked: Sequence[int], relevant: set[int], k: int = 10) -> float:
    for i, r in enumerate(ranked[:k], start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def rrf(rankings: Iterable[Sequence[int]], k: int = 60, limit: int | None = None) -> list[int]:
    """Reciprocal-rank fusion: score(d) = sum over rankings of 1 / (k + rank).

    Ties keep first-seen order, so fusion is deterministic for a fixed input order.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores, key=lambda d: -scores[d])  # stable sort keeps first-seen order
    return fused[:limit] if limit else fused


def bootstrap_ci(values: Sequence[float], iters: int = 5000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI, resampling queries."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    lo = means[int(alpha / 2 * iters)]
    hi = means[min(int((1 - alpha / 2) * iters), iters - 1)]
    return sum(values) / n, lo, hi


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], iters: int = 5000,
                          seed: int = 0) -> tuple[float, float, float]:
    """Mean of (b - a) per query with a bootstrap CI. a and b must be aligned by query."""
    if len(a) != len(b):
        raise ValueError("paired metrics must have one value per query in both arms")
    return bootstrap_ci([y - x for x, y in zip(a, b)], iters=iters, seed=seed)
