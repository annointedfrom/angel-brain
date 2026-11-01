import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval_metrics import (bootstrap_ci, hit_at_k, paired_bootstrap_diff,  # noqa: E402
                               recall_at_k, reciprocal_rank, rrf)


def test_hit_at_k_counts_only_top_k():
    assert hit_at_k([9, 8, 7, 1], {1}, k=3) == 0.0
    assert hit_at_k([9, 8, 7, 1], {1}, k=4) == 1.0


def test_recall_at_k_is_fraction_of_relevant_set():
    assert recall_at_k([1, 2, 3], {1, 3, 5, 6}, k=3) == 0.5
    assert recall_at_k([1], set(), k=5) == 0.0


def test_reciprocal_rank_uses_first_relevant():
    assert reciprocal_rank([4, 2, 1], {1, 2}) == 0.5
    assert reciprocal_rank([4, 5], {1}) == 0.0
    assert reciprocal_rank([4, 5, 1], {1}, k=2) == 0.0


def test_rrf_rewards_agreement_between_rankers():
    # 2 is mid-ranked in both lists; 1 and 3 are each first in only one.
    fused = rrf([[1, 2, 4], [3, 2, 5]], k=60)
    assert fused[0] == 2


def test_rrf_is_deterministic_on_ties_and_respects_limit():
    assert rrf([[1, 2], [2, 1]], k=60) == [1, 2]
    assert rrf([[1, 2, 3]], limit=2) == [1, 2]


def test_bootstrap_ci_brackets_the_mean():
    mean, lo, hi = bootstrap_ci([0.0, 1.0] * 50, iters=2000)
    assert mean == 0.5 and lo < 0.5 < hi


def test_paired_diff_requires_aligned_queries():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1.0], [1.0, 0.0])
    diff, lo, hi = paired_bootstrap_diff([0.0] * 40, [1.0] * 40, iters=500)
    assert diff == lo == hi == 1.0
