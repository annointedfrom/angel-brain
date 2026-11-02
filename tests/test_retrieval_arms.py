import os
import sqlite3
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from label_retrieval import is_relevant, parse_verdict  # noqa: E402
from retrieval_arms import (Snapshot, bm25_arm, fts_and_query, fts_or_query,  # noqa: E402
                            hybrid_arm, production_arm)


def _fake_bv():
    """The four brain_voice names the arms rely on, with production-like values."""
    stop = {"what", "is", "the", "my", "do", "i", "use", "for"}

    def distill(q):
        toks = "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in q).split()
        return " ".join(t for t in toks if t not in stop)

    def salient(q, max_kw=6):
        return sorted({w for w in distill(q).split() if len(w) >= 4}, key=len, reverse=True)[:max_kw]

    return types.SimpleNamespace(_distill=distill, _salient_keywords=salient,
                                 _RECALL_EXCLUDE_SOURCES={"brain_voice"},
                                 _CURATED_SOURCES={"vault-rules"}, _CURATED_BONUS=2)


def _snapshot(tmp_path, rows):
    db = str(tmp_path / "k.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE knowledge (id INTEGER PRIMARY KEY, text TEXT, source TEXT, "
                 "tags TEXT, status TEXT)")
    conn.execute("CREATE VIRTUAL TABLE knowledge_fts USING fts5(text, tags, source, "
                 "content=knowledge, content_rowid=id)")
    for i, (text, source, status) in enumerate(rows, start=1):
        conn.execute("INSERT INTO knowledge VALUES (?,?,?,?,?)", (i, text, source, "", status))
        conn.execute("INSERT INTO knowledge_fts(rowid, text, tags, source) VALUES (?,?,?,?)",
                     (i, text, "", source))
    conn.commit()
    conn.close()
    return Snapshot(db)


def test_fts_queries_quote_and_prefix_each_word():
    assert fts_and_query("NAB, bible!") == '"NAB"* "bible"*'
    assert fts_or_query(["nab", "bible"]) == '"nab"* OR "bible"*'
    assert fts_or_query(["--"]) == ""


def test_production_arm_ranks_curated_and_multi_keyword_hits_first(tmp_path):
    snap = _snapshot(tmp_path, [
        ("translation note about bible", "socialhub", "active"),
        ("bible translation rule: NAB", "vault-rules", "active"),
        ("bible translation mentioned in a voice log", "brain_voice", "active"),
        ("bible translation archived", "vault-rules", "archived"),
    ])
    ranked = production_arm(snap, _fake_bv())("What bible translation do I use?", 5)
    assert ranked == [2, 1]  # curated first; voice log and archived rows excluded


def test_bm25_arm_survives_terms_that_match_nothing(tmp_path):
    # An AND query over all three words would return nothing; OR still finds the row.
    snap = _snapshot(tmp_path, [("commit format is type colon description", "vault-rules", "active")])
    assert bm25_arm(snap, _fake_bv())("commit format zebra", 5) == [1]


def test_hybrid_fuses_arms():
    a = lambda q, k: [1, 2, 3][:k]  # noqa: E731
    b = lambda q, k: [3, 2, 9][:k]  # noqa: E731
    # 3 is ranked 3rd and 1st (1/63 + 1/61) and edges out 2, ranked 2nd twice (2/62).
    assert hybrid_arm(a, b)("q", 2) == [3, 2]


def test_parse_verdict():
    assert parse_verdict('{"relevant": true}') is True
    assert parse_verdict('{ "Relevant" : FALSE }') is False
    assert parse_verdict("not json") is None


def test_relevance_needs_judge_yes_and_the_answer_terms():
    mem = "What's your test-coverage target? -> 80%+, TDD workflow."
    assert is_relevant(True, mem, ["80"]) is True
    assert is_relevant(True, "Retrain: extend gold files, gate via eval.", ["80"]) is False
    assert is_relevant(False, mem, ["80"]) is False
    assert is_relevant(None, mem, []) is False
    assert is_relevant(True, mem, []) is True  # no spec terms: the judge alone decides


def test_eval_system_snapshot_recall_returns_memory_texts(tmp_path):
    from eval_system import make_recall
    _snapshot(tmp_path, [("bible translation rule: NAB", "vault-rules", "active"),
                         ("unrelated note", "socialhub", "active")])
    recall = make_recall(_fake_bv(), "production", db=str(tmp_path / "k.db"))
    assert recall("What bible translation do I use?", 5) == ["bible translation rule: NAB"]


def test_eval_system_live_recall_is_brain_voice(tmp_path):
    from eval_system import make_recall
    bv = _fake_bv()
    bv.recall = lambda q, k: ["live"]
    assert make_recall(bv, "live") is bv.recall


def test_log_sources_are_the_automated_feeds():
    from retrieval_arms import is_log_source
    assert all(is_log_source(s) for s in ("socialhub", "socialhub-live", "spector", "post-analysis"))
    assert not any(is_log_source(s) for s in ("vault-rules", "angel-gold", "claude", "faith"))


def test_vector_arm_drops_or_downweights_log_sources(tmp_path, monkeypatch):
    np = __import__("pytest").importorskip("numpy")
    import json
    import retrieval_arms
    snap = _snapshot(tmp_path, [("scanner log", "spector", "active"),
                                ("the fact", "vault-rules", "active"),
                                ("off topic", "vault-rules", "active")])
    prefix = str(tmp_path / "vec")
    # Query points along x; the log memory is the closest match, the fact a near second.
    mat = np.array([[1.0, 0.0], [0.98, 0.199], [0.0, 1.0]], dtype=np.float32)
    np.save(prefix + ".npy", mat)
    with open(prefix + ".ids.json", "w", encoding="utf-8") as f:
        json.dump([1, 2, 3], f)
    monkeypatch.setattr(retrieval_arms, "embed_query", lambda text, model: (1.0, 0.0))
    retrieval_arms._load_vectors.cache_clear()

    def top(**kw):
        return retrieval_arms.vector_arm(snap, prefix, **kw)("q", 3)

    assert top() == [1, 2, 3]
    assert top(log_penalty=0.05) == [2, 1, 3]
    assert top(log_penalty=float("inf")) == [2, 3]


def test_floor_returns_a_prefix_of_the_unfloored_ranking(tmp_path, monkeypatch):
    np = __import__("pytest").importorskip("numpy")
    import json
    import retrieval_arms
    snap = _snapshot(tmp_path, [("strong", "vault-rules", "active"),
                                ("weak", "vault-rules", "active"),
                                ("weaker", "vault-rules", "active")])
    prefix = str(tmp_path / "vec")
    np.save(prefix + ".npy", np.array([[1.0, 0.0], [0.6, 0.8], [0.3, 0.954]], dtype=np.float32))
    with open(prefix + ".ids.json", "w", encoding="utf-8") as f:
        json.dump([1, 2, 3], f)
    monkeypatch.setattr(retrieval_arms, "embed_query", lambda text, model: (1.0, 0.0))
    retrieval_arms._load_vectors.cache_clear()

    full = retrieval_arms.vector_arm(snap, prefix)("q", 3)
    floored = retrieval_arms.vector_arm(snap, prefix, min_sim=0.5)
    assert floored("q", 3) == full[:2] == [1, 2]
    assert retrieval_arms.vector_arm(snap, prefix, min_sim=1.5)("q", 3) == []
    assert [round(s, 2) for _, s in floored.scored("q", 3)] == [1.0, 0.6]

    # The gate keeps every memory, weak ones included, once the top-1 clears it.
    assert retrieval_arms.vector_arm(snap, prefix, gate=0.9)("q", 3) == full == [1, 2, 3]
    assert retrieval_arms.vector_arm(snap, prefix, gate=1.5)("q", 3) == []


def test_live_recall_matches_the_arm_the_eval_measured():
    """Skips in the public release, which doesn't ship the live service."""
    pytest = __import__("pytest")
    from eval_system import DEFAULT_BV_DIR
    import retrieval_arms
    if not os.path.isfile(os.path.join(DEFAULT_BV_DIR, "vector_recall.py")):
        pytest.skip("live service not present")
    sys.path.insert(0, DEFAULT_BV_DIR)
    import vector_recall
    assert vector_recall.QUERY_INSTRUCTION == retrieval_arms.QUERY_INSTRUCTION
    assert vector_recall.LOG_SOURCES == retrieval_arms._LOG_SOURCES
    for source in ("socialhub-live", "spector", "faith", "vault-rules", "claude"):
        assert vector_recall.is_indexable(source) != retrieval_arms.is_log_source(source)


def test_pick_floor_is_second_lowest():
    from calibrate_floor import pick_floor
    assert pick_floor([0.7, 0.5, 0.6, 0.9]) == 0.6


def test_group_by_recall_shares_identical_prompts():
    from eval_paired import group_by_recall
    groups = group_by_recall({"a": ["m1", "m2"], "b": ["m1", "m2"], "c": []})
    assert sorted(sorted(arms) for arms in groups.values()) == [["a", "b"], ["c"]]


def test_paired_stats_counts_unsampled_questions_as_ties():
    from eval_paired import paired_stats
    a = {"q1": 0.0, "q2": 1.0}
    b = {"q1": 1.0, "q2": 1.0}
    stats = paired_stats(a, b, total_questions=4, iters=200, seed=1)
    assert stats["diff"] == 25.0  # one full win over four questions; q3 and q4 tie
    assert stats["wins"] == ["q1"] and stats["losses"] == []
