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
