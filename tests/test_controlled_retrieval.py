"""Tests for subcategory-controlled retrieval (converged implementation).

After merging the two parallel implementations (upstream's delta-based
orchestrator rerank + this branch's env/CLI ergonomics), the mechanism is:
overfetch by similarity, re-rank by leader−member PAR-2 delta on the target
category, keep top-k. Pure-logic tests: no FAISS/embedding model loaded.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from experience_pool.pools import _par2_from_raw_list
from experience_pool.types import (
    MutationExperienceRecord,
    OutcomeLabel,
    Par2Scores,
    RetrievedExperience,
)
from llmsat.pipelines.parallel_orchestrator import (
    _parse_par2_filter_env,
    _rerank_by_par2_delta,
)


def _hit(record_id, leader_par2=None, member_par2=None, score=0.5):
    return RetrievedExperience(
        record_id=record_id,
        pool_name="mutation",
        outcome=OutcomeLabel.GOOD,
        score=score,
        payload=MutationExperienceRecord(
            leader_algorithm_description="leader",
            member_algorithm_description=f"member {record_id}",
            step="Step 1: x",
            analysis="a",
            leader_par2=leader_par2,
            member_par2=member_par2,
        ),
    )


def _scores(**kw):
    base = dict(sat=None, unsat=None, hard=None, easy=None, overall=None)
    base.update(kw)
    return Par2Scores(**base)


# --- env parsing -----------------------------------------------------------

def test_boolean_style_env(monkeypatch):
    for var in ("SAT", "UNSAT", "HARD", "EASY", "TARGET_SUBCATEGORY", "LLMSAT_TARGET_SUBCATEGORY"):
        monkeypatch.delenv(var, raising=False)
    assert _parse_par2_filter_env() is None
    monkeypatch.setenv("HARD", "1")
    assert _parse_par2_filter_env() == "hard"


def test_single_var_style_env(monkeypatch):
    for var in ("SAT", "UNSAT", "HARD", "EASY", "LLMSAT_TARGET_SUBCATEGORY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TARGET_SUBCATEGORY", "unsat")
    assert _parse_par2_filter_env() == "unsat"
    monkeypatch.delenv("TARGET_SUBCATEGORY")
    monkeypatch.setenv("LLMSAT_TARGET_SUBCATEGORY", "EASY")
    assert _parse_par2_filter_env() == "easy"


def test_agreeing_styles_are_fine_conflicts_raise(monkeypatch):
    for var in ("SAT", "UNSAT", "HARD", "EASY", "LLMSAT_TARGET_SUBCATEGORY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SAT", "1")
    monkeypatch.setenv("TARGET_SUBCATEGORY", "sat")
    assert _parse_par2_filter_env() == "sat"
    monkeypatch.setenv("TARGET_SUBCATEGORY", "hard")
    with pytest.raises(ValueError, match="conflicting"):
        _parse_par2_filter_env()


def test_invalid_single_var_raises(monkeypatch):
    for var in ("SAT", "UNSAT", "HARD", "EASY", "LLMSAT_TARGET_SUBCATEGORY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TARGET_SUBCATEGORY", "medium")
    with pytest.raises(ValueError, match="sat|unsat|hard|easy"):
        _parse_par2_filter_env()


# --- delta rerank ----------------------------------------------------------

def test_good_ranks_strongest_improvement_first():
    hits = [
        _hit("small_gain", _scores(hard=500), _scores(hard=450)),   # delta 50
        _hit("big_gain", _scores(hard=900), _scores(hard=300)),     # delta 600
        _hit("regression", _scores(hard=400), _scores(hard=500)),   # delta -100
    ]
    scored, dropped = _rerank_by_par2_delta(hits, "hard", descending=True)
    assert dropped == 0
    assert [h.record_id for _, h in scored] == ["big_gain", "small_gain", "regression"]


def test_bad_ranks_worst_regression_first():
    hits = [
        _hit("mild", _scores(sat=300), _scores(sat=350)),     # delta -50
        _hit("severe", _scores(sat=300), _scores(sat=900)),   # delta -600
    ]
    scored, _ = _rerank_by_par2_delta(hits, "sat", descending=False)
    assert [h.record_id for _, h in scored] == ["severe", "mild"]


def test_category_changes_ordering():
    hits = [
        _hit("hard_specialist", _scores(hard=900, easy=100), _scores(hard=200, easy=100)),
        _hit("easy_specialist", _scores(hard=500, easy=800), _scores(hard=500, easy=100)),
    ]
    by_hard, _ = _rerank_by_par2_delta(hits, "hard", descending=True)
    by_easy, _ = _rerank_by_par2_delta(hits, "easy", descending=True)
    assert by_hard[0][1].record_id == "hard_specialist"
    assert by_easy[0][1].record_id == "easy_specialist"


def test_missing_scores_are_dropped_and_counted():
    hits = [
        _hit("legacy_no_scores"),                                       # both None
        _hit("half", _scores(unsat=100), None),                         # member missing
        _hit("cat_missing", _scores(unsat=None), _scores(unsat=50)),    # category None
        _hit("ok", _scores(unsat=500), _scores(unsat=100)),
    ]
    scored, dropped = _rerank_by_par2_delta(hits, "unsat", descending=True)
    assert dropped == 3
    assert [h.record_id for _, h in scored] == ["ok"]


# --- storage conversion ----------------------------------------------------

def test_par2_from_raw_list_field_mapping():
    # raw_par2_score order is [easy, hard, sat, unsat, all]
    p = _par2_from_raw_list([1.0, 2.0, 3.0, 4.0, 5.0])
    assert (p.easy, p.hard, p.sat, p.unsat, p.overall) == (1.0, 2.0, 3.0, 4.0, 5.0)
    assert _par2_from_raw_list(None) is None
    assert _par2_from_raw_list([1, 2, 3]) is None
    assert _par2_from_raw_list([1.0, None, 3.0, 4.0, 5.0]).hard is None


def test_old_persisted_payloads_still_deserialize():
    old_payload = {
        "leader_algorithm_description": "L",
        "member_algorithm_description": "M",
        "step": "Step 2: y",
        "analysis": "improved",
    }
    rec = MutationExperienceRecord(**old_payload)
    assert rec.leader_par2 is None and rec.member_par2 is None and rec.extra == {}
