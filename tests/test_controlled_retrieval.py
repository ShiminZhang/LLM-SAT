"""Tests for subcategory-controlled retrieval (paper §3.2).

Pure-logic tests: no FAISS index or embedding model is loaded.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from experience_pool.manager import (
    SUBCATEGORY_INDEX,
    _rerank_by_subcategory,
    _subcategory_par2,
)
from experience_pool.types import (
    MutationExperienceRecord,
    OutcomeLabel,
    RetrievedExperience,
)


def _hit(record_id, member_raw_par2, score=0.5):
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
            member_raw_par2=member_raw_par2,
        ),
    )


def test_subcategory_index_order_matches_raw_par2_convention():
    # AlgorithmResult.raw_par2_score is [easy, hard, sat, unsat, all]
    # (see evaluation.py collect_results); the index map must agree.
    assert SUBCATEGORY_INDEX == {"easy": 0, "hard": 1, "sat": 2, "unsat": 3, "all": 4}


def test_good_sorts_ascending_by_target_subcategory():
    hits = [
        _hit("worse_hard", [100, 900, 500, 500, 500]),
        _hit("best_hard", [400, 100, 500, 500, 500]),
        _hit("mid_hard", [200, 400, 500, 500, 500]),
    ]
    ranked = _rerank_by_subcategory(hits, OutcomeLabel.GOOD, "hard")
    assert [h.record_id for h in ranked] == ["best_hard", "mid_hard", "worse_hard"]
    # same hits ranked on "easy" give a different order — the steering is real
    ranked_easy = _rerank_by_subcategory(hits, OutcomeLabel.GOOD, "easy")
    assert [h.record_id for h in ranked_easy] == ["worse_hard", "mid_hard", "best_hard"]


def test_bad_sorts_descending_worst_first():
    hits = [
        _hit("mild_regression", [0, 0, 300, 0, 0]),
        _hit("severe_regression", [0, 0, 1200, 0, 0]),
    ]
    ranked = _rerank_by_subcategory(hits, OutcomeLabel.BAD, "sat")
    assert [h.record_id for h in ranked] == ["severe_regression", "mild_regression"]


def test_records_without_scores_keep_similarity_order_at_end():
    hits = [
        _hit("legacy_a", None, score=0.9),
        _hit("scored", [10, 10, 10, 10, 10], score=0.1),
        _hit("legacy_b", None, score=0.8),
    ]
    ranked = _rerank_by_subcategory(hits, OutcomeLabel.GOOD, "unsat")
    assert [h.record_id for h in ranked] == ["scored", "legacy_a", "legacy_b"]


def test_malformed_arrays_treated_as_unscored():
    assert _subcategory_par2(_hit("short", [1, 2, 3]), 0) is None
    assert _subcategory_par2(_hit("nan", [float("nan")] * 5), 0) is None
    assert _subcategory_par2(_hit("none_entry", [None, 2, 3, 4, 5]), 0) is None
    assert _subcategory_par2(_hit("ok", [1, 2, 3, 4, 5]), 3) == 4.0


def test_old_persisted_payloads_still_deserialize():
    # Records persisted before the schema carried PAR-2 arrays must load.
    old_payload = {
        "leader_algorithm_description": "L",
        "member_algorithm_description": "M",
        "step": "Step 2: y",
        "analysis": "improved",
    }
    rec = MutationExperienceRecord(**old_payload)
    assert rec.leader_raw_par2 is None and rec.member_raw_par2 is None


def test_unknown_subcategory_raises():
    with pytest.raises(KeyError):
        _rerank_by_subcategory([_hit("x", [1, 2, 3, 4, 5])], OutcomeLabel.GOOD, "medium")
