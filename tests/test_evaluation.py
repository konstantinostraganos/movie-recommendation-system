"""Tests for evaluation metrics — verify math is correct."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    ap_at_k,
)


def test_precision_perfect():
    """All recommendations are relevant."""
    rec = [1, 2, 3, 4, 5]
    rel = {1, 2, 3, 4, 5}
    assert precision_at_k(rec, rel, 5) == 1.0


def test_precision_none():
    """No recommendations are relevant."""
    rec = [1, 2, 3, 4, 5]
    rel = {6, 7, 8}
    assert precision_at_k(rec, rel, 5) == 0.0


def test_precision_partial():
    """2 out of 5 are relevant."""
    rec = [1, 2, 3, 4, 5]
    rel = {2, 4, 6}
    assert precision_at_k(rec, rel, 5) == 0.4


def test_recall_perfect():
    """All relevant items found."""
    rec = [1, 2, 3]
    rel = {1, 2, 3}
    assert recall_at_k(rec, rel, 3) == 1.0


def test_recall_partial():
    """Found 2 out of 3 relevant."""
    rec = [1, 2, 3, 4, 5]
    rel = {2, 4, 6}
    assert abs(recall_at_k(rec, rel, 5) - 2 / 3) < 1e-9


def test_recall_empty_relevant():
    """No relevant items exist."""
    rec = [1, 2, 3]
    rel = set()
    assert recall_at_k(rec, rel, 3) == 0.0


def test_ndcg_perfect():
    """Relevant item at position 1 — best possible."""
    rec = [1, 2, 3]
    rel = {1}
    assert ndcg_at_k(rec, rel, 3) == 1.0


def test_ndcg_last():
    """Relevant item at last position — worst ranking."""
    rec = [2, 3, 1]
    rel = {1}
    result = ndcg_at_k(rec, rel, 3)
    assert result < 1.0
    assert result > 0.0


def test_ndcg_empty():
    """No relevant items."""
    rec = [1, 2, 3]
    rel = set()
    assert ndcg_at_k(rec, rel, 3) == 0.0


def test_hit_rate_hit():
    """At least one relevant item found."""
    rec = [1, 2, 3]
    rel = {3, 5}
    assert hit_rate_at_k(rec, rel, 3) == 1.0


def test_hit_rate_miss():
    """No relevant items found."""
    rec = [1, 2, 3]
    rel = {4, 5}
    assert hit_rate_at_k(rec, rel, 3) == 0.0


def test_ap_perfect_order():
    """All relevant items at the top."""
    rec = [1, 2, 3, 4, 5]
    rel = {1, 2}
    assert ap_at_k(rec, rel, 5) == 1.0


def test_ap_worst_order():
    """Relevant items at the bottom."""
    rec = [3, 4, 5, 1, 2]
    rel = {1, 2}
    result = ap_at_k(rec, rel, 5)
    assert result < 1.0
    assert result > 0.0