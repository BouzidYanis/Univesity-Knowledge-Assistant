"""
Tests of the retriever metrics, on examples computed by hand.

Run from the project root:
    python -m pytest tests/test_metrics.py
"""
import pytest

from evaluation.metrics import hit_at_k, recall_at_k, reciprocal_rank

# Example of the guide: relevant pages A and B, retrieved C, A, D, B
RELEVANT = {"A", "B"}
RETRIEVED = ["C", "A", "D", "B"]


@pytest.mark.parametrize("k, expected", [(1, 0), (2, 1), (3, 1), (10, 1)])
def test_hit_at_k(k, expected):
    assert hit_at_k(RETRIEVED, RELEVANT, k) == expected


@pytest.mark.parametrize("k, expected", [(1, 0.0), (3, 0.5), (4, 1.0), (10, 1.0)])
def test_recall_at_k(k, expected):
    assert recall_at_k(RETRIEVED, RELEVANT, k) == expected


def test_reciprocal_rank_uses_the_first_relevant_page():
    assert reciprocal_rank(RETRIEVED, RELEVANT) == 0.5


def test_no_relevant_page_retrieved():
    assert hit_at_k(["C", "D"], RELEVANT, 5) == 0
    assert recall_at_k(["C", "D"], RELEVANT, 5) == 0.0
    assert reciprocal_rank(["C", "D"], RELEVANT) == 0.0


def test_empty_retrieved_list():
    assert hit_at_k([], RELEVANT, 5) == 0
    assert recall_at_k([], RELEVANT, 5) == 0.0
    assert reciprocal_rank([], RELEVANT) == 0.0


def test_k_larger_than_the_list():
    assert hit_at_k(["A"], RELEVANT, 10) == 1
    assert recall_at_k(["A"], RELEVANT, 10) == 0.5


def test_duplicates_must_be_removed_before_the_metrics():
    # Exercise of the guide: relevant A, retrieved B, B, A (two chunks of the same page B).
    # Without deduplication the rank of A is 3; on the deduplicated list (B, A) it is 2.
    assert reciprocal_rank(["B", "B", "A"], {"A"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["B", "A"], {"A"}) == 0.5
