import pytest

from kilokb.markdown import Chunk


def test_rrf_rewards_shared_candidates_without_using_branch_scores():
    from kilokb.rrf import fuse_rrf

    lexical = Chunk(1, 2, (), "lexical")
    semantic = Chunk(4, 5, (), "semantic")
    shared = Chunk(8, 10, (), "shared")
    hits = fuse_rrf(
        [(lexical, 30.0), (shared, 0.01)],
        [(semantic, 0.99), (shared, 0.1)],
        top_k=2,
    )
    assert [hit.chunk for hit in hits] == [shared, lexical]
    assert hits[0].ranks == {"bm25": 2, "dense": 2}
    assert hits[0].score == pytest.approx(2 / 62)
    assert hits[1].ranks == {"bm25": 1, "dense": None}
    assert hits[1].score == pytest.approx(1 / 61)


def test_rrf_with_empty_bm25_preserves_dense_order():
    from kilokb.rrf import fuse_rrf

    first = Chunk(100, 110, (), "first")
    second = Chunk(1, 10, (), "second")
    hits = fuse_rrf([], [(first, 0.6), (second, 0.6)], top_k=2)
    assert [hit.chunk for hit in hits] == [first, second]
    assert [hit.ranks for hit in hits] == [{"bm25": None, "dense": 1}, {"bm25": None, "dense": 2}]
    assert fuse_rrf([], [], top_k=5) == []


def test_rrf_can_promote_a_candidate_outside_both_top_fives():
    from kilokb.rrf import fuse_rrf

    shared = Chunk(100, 100, (), "shared at rank six")
    bm25 = [(Chunk(i, i, (), "bm25"), 1.0) for i in range(1, 6)]
    dense = [(Chunk(i, i, (), "dense"), 1.0) for i in range(6, 11)]
    hits = fuse_rrf([*bm25, (shared, 0.5)], [*dense, (shared, 0.5)], top_k=5)
    assert hits[0].chunk == shared
    assert hits[0].ranks == {"bm25": 6, "dense": 6}
