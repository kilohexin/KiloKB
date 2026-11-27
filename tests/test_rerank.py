import json
from dataclasses import replace
from pathlib import Path

import pytest

from kilokb.corpus import load_source, read_lock
from kilokb.markdown import chunk_markdown
from kilokb.rrf import FusedHit


def test_freeze_replays_exact_ordered_source_windows_and_rejects_modified_file(tmp_path):
    from kilokb.frozen import FrozenCandidate, freeze_case, read_case_candidates, read_rows, write_rows

    lock = read_lock()
    source = load_source(lock, "v2.7.4")
    chunks = chunk_markdown(source.text, 112)
    chosen = [chunks[2], chunks[0]]
    rows = freeze_case(source, "orm-compare", "generation", [
        FusedHit(chosen[0], 0.015, {"bm25": None, "dense": 4}),
        FusedHit(chosen[1], 0.013, {"bm25": None, "dense": 5}),
    ])
    target = tmp_path / "candidates.jsonl"
    checksum = write_rows(target, rows)
    frozen = read_case_candidates(source, chunks, read_rows(target, checksum), "generation")
    assert all(isinstance(hit, FrozenCandidate) for hit in frozen)
    assert [hit.chunk for hit in frozen] == chosen
    assert [hit.input_rank for hit in frozen] == [1, 2]
    assert frozen[0].chunk_id != frozen[1].chunk_id

    bad_rows = [dict(rows[0], text_hash="invalid"), rows[1]]
    with pytest.raises(ValueError, match="text hash"):
        read_case_candidates(source, chunks, bad_rows, "generation")
    target.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        read_rows(target, checksum)


def test_rerank_reorders_only_frozen_candidates_and_keeps_rrf_order_for_ties():
    from kilokb.frozen import FrozenCandidate
    from kilokb.rerank import reorder

    source = load_source(read_lock(), "v2.7.4")
    chunks = chunk_markdown(source.text, 112)
    first = FrozenCandidate(chunks[0], "first", 1, 0.03)
    second = replace(first, chunk=chunks[1], chunk_id="second", input_rank=2)
    third = replace(first, chunk=chunks[2], chunk_id="third", input_rank=3)
    frozen = (first, second, third)
    assert reorder(frozen, [0.1, 0.9, 0.9], 2) == [(second, 0.9), (third, 0.9)]
    assert reorder(frozen, [0.9, 0.9, 0.9], 2) == [(first, 0.9), (second, 0.9)]
    with pytest.raises(ValueError):
        reorder(frozen, [0.9], 2)


def test_maxsim_averages_each_query_tokens_best_passage_match():
    torch = pytest.importorskip("torch")
    from kilokb.rerank import maxsim_score

    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    passage = torch.tensor([[1.0, 0.0], [0.6, 0.8], [-1.0, 0.0]])
    assert maxsim_score(query, passage) == pytest.approx(0.9)


def test_published_reranker_report_replays_frozen_ids_and_source_anchors():
    from kilokb.frozen import load_frozen

    reports = Path(__file__).resolve().parents[1] / "reports"
    manifest, cases = load_frozen(read_lock(), reports / "pydantic-orm-rrf-candidates.jsonl")
    report = json.loads((reports / "pydantic-orm-rerank-cuda.json").read_bytes())
    assert report["freeze"] == manifest
    assert manifest["candidate_count"] == 240
    assert len(report["cases"]) == len(cases) == 8
    for case, result in zip(cases, report["cases"], strict=True):
        assert (result["qid"], result["version"]) == (case.qid, case.source.release)
        candidates = {candidate.chunk_id: candidate for candidate in case.candidates}
        assert len(candidates) == result["candidate_count"] == 30
        for engine in ("rrf", "cross_encoder", "maxsim"):
            hits = result[engine]["hits"]
            assert len(hits) == len({hit["chunk_id"] for hit in hits}) == 5
            assert len(result[engine]["latency_ms"]) == 5
            ranks = {str(line): None for line in case.lines}
            for position, hit in enumerate(hits, 1):
                frozen = candidates[hit["chunk_id"]]
                assert hit["input_rank"] == frozen.input_rank
                assert hit["url"] == case.source.hit(frozen.chunk, hit["score"])["url"]
                for line in case.lines:
                    if frozen.chunk.start_line <= line <= frozen.chunk.end_line and ranks[str(line)] is None:
                        ranks[str(line)] = position
            assert result[engine]["evidence_ranks"] == ranks
    subject = next(
        row for row in report["cases"]
        if row["qid"] == "orm-compare" and row["version"] == "v2.7.4"
    )
    assert subject["input_evidence_ranks"]["285"] == 8
    assert subject["rrf"]["evidence_ranks"]["285"] is None
    assert subject["cross_encoder"]["evidence_ranks"]["285"] is None
    recovered_rank = subject["maxsim"]["evidence_ranks"]["285"]
    assert recovered_rank == 2
    recovered_hit = subject["maxsim"]["hits"][recovered_rank - 1]
    assert recovered_hit["input_rank"] == 8
    assert "#L250-L291" in recovered_hit["url"]
    assert all(report["latency"][engine]["samples"] == 40
               for engine in ("rrf", "cross_encoder", "maxsim"))
