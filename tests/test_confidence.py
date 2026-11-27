import hashlib
import json
from pathlib import Path

import pytest


def test_score_uses_dense_cosine_only_for_returned_evidence():
    from kilokb.confidence import confidence_score

    hits = [{"chunk_id": "a"}, {"chunk_id": "missing"}]
    dense = [{"chunk_id": "a", "score": 0.7}, {"chunk_id": "excluded", "score": 0.95}]
    assert confidence_score(hits, dense) == 0.7
    assert confidence_score([], dense) is None


def test_threshold_is_selected_on_dev_under_false_reject_budget():
    from kilokb.confidence import choose_threshold, metrics

    def observation(gold, score):
        return {"answerable": gold, "score": score, "has_hits": True, "covers_gold": gold}

    dev = [observation(True, 0.82), observation(True, 0.75),
           observation(False, 0.70), observation(False, 0.76)]
    holdout = [observation(True, 0.74), observation(False, 0.75)]
    threshold = choose_threshold(dev)
    assert threshold == 0.75
    assert metrics(dev, threshold)["false_reject"] == {"count": 0, "total": 2, "rate": 0.0}
    assert metrics(holdout, threshold)["false_reject"] == {"count": 1, "total": 1, "rate": 1.0}
    assert metrics(holdout, threshold)["false_accept"] == {"count": 1, "total": 1, "rate": 1.0}


def test_corpus_eval_has_heldout_families_and_resolved_evidence_anchors():
    from kilokb.confidence_eval import DATASET, load_questions
    from kilokb.search import SearchService

    rows = load_questions(DATASET, SearchService())
    assert len(rows) == 50
    assert sum(row["split"] == "dev" for row in rows) == 30
    families = {split: {row["family"] for row in rows if row["split"] == split}
                for split in ("dev", "holdout")}
    assert families["dev"].isdisjoint(families["holdout"])
    for split in families:
        assert {row["kind"] for row in rows if row["split"] == split} == {
            "answerable", "unanswerable", "version_conflict",
        }


def test_gate_can_remove_one_side_and_keeps_rejected_candidates_in_trace(monkeypatch):
    from kilokb.search import SearchRequest, SearchService

    service = SearchService()
    selected = ("v1.10.15", "v2.7.4")
    dense = {
        tag: ([(service.chunks[tag][0], score)], 0.0)
        for tag, score in zip(selected, (0.8, 0.6), strict=True)
    }
    monkeypatch.setattr(service, "_dense_candidates", lambda query, tags: (dense, 0.0))
    monkeypatch.setattr(service, "_threshold", lambda request: 0.75)
    result = service.search(SearchRequest(
        query="怎样使用它？", mode="compare", versions=selected, profile="hybrid_rrf", abstain=True,
    ))
    assert result["status"] == "partial"
    assert len(result["groups"][0]["hits"]) == 1
    assert result["groups"][1]["hits"] == []
    decision = result["groups"][1]["confidence"]
    assert decision["accepted"] is False and decision["score"] == 0.6
    assert service.trace(result["trace_id"])["groups"][1]["stages"]["rrf"]


def test_opt_in_rejects_uncalibrated_search_contract():
    from kilokb.search import SearchRequest, SearchService
    from kilokb.version_scope import SearchError

    with pytest.raises(SearchError) as error:
        SearchService().search(SearchRequest(query="GetterDict", abstain=True))
    assert error.value.code == "uncalibrated_profile"


def test_published_policy_and_metrics_reproduce_from_recorded_observations():
    from kilokb.confidence import choose_threshold, metrics
    from kilokb.confidence_eval import DATASET
    from kilokb.search import CONFIDENCE_POLICY, SearchRequest, SearchService

    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "reports/pydantic-confidence-cuda.json").read_bytes())
    policy = json.loads(CONFIDENCE_POLICY.read_bytes())
    assert policy == report["policy"]
    assert policy["dataset_sha256"] == hashlib.sha256(DATASET.read_bytes()).hexdigest()
    threshold = choose_threshold(report["splits"]["dev"]["observations"])
    assert SearchService()._threshold(SearchRequest(query="ORM", profile="hybrid_rrf")) == threshold
    for split in report["splits"].values():
        assert split["baseline"] == metrics(split["observations"], None)
        assert split["calibrated"] == metrics(split["observations"], threshold)
        assert len(split["version_conflicts_rejected"]) == 3


def test_rejects_stale_ranking_contract_before_loading_model(tmp_path, monkeypatch):
    from kilokb import search
    from kilokb.version_scope import SearchError

    policy = json.loads(search.CONFIDENCE_POLICY.read_bytes())
    policy.setdefault("contract", {})["rrf_k"] = -1
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps(policy), encoding="utf-8")
    monkeypatch.setattr(search, "CONFIDENCE_POLICY", stale)
    service = search.SearchService()
    with pytest.raises(SearchError) as error:
        service.search(search.SearchRequest(query="ORM", profile="hybrid_rrf", abstain=True))
    assert error.value.code == "confidence_policy_unavailable"
    assert service.model is None
