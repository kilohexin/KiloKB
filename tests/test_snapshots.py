import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from kilokb.bm25 import tokens
from kilokb.search import SearchRequest


def manifest(root: Path, first: str, *, second: str | None = "Obsolete widget obsoletetoken.", revision=1):
    entries = []
    for name, text in (("manual", first), ("legacy", second)):
        if text is None:
            continue
        raw = text.encode()
        local = root / f"{name}.md"
        local.write_bytes(raw)
        entries.append({
            "path": f"docs/{name}.md", "snapshot": local.name,
            "commit": f"{revision:040x}" if name == "manual" else f"{99:040x}",
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    path = root / "input.json"
    path.write_text(json.dumps({
        "kb_id": "demo", "repo": "example/docs", "default_release": "v2.7.4",
        "aliases": {"v2": "v2.7.4"}, "releases": {"v2.7.4": entries},
    }))
    return path


def encode(texts):
    vectors = np.zeros((len(texts), 1024), dtype=np.float32)
    for vector, text in zip(vectors, texts, strict=True):
        for token in tokens(text):
            index = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 1024
            vector[index] += 1
        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
    return vectors


class Encoder:
    def encode(self, texts):
        return encode(texts)


def results(service, query, profile="bm25"):
    response = service.search(SearchRequest(query, profile=profile, kb_id="demo"))
    return response["generation_id"], response["groups"][0]["hits"]


def test_update_delete_and_incremental_equal_cold_build(tmp_path, monkeypatch):
    from kilokb import search
    from kilokb.snapshots import SnapshotStore
    from kilokb.snapshot_search import PublishedSearch

    monkeypatch.setattr(search, "BgeM3", lambda *args: Encoder())
    store = SnapshotStore(tmp_path / "live.db")
    path = manifest(tmp_path, "# Settings\n\nCurrent mode uses old_token.\n\n# API\n\nStable stable_token.")
    first = store.enqueue(path)
    assert store.enqueue(path) == first
    store.run(first, encode)
    view = PublishedSearch(store)
    old, hits = results(view, "obsoletetoken")
    assert hits[0]["path"] == "docs/legacy.md"
    pinned = view.engine()

    path = manifest(tmp_path, "# Settings\n\nCurrent mode uses new_token.\n\n# API\n\nStable stable_token.",
                    second=None, revision=2)
    updated = store.enqueue(path)
    store.run(updated, encode)
    assert store.run(updated, encode)["state"] == "published"
    stats = store.status(updated)["stats"]
    assert (stats["documents_changed"], stats["documents_deleted"], stats["encoded"], stats["reused"]) == (1, 1, 1, 1)
    generation, hits = results(view, "new_token")
    assert generation != old and hits[0]["source_commit"] == f"{2:040x}"
    assert results(view, "obsoletetoken")[1] == []
    assert results(pinned, "obsoletetoken")[0] == old and results(pinned, "obsoletetoken")[1]
    assert view.trace(view.search(SearchRequest("new_token", kb_id="demo"))["trace_id"])["generation_id"] == generation

    cold = SnapshotStore(tmp_path / "cold.db")
    cold.run(cold.enqueue(path), encode)
    for profile in ("bm25", "dense", "hybrid_rrf"):
        assert results(view, "new_token stable_token", profile) == results(
            PublishedSearch(cold), "new_token stable_token", profile
        )


def test_retry_after_staging_and_reject_stale_publish(tmp_path):
    from kilokb.snapshots import SnapshotStore

    clock = [100.0]
    store = SnapshotStore(tmp_path / "live.db", clock=lambda: clock[0])
    old = store.enqueue(manifest(tmp_path, "original", second=None))
    store.run(old, encode)
    current = store.active_id()
    pending = store.enqueue(manifest(tmp_path, "replacement", second=None, revision=2))
    worker = store.claim(pending, lease_seconds=10)
    store.stage(worker, encode)
    assert store.active_id() == current
    clock[0] += 11
    replacement = store.claim(pending)
    assert not store.publish(worker)
    stats = store.stage(replacement, encode)
    assert stats["encoded"] == 0 and stats["reused"] == 1
    assert store.publish(replacement)
    assert store.status(pending)["attempts"] == 2

    stale = store.enqueue(manifest(tmp_path, "stale", second=None, revision=3))
    late = store.claim(stale)
    store.stage(late, encode)
    latest = store.enqueue(manifest(tmp_path, "latest", second=None, revision=4))
    store.run(latest, encode)
    newest = store.active_id()
    assert not store.publish(late)
    assert store.status(stale)["state"] == "superseded"
    assert store.active_id() == newest


def test_failed_encoding_leaves_active_snapshot_and_retries(tmp_path):
    from kilokb.snapshots import SnapshotStore

    store = SnapshotStore(tmp_path / "live.db")
    first = store.enqueue(manifest(tmp_path, "initial", second=None))
    store.run(first, encode)
    current = store.active_id()
    pending = store.enqueue(manifest(tmp_path, "updated", second=None, revision=2))

    def fail(texts):
        raise RuntimeError("injected encoder outage")

    with pytest.raises(RuntimeError, match="outage"):
        store.run(pending, fail)
    assert store.active_id() == current
    assert store.status(pending)["state"] == "failed"
    assert store.run(pending, encode)["state"] == "published"
    assert store.status(pending)["attempts"] == 2


def test_http_can_pin_historical_generation_after_publish(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from kilokb.api import create_app
    from kilokb.snapshots import SnapshotStore

    database = tmp_path / "live.db"
    store = SnapshotStore(database)
    request = {"query": "obsoletetoken", "kb_id": "demo"}
    with TestClient(create_app(database)) as client:
        assert client.post("/v1/search", json=request).status_code == 503
        store.run(store.enqueue(manifest(tmp_path, "introduction")), encode)
        old = client.post("/v1/search", json=request).json()
        store.run(store.enqueue(manifest(tmp_path, "introduction", second=None)), encode)
        fresh = client.post("/v1/search", json=request).json()
        replay = client.post("/v1/search", json={**request, "generation_id": old["generation_id"]}).json()
        assert old["generation_id"] != fresh["generation_id"]
        assert fresh["status"] == "insufficient"
        assert replay["groups"] == old["groups"]
        assert client.get(f"/v1/traces/{old['trace_id']}").json()["generation_id"] == old["generation_id"]
        assert client.post("/v1/search", json={**request, "generation_id": "f" * 64}).status_code == 404


def test_moved_source_reuses_vectors_and_updates_citation(tmp_path):
    from kilokb.snapshots import SnapshotStore
    from kilokb.snapshot_search import PublishedSearch

    store = SnapshotStore(tmp_path / "live.db")
    text = "# Settings\n\nStable keyword."
    store.run(store.enqueue(manifest(tmp_path, text, second=None)), encode)
    before = results(PublishedSearch(store), "keyword")[1][0]
    job = store.enqueue(manifest(tmp_path, "\n" + text, second=None, revision=2))

    def unexpected_encode(texts):
        raise AssertionError(f"Changed source location triggered re-encoding: {texts}")

    stats = store.run(job, unexpected_encode)["stats"]
    after = results(PublishedSearch(store), "keyword")[1][0]
    assert stats["encoded"] == 0 and stats["reused"] == 1
    assert before["start_line"] == 1 and after["start_line"] == 2
    assert before["chunk_id"] != after["chunk_id"]
    assert after["source_url"].endswith("#L2-L4")


def test_same_path_from_two_repositories_keeps_distinct_citations(tmp_path):
    from kilokb.snapshots import SnapshotStore
    from kilokb.snapshot_search import PublishedSearch

    path = manifest(tmp_path, "same keyword", second=None)
    data = json.loads(path.read_bytes())
    document = data["releases"]["v2.7.4"][0]
    data["releases"]["v2.7.4"].append({**document, "repo": "example/other"})
    path.write_text(json.dumps(data))
    store = SnapshotStore(tmp_path / "live.db")
    stats = store.run(store.enqueue(path), encode)["stats"]
    hits = results(PublishedSearch(store), "keyword")[1]
    assert len({hit["chunk_id"] for hit in hits}) == 2
    assert {hit["source_url"].split("/blob/")[0] for hit in hits} == {
        "https://github.com/example/docs", "https://github.com/example/other",
    }
    assert stats["encoded"] == 1 and stats["chunks_written"] == 2


@pytest.mark.skipif(os.getenv("KILOKB_RUN_MODEL_TESTS") != "1", reason="requires local BGE-M3 weights")
def test_real_dense_snapshot_retry_matches_clean_rebuild(tmp_path, monkeypatch):
    from kilokb import search
    from kilokb.dense import BgeM3
    from kilokb.snapshots import SnapshotStore
    from kilokb.snapshot_search import PublishedSearch

    model = BgeM3("cpu", local_files_only=True)
    monkeypatch.setattr(search, "BgeM3", lambda *args: model)
    original = "# Configuration\n\nUse the old validator.\n\n# Shared\n\nThe original defaults apply."
    updated = original.replace("old validator", "new validator")
    clock = [100.0]
    store = SnapshotStore(tmp_path / "live.db", clock=lambda: clock[0])
    store.run(store.enqueue(manifest(tmp_path, original, second="Use the discontinued legacy adapter.")), model.encode)
    view = PublishedSearch(store)
    previous = view.engine()
    final = manifest(tmp_path, updated, second=None, revision=2)
    job = store.enqueue(final)
    stopped = store.claim(job, lease_seconds=5)
    initial_stage = store.stage(stopped, model.encode)
    assert initial_stage["encoded"] == 1 and initial_stage["reused"] == 1
    assert view.engine().generation_id == previous.generation_id
    clock[0] += 6
    assert store.run(job, model.encode)["stats"]["encoded"] == 0
    cold = SnapshotStore(tmp_path / "cold.db")
    assert cold.run(cold.enqueue(final), model.encode)["stats"]["encoded"] == 2
    for profile in ("bm25", "dense", "hybrid_rrf"):
        actual_gen, actual = results(view, "How do I use the new validator?", profile)
        expected_gen, expected = results(PublishedSearch(cold), "How do I use the new validator?", profile)
        assert actual_gen == expected_gen
        assert [hit["chunk_id"] for hit in actual] == [hit["chunk_id"] for hit in expected]
        assert [hit["final_score"] for hit in actual] == pytest.approx(
            [hit["final_score"] for hit in expected], abs=2e-6
        )
