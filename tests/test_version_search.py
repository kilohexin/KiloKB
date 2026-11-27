import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
OLD = "v1.10.15"
NEW = "v2.7.4"


@pytest.mark.parametrize(
    "query,mode,versions,expected,origin",
    [
        ("V1 怎样使用 ORM？", "single", (), (OLD,), "query"),
        ("Pydantic 2.7.4 如何读取属性？", "single", (), (NEW,), "query"),
        ("GetterDict", "single", ("V1",), (OLD,), "request"),
        ("from V1 到 V2，model_validate 怎么用？", "compare", (), (OLD, NEW), "query"),
        ("从 V1 到 V2", "compare", (NEW, OLD), (NEW, OLD), "request"),
        ("schema.v1 与 v1_serializer、v1.method", "single", (), (NEW,), "default"),
        ("How does ORM work in V1.", "single", (), (OLD,), "query"),
    ],
)
def test_version_resolution(query, mode, versions, expected, origin):
    from kilokb.corpus import read_lock
    from kilokb.version_scope import resolve_scope

    assert resolve_scope(query, mode, versions, read_lock()) == (expected, origin)


@pytest.mark.parametrize(
    "query,mode,versions,code",
    [
        ("在 v3 里是什么？", "single", (), "unknown_version"),
        ("如何使用？", "single", ("v1.10",), "unknown_version"),
        ("V2 怎么用？", "single", ("v1",), "version_conflict"),
        ("从 V1 到 V2", "single", (), "version_count"),
        ("从 V1 迁移", "compare", ("v1", OLD), "version_count"),
    ],
)
def test_version_resolution_rejects_unclear_scope(query, mode, versions, code):
    from kilokb.corpus import read_lock
    from kilokb.version_scope import SearchError, resolve_scope

    with pytest.raises(SearchError) as failure:
        resolve_scope(query, mode, versions, read_lock())
    assert failure.value.code == code


def test_compare_reports_missing_side_and_keeps_original_citation():
    from kilokb.search import SearchRequest, SearchService

    service = SearchService()
    result = service.search(SearchRequest(query="GetterDict", mode="compare", versions=(OLD, NEW)))
    assert result["resolved_versions"] == [OLD, NEW]
    assert result["status"] == "partial"
    old, new = result["groups"]
    assert old["release"] == OLD and old["evidence_status"] == "available"
    assert new == {"release": NEW, "evidence_status": "insufficient", "hits": []}
    hit = old["hits"][0]
    assert "GetterDict" in hit["raw_text"]
    assert hit["source_commit"] == "5476a758c8ac59887dbfa3aa1c3481d0a0e20837"
    assert hit["source_url"].endswith(f"#L{hit['start_line']}-L{hit['end_line']}")
    trace = service.trace(result["trace_id"])
    assert trace["generation_id"] == result["generation_id"]
    assert trace["groups"][0]["stages"]["bm25"][0]["chunk_id"] == hit["chunk_id"]
    assert trace["groups"][1]["stages"]["bm25"] == []


def test_cli_and_http_return_same_ranked_evidence_and_http_exposes_trace():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from kilokb.api import app

    query = "V1 orm_mode from_orm and V2 from_attributes model_validate"
    cli = subprocess.run(
        [sys.executable, "-m", "kilokb", "search", "--query", query, "--mode", "compare", "--top-k", "2"],
        cwd=ROOT, capture_output=True,
    )
    assert cli.returncode == 0, cli.stderr
    local = json.loads(cli.stdout)
    with TestClient(app) as client:
        response = client.post("/v1/search", json={"query": query, "mode": "compare", "top_k": 2})
        assert response.status_code == 200, response.text
        remote = response.json()
        assert remote["trace_id"] != local["trace_id"]
        assert {key: value for key, value in remote.items() if key != "trace_id"} == {
            key: value for key, value in local.items() if key != "trace_id"
        }
        assert remote["status"] == "complete"
        trace = client.get(f"/v1/traces/{remote['trace_id']}")
        assert trace.status_code == 200
        replay = trace.json()
        assert [group["release"] for group in replay["groups"]] == [OLD, NEW]
        assert replay["request"] == {
            "kb_id": "pydantic", "query": query, "mode": "compare",
            "versions": [], "top_k": 2, "profile": "bm25",
            "abstain": False,
        }
        assert client.get("/v1/traces/missing").status_code == 404


def test_http_error_contains_supported_versions():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from kilokb.api import app

    with TestClient(app) as client:
        response = client.post("/v1/search", json={"query": "V2", "versions": ["v1"]})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "version_conflict"
    assert response.json()["detail"]["supported_versions"] == [OLD, NEW]
