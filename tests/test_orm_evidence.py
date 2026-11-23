import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
QUERY = "Pydantic ORM 迁移时，orm_mode / from_orm 与 from_attributes / model_validate 怎么用？"


def run_demo(chunk_words, query=QUERY, versions=()):
    return subprocess.run(
        [
            sys.executable, "-m", "kilokb", "orm-demo",
            "--query", query, "--chunk-words", str(chunk_words),
            "--top-k", "5",
            *[argument for version in versions for argument in ("--version", version)],
        ],
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
    )


def test_rechunking_preserves_upstream_evidence_in_both_versions():
    expected = {
        "v1.10.15": {
            "commit": "5476a758c8ac59887dbfa3aa1c3481d0a0e20837",
            "path": "docs/usage/models.md",
            "anchors": [(128, "orm_mode"), (129, "from_orm")],
        },
        "v2.7.4": {
            "commit": "1ab655a54c0defdc5a2a9b13bc90feb6fba78719",
            "path": "docs/concepts/models.md",
            "anchors": [(241, "from_attributes"), (285, "CompanyModel.model_validate")],
        },
    }
    source_lines = {
        version: (ROOT / "corpora" / "pydantic" / f"{version}-models.md")
        .read_text(encoding="utf-8")
        .splitlines()
        for version in expected
    }
    partitions = []
    for size in (48, 112):
        completed = run_demo(size)
        assert completed.returncode == 0, completed.stderr
        report = json.loads(completed.stdout)
        assert set(report["versions"]) == set(expected)
        partition = {}
        for version, evidence in expected.items():
            hits = report["versions"][version]
            assert 1 <= len(hits) <= 5
            partition[version] = set()
            for hit in hits:
                assert hit["release"] == version
                assert hit["commit"] == evidence["commit"]
                assert hit["path"] == evidence["path"]
                assert hit["text"] == "\n".join(
                    source_lines[version][hit["start_line"] - 1:hit["end_line"]]
                )
                prefix = f"https://github.com/pydantic/pydantic/blob/{evidence['commit']}/"
                assert hit["url"] == (
                    f"{prefix}{evidence['path']}#L{hit['start_line']}-L{hit['end_line']}"
                )
            for line, marker in evidence["anchors"]:
                anchor_hits = [
                    hit for hit in hits
                    if hit["start_line"] <= line <= hit["end_line"]
                    and marker in hit["text"].splitlines()[line - hit["start_line"]]
                ]
                assert anchor_hits, (size, version, line)
                partition[version].update(
                    (hit["start_line"], hit["end_line"]) for hit in anchor_hits
                )
        partitions.append(partition)
    assert any(partitions[0][version] != partitions[1][version] for version in expected)


def test_a_single_version_returns_only_its_evidence():
    completed = run_demo(112, "orm_mode", ("v1.10.15",))
    assert completed.returncode == 0, completed.stderr
    versions = json.loads(completed.stdout)["versions"]
    assert set(versions) == {"v1.10.15"}
    assert versions["v1.10.15"]
    assert all(hit["release"] == "v1.10.15" for hit in versions["v1.10.15"])


def test_unknown_search_term_does_not_return_zero_score_chunks():
    completed = run_demo(48, "kilokbfictionalormsetting")
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["versions"] == {"v1.10.15": [], "v2.7.4": []}


@pytest.mark.parametrize("size, versions", [(0, ()), (48, ("v3",))])
def test_invalid_size_or_version_returns_a_usage_error(size, versions):
    assert run_demo(size, versions=versions).returncode == 2
