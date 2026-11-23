import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evals" / "pydantic-orm-zh.json"


def test_evidence_ranks_match_line_coverage_and_keep_missing_anchors():
    from kilokb.evaluate import evidence_ranks
    from kilokb.markdown import Chunk

    ranked = [
        (Chunk(10, 12, (), "x"), 0.9),
        (Chunk(4, 8, (), "y"), 0.6),
        (Chunk(12, 14, (), "z"), 0.3),
    ]
    assert evidence_ranks([4, 12, 14, 15], ranked) == {4: 2, 12: 1, 14: 3, 15: None}
    assert evidence_ranks([4], []) == {4: None}


def test_compare_questions_are_chinese_and_cover_both_releases():
    questions = json.loads(DATASET.read_text(encoding="utf-8"))
    assert len({case["qid"] for case in questions}) == len(questions)
    for case in questions:
        query = case["query"]
        assert re.search(r"[\u4e00-\u9fff]", query)
        assert not re.search(r"[A-Za-z0-9_]", query)
    assert {tag for case in questions for tag in case["versions"]} == {"v1.10.15", "v2.7.4"}


@pytest.mark.skipif(os.getenv("KILOKB_RUN_MODEL_TESTS") != "1", reason="requires BGE-M3 weights")
def test_real_bge_m3_retrieves_versioned_evidence_for_chinese_questions():
    completed = subprocess.run(
        [
            sys.executable, "-m", "kilokb", "orm-compare",
            "--device", "cpu", "--local-files-only", "--chunk-words", "112", "--top-k", "5",
        ],
        cwd=ROOT, capture_output=True, timeout=600,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    report = json.loads(completed.stdout)
    rows = {(row["qid"], row["version"]): row for row in report["cases"]}
    expected = {
        ("orm-compare", "v1.10.15"): {"128", "129"},
        ("orm-compare", "v2.7.4"): {"241"},
        ("orm-v1", "v1.10.15"): {"128", "129"},
        ("orm-v2", "v2.7.4"): {"241", "285"},
    }
    assert set(rows) == set(expected)
    for (qid, tag), lines in expected.items():
        row = rows[qid, tag]
        assert row["bm25"]["returned"] == 0
        ranks = row["dense"]["evidence_ranks"]
        assert all(ranks[line] in range(1, 6) for line in lines)
        assert all(rank is None for rank in row["bm25"]["evidence_ranks"].values())
        assert all(hit["release"] == tag for hit in row["dense"]["hits"])
