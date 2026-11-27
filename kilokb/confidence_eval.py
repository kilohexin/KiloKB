"""Calibrate on development families and evaluate once on held-out families."""

import hashlib
import json
from pathlib import Path

from .confidence import FALSE_REJECT_LIMIT, PROFILE, SCORE, TOP_K, accepts, choose_threshold, confidence_score, metrics
from .corpus import LOCK
from .evaluate import runtime_metadata, source_hashes
from .search import SearchRequest, SearchService, calibration_contract
from .version_scope import SearchError, resolve_scope


DATASET = LOCK.parent.parent / "evals" / "pydantic-confidence.json"
KINDS = {"answerable", "unanswerable", "version_conflict"}


def load_questions(path: Path, service: SearchService) -> list[dict]:
    rows = json.loads(path.read_bytes())
    seen = set()
    families = {}
    for row in rows:
        qid, family, split, kind = (row[key] for key in ("qid", "family", "split", "kind"))
        if qid in seen or split not in ("dev", "holdout") or kind not in KINDS:
            raise ValueError(f"{qid}: duplicate ID or invalid split/label")
        if not row["query"].strip() or row["mode"] not in ("single", "compare") or not row["note"].strip():
            raise ValueError(f"{qid}: invalid query, mode or annotation")
        seen.add(qid)
        if families.setdefault(family, split) != split:
            raise ValueError(f"{qid}: family appears in both splits")
        try:
            tags, _ = resolve_scope(row["query"], row["mode"], tuple(row["versions"]), service.lock)
        except SearchError as exc:
            if exc.code == "version_conflict" and kind == "version_conflict" and not row["expected"]:
                continue
            raise ValueError(f"{qid}: scope does not match label") from exc
        expected = row["expected"]
        if kind == "version_conflict" or list(expected) != list(tags):
            raise ValueError(f"{qid}: labels do not match the version scope")
        if any(expected.values()) != (kind == "answerable"):
            raise ValueError(f"{qid}: anchors do not match the answerability label")
        for tag, anchors in expected.items():
            source_lines = service.sources[tag].text.splitlines()
            if len(anchors) != len(set(anchors)) or any(
                type(line) is not int or not 1 <= line <= len(source_lines) or not source_lines[line - 1].strip()
                for line in anchors
            ):
                raise ValueError(f"{qid}: invalid anchor in {tag}")
    return rows


def observations(questions: list[dict], service: SearchService) -> tuple[list[dict], list[str]]:
    rows, conflicts = [], []
    for question in questions:
        qid = question["qid"]
        request = SearchRequest(
            query=question["query"], mode=question["mode"], versions=tuple(question["versions"]),
            top_k=TOP_K, profile=PROFILE,
        )
        try:
            result = service.search(request)
        except SearchError as exc:
            if exc.code != "version_conflict" or exc.status != 422 or question["kind"] != "version_conflict":
                raise
            conflicts.append(qid)
            continue
        if question["kind"] == "version_conflict":
            raise ValueError(f"{qid}: conflicting versions reached retrieval")
        trace = service.trace(result["trace_id"])
        for group, group_trace in zip(result["groups"], trace["groups"], strict=True):
            release, hits = group["release"], group["hits"]
            anchors = question["expected"][release]
            found = [line for line in anchors if any(hit["start_line"] <= line <= hit["end_line"] for hit in hits)]
            rows.append({
                "qid": qid, "family": question["family"], "release": release,
                "answerable": bool(anchors), "score": confidence_score(hits, group_trace["stages"]["dense"]),
                "has_hits": bool(hits), "covers_gold": bool(anchors) and found == anchors,
                "gold_lines": anchors, "found_lines": found,
                "source_urls": [hit["source_url"] for hit in hits],
            })
    return rows, conflicts


def run_evaluation(path: Path, device: str, local_files_only: bool) -> dict:
    service = SearchService(device=device, local_files_only=local_files_only)
    questions = load_questions(path, service)
    dev, dev_conflicts = observations([row for row in questions if row["split"] == "dev"], service)
    threshold = choose_threshold(dev)
    policy = {
        "schema": 1, "profile": PROFILE, "top_k": TOP_K, "score": SCORE,
        "threshold": threshold, "generation_id": service.generation_id,
        "contract": calibration_contract(),
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fitted_on": "dev", "dev_false_reject_limit": FALSE_REJECT_LIMIT,
    }
    holdout, holdout_conflicts = observations([row for row in questions if row["split"] == "holdout"], service)
    splits = {}
    for split, rows, conflicts in (("dev", dev, dev_conflicts), ("holdout", holdout, holdout_conflicts)):
        splits[split] = {
            "questions": sum(row["split"] == split for row in questions),
            "resolved_sides": len(rows),
            "version_conflicts_rejected": conflicts,
            "baseline": metrics(rows, None),
            "calibrated": metrics(rows, threshold),
            "observations": [
                {**row, "accepted": accepts(row["score"], threshold)} for row in rows
            ],
        }
    return {
        "dataset": path.name,
        "policy": policy,
        "runtime": runtime_metadata(device),
        "device": device,
        "source_sha256": source_hashes(),
        "splits": splits,
    }
