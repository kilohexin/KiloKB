"""Persist and verify the RRF candidate windows used in reranking experiments."""

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .corpus import Source, load_source
from .dense import MODEL_ID, MODEL_REVISION, BgeM3
from .evaluate import question_cases, question_paths, retrieve_cases
from .markdown import Chunk, chunk_markdown
from .rrf import RRF_CANDIDATES, RRF_K, FusedHit, fuse_rrf


POOL_SIZE = 30


@dataclass(frozen=True)
class FrozenCandidate:
    chunk: Chunk
    chunk_id: str
    input_rank: int
    rrf_score: float


@dataclass(frozen=True)
class FrozenCase:
    qid: str
    group: str
    query: str
    source: Source
    lines: list[int]
    candidates: tuple[FrozenCandidate, ...]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def identity(source: Source, chunk: Chunk) -> str:
    fields = [source.release, source.commit, source.path, chunk.start_line, chunk.end_line]
    return digest(json.dumps(fields, separators=(",", ":")).encode())


def freeze_case(source: Source, qid: str, generation_id: str, hits: list[FusedHit]) -> list[dict]:
    return [
        {
            "qid": qid,
            "release": source.release,
            "generation_id": generation_id,
            "chunk_id": identity(source, hit.chunk),
            "input_rank": index,
            "start_line": hit.chunk.start_line,
            "end_line": hit.chunk.end_line,
            "text_hash": digest(hit.chunk.retrieval_text.encode()),
            "rrf_score": hit.score,
        }
        for index, hit in enumerate(hits, 1)
    ]


def write_rows(path: Path, rows: list[dict]) -> str:
    data = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows).encode()
    path.write_bytes(data)
    return digest(data)


def read_rows(path: Path, checksum: str) -> list[dict]:
    data = path.read_bytes()
    if digest(data) != checksum:
        raise ValueError("Frozen candidate file checksum mismatch")
    return [json.loads(line) for line in data.splitlines()]


def read_case_candidates(
    source: Source, chunks: list[Chunk], rows: list[dict], generation_id: str
) -> tuple[FrozenCandidate, ...]:
    by_id = {identity(source, chunk): chunk for chunk in chunks}
    found = set()
    result = []
    for index, row in enumerate(rows, 1):
        chunk_id = row["chunk_id"]
        chunk = by_id.get(chunk_id)
        if (
            row["generation_id"] != generation_id or row["release"] != source.release
            or row["input_rank"] != index or chunk_id in found or chunk is None
            or [row["start_line"], row["end_line"]] != [chunk.start_line, chunk.end_line]
        ):
            raise ValueError("Frozen candidate identity, source or order mismatch")
        if row["text_hash"] != digest(chunk.retrieval_text.encode()):
            raise ValueError("Frozen candidate text hash mismatch")
        found.add(chunk_id)
        result.append(FrozenCandidate(chunk, chunk_id, index, row["rrf_score"]))
    return tuple(result)


def freeze_inputs(lock: dict, chunk_words: int) -> tuple[dict, list[tuple[str, dict]]]:
    datasets = {group: path.read_bytes() for group, path in question_paths(True).items()}
    settings = {
        "schema": 1,
        "chunk_words": chunk_words,
        "dataset_sha256": {group: digest(raw) for group, raw in datasets.items()},
        "corpus_sha256": {tag: data["sha256"] for tag, data in lock["releases"].items()},
        "source_commits": {tag: data["commit"] for tag, data in lock["releases"].items()},
        "dense": {"model": MODEL_ID, "revision": MODEL_REVISION},
        "rrf": {"k": RRF_K, "branch_limit": RRF_CANDIDATES, "pool_size": POOL_SIZE},
    }
    settings["generation_id"] = digest(json.dumps(settings, sort_keys=True).encode())
    return settings, question_cases(datasets)


def freeze_orm(lock: dict, model: BgeM3, output: Path, chunk_words: int) -> dict:
    manifest, cases = freeze_inputs(lock, chunk_words)
    rows = []
    for row in retrieve_cases(lock, model, chunk_words, RRF_CANDIDATES, cases):
        hits = fuse_rrf(row.bm25, row.dense, POOL_SIZE)
        rows.extend(freeze_case(row.source, row.qid, manifest["generation_id"], hits))
    manifest["candidate_sha256"] = write_rows(output, rows)
    manifest["candidate_count"] = len(rows)
    output.with_suffix(".manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode())
    return manifest


def load_frozen(lock: dict, path: Path) -> tuple[dict, list[FrozenCase]]:
    manifest = json.loads(path.with_suffix(".manifest.json").read_bytes())
    expected, questions = freeze_inputs(lock, manifest["chunk_words"])
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen candidate manifest is incompatible with the current inputs")
    rows = read_rows(path, manifest["candidate_sha256"])
    if len(rows) != manifest["candidate_count"]:
        raise ValueError("Frozen candidate count mismatch")
    by_case = defaultdict(list)
    for row in rows:
        by_case[row["qid"], row["release"]].append(row)
    sources = {tag: load_source(lock, tag) for tag in lock["releases"]}
    chunks = {tag: chunk_markdown(source.text, manifest["chunk_words"]) for tag, source in sources.items()}
    cases = []
    for group, question in questions:
        for tag, lines in question["versions"].items():
            case_rows = by_case.pop((question["qid"], tag), [])
            if not 1 <= len(case_rows) <= POOL_SIZE:
                raise ValueError("Unexpected frozen candidate pool size")
            candidates = read_case_candidates(sources[tag], chunks[tag], case_rows, manifest["generation_id"])
            cases.append(FrozenCase(question["qid"], group, question["query"], sources[tag], lines, candidates))
    if by_case:
        raise ValueError("Frozen file contains an unexpected question or release")
    return manifest, cases
