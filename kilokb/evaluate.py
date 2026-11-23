"""Compare lexical and semantic retrieval on fixed Chinese ORM questions."""

import hashlib
import json
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from .bm25 import search_bm25
from .corpus import LOCK, Source, load_source
from .dense import BATCH_SIZE, MAX_TOKENS, MODEL_ID, MODEL_REVISION, BgeM3, rank_dense
from .markdown import Chunk, chunk_markdown
from .rrf import RRF_CANDIDATES, RRF_K, fuse_rrf


DATASET = LOCK.parent.parent / "evals" / "pydantic-orm-zh.json"
API_DATASET = DATASET.with_name("pydantic-orm-api.json")


@dataclass(frozen=True)
class RetrievalRow:
    group: str
    qid: str
    query: str
    source: Source
    lines: list[int]
    bm25: list[tuple[Chunk, float]]
    dense: list[tuple[Chunk, float]]


def question_paths(include_api: bool) -> dict[str, Path]:
    return {"zh": DATASET, "api": API_DATASET} if include_api else {"zh": DATASET}


def question_cases(datasets: dict[str, bytes]) -> list[tuple[str, dict]]:
    return [(group, case) for group, raw in datasets.items() for case in json.loads(raw)]


def retrieve_cases(
    lock: dict, model: BgeM3, chunk_words: int, candidate_limit: int, cases: list[tuple[str, dict]]
):
    releases = dict.fromkeys(tag for _, case in cases for tag in case["versions"])
    sources = {tag: load_source(lock, tag) for tag in releases}
    chunks = {tag: chunk_markdown(source.text, chunk_words) for tag, source in sources.items()}
    vectors = {tag: model.encode([chunk.retrieval_text for chunk in chunks[tag]]) for tag in releases}
    queries = model.encode([case["query"] for _, case in cases])
    for (group, case), query_vector in zip(cases, queries, strict=True):
        for tag, lines in case["versions"].items():
            yield RetrievalRow(
                group, case["qid"], case["query"], sources[tag], lines,
                search_bm25(case["query"], chunks[tag], candidate_limit),
                rank_dense(chunks[tag], vectors[tag], query_vector, candidate_limit),
            )


def source_hashes() -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }


def runtime_metadata(device: str) -> dict:
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
        "torch_threads": torch.get_num_threads(),
        "dependencies": {name: version(name) for name in (
            "torch", "transformers", "sentence-transformers", "huggingface-hub", "numpy", "safetensors",
        )},
    }


def evidence_ranks(lines: list[int], hits: list[tuple[Chunk, float]]) -> dict[int, int | None]:
    return {
        line: next((
            rank for rank, (chunk, _) in enumerate(hits, 1)
            if chunk.start_line <= line <= chunk.end_line
        ), None)
        for line in lines
    }


def measure(
    source: Source, lines: list[int], hits: list[tuple[Chunk, float]],
    *, ranks: list[dict[str, int | None]] | None = None,
) -> dict:
    return {
        "returned": len(hits),
        "evidence_ranks": evidence_ranks(lines, hits),
        "hits": [
            source.hit(chunk, score, ranks=ranks[index] if ranks is not None else None)
            for index, (chunk, score) in enumerate(hits)
        ],
    }


def summarize(rows: list[dict], engines: tuple[str, ...]) -> dict:
    return {
        engine: {
            "all_evidence_cases": sum(
                all(rank is not None for rank in row[engine]["evidence_ranks"].values()) for row in rows
            ),
            "total_cases": len(rows),
            "anchors_found": sum(
                rank is not None for row in rows for rank in row[engine]["evidence_ranks"].values()
            ),
            "total_anchors": sum(len(row[engine]["evidence_ranks"]) for row in rows),
        }
        for engine in engines
    }


def compare_orm(
    lock: dict, model: BgeM3, chunk_words: int, top_k: int, device: str,
    *, include_hybrid: bool = False,
) -> dict:
    paths = question_paths(include_hybrid)
    datasets = {group: path.read_bytes() for group, path in paths.items()}
    cases = question_cases(datasets)
    releases = dict.fromkeys(tag for _, case in cases for tag in case["versions"])
    candidate_limit = max(RRF_CANDIDATES, top_k) if include_hybrid else top_k

    rows = []
    for result in retrieve_cases(lock, model, chunk_words, candidate_limit, cases):
        source, lines = result.source, result.lines
        row = {
            "qid": result.qid,
            "query": result.query,
            "version": source.release,
            "source_commit": source.commit,
            "source_path": source.path,
            "bm25": measure(source, lines, result.bm25[:top_k]),
            "dense": measure(source, lines, result.dense[:top_k]),
        }
        if include_hybrid:
            fused = fuse_rrf(result.bm25, result.dense, top_k)
            row["question_set"] = result.group
            row["hybrid"] = measure(
                source, lines, [(hit.chunk, hit.score) for hit in fused],
                ranks=[hit.ranks for hit in fused],
            )
        rows.append(row)

    engines = ("bm25", "dense", "hybrid") if include_hybrid else ("bm25", "dense")
    report = {
        "dataset": DATASET.name,
        "dataset_sha256": hashlib.sha256(datasets["zh"]).hexdigest(),
        "corpus_sha256": {tag: lock["releases"][tag]["sha256"] for tag in releases},
        "source_sha256": source_hashes(),
        "chunk_words": chunk_words,
        "top_k": top_k,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "max_tokens": MAX_TOKENS,
            "batch_size": BATCH_SIZE,
            "device": device,
            "precision": "float32",
            "pooling": "CLS",
            "ranking": "cosine",
        },
        "runtime": runtime_metadata(device),
        "summary": summarize(rows, engines),
        "cases": rows,
    }
    if include_hybrid:
        report["datasets"] = {
            group: {"file": path.name, "sha256": hashlib.sha256(datasets[group]).hexdigest()}
            for group, path in paths.items()
        }
        report["fusion"] = {"method": "rrf", "k": RRF_K, "candidate_limit_per_branch": RRF_CANDIDATES}
        report["summary_by_question_set"] = {
            group: summarize([row for row in rows if row["question_set"] == group], engines)
            for group in paths
        }
    return report
