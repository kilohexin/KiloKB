"""Compare lexical and semantic retrieval on fixed Chinese ORM questions."""

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

from .bm25 import search_bm25
from .corpus import LOCK, Source, load_source
from .dense import BATCH_SIZE, MAX_TOKENS, MODEL_ID, MODEL_REVISION, BgeM3, rank_dense
from .markdown import Chunk, chunk_markdown
from .rrf import RRF_CANDIDATES, RRF_K, fuse_rrf


DATASET = LOCK.parent.parent / "evals" / "pydantic-orm-zh.json"
API_DATASET = DATASET.with_name("pydantic-orm-api.json")


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
    paths = {"zh": DATASET}
    if include_hybrid:
        paths["api"] = API_DATASET
    datasets = {group: path.read_bytes() for group, path in paths.items()}
    cases = [(group, case) for group, raw in datasets.items() for case in json.loads(raw)]
    releases = dict.fromkeys(tag for _, case in cases for tag in case["versions"])
    sources = {tag: load_source(lock, tag) for tag in releases}
    chunks = {tag: chunk_markdown(source.text, chunk_words) for tag, source in sources.items()}
    vectors = {tag: model.encode([chunk.retrieval_text for chunk in chunks[tag]]) for tag in releases}
    queries = model.encode([case["query"] for _, case in cases])
    candidate_limit = max(RRF_CANDIDATES, top_k) if include_hybrid else top_k

    rows = []
    for (group, case), query_vector in zip(cases, queries, strict=True):
        for tag, lines in case["versions"].items():
            bm25 = search_bm25(case["query"], chunks[tag], candidate_limit)
            dense = rank_dense(chunks[tag], vectors[tag], query_vector, candidate_limit)
            row = {
                "qid": case["qid"],
                "query": case["query"],
                "version": tag,
                "source_commit": sources[tag].commit,
                "source_path": sources[tag].path,
                "bm25": measure(sources[tag], lines, bm25[:top_k]),
                "dense": measure(sources[tag], lines, dense[:top_k]),
            }
            if include_hybrid:
                fused = fuse_rrf(bm25, dense, top_k)
                row["question_set"] = group
                row["hybrid"] = measure(
                    sources[tag], lines,
                    [(hit.chunk, hit.score) for hit in fused], ranks=[hit.ranks for hit in fused],
                )
            rows.append(row)

    import torch

    source_files = sorted(Path(__file__).parent.glob("*.py"))
    engines = ("bm25", "dense", "hybrid") if include_hybrid else ("bm25", "dense")
    report = {
        "dataset": DATASET.name,
        "dataset_sha256": hashlib.sha256(datasets["zh"]).hexdigest(),
        "corpus_sha256": {tag: lock["releases"][tag]["sha256"] for tag in releases},
        "source_sha256": {
            path.name: hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
            for path in source_files
        },
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
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
            "torch_threads": torch.get_num_threads(),
            "dependencies": {name: version(name) for name in ("torch", "transformers", "sentence-transformers")},
        },
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
