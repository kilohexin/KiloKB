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


DATASET = LOCK.parent.parent / "evals" / "pydantic-orm-zh.json"


def evidence_ranks(lines: list[int], hits: list[tuple[Chunk, float]]) -> dict[int, int | None]:
    return {
        line: next((
            rank for rank, (chunk, _) in enumerate(hits, 1)
            if chunk.start_line <= line <= chunk.end_line
        ), None)
        for line in lines
    }


def measure(source: Source, lines: list[int], hits: list[tuple[Chunk, float]]) -> dict:
    return {
        "returned": len(hits),
        "evidence_ranks": evidence_ranks(lines, hits),
        "hits": [source.hit(chunk, score) for chunk, score in hits],
    }


def compare_orm(lock: dict, model: BgeM3, chunk_words: int, top_k: int, device: str) -> dict:
    raw_dataset = DATASET.read_bytes()
    dataset = json.loads(raw_dataset)
    releases = dict.fromkeys(tag for case in dataset for tag in case["versions"])
    sources = {tag: load_source(lock, tag) for tag in releases}
    chunks = {tag: chunk_markdown(source.text, chunk_words) for tag, source in sources.items()}
    vectors = {tag: model.encode([chunk.retrieval_text for chunk in chunks[tag]]) for tag in releases}
    queries = model.encode([case["query"] for case in dataset])

    rows = []
    for case, query_vector in zip(dataset, queries, strict=True):
        for tag, lines in case["versions"].items():
            bm25 = search_bm25(case["query"], chunks[tag], top_k)
            dense = rank_dense(chunks[tag], vectors[tag], query_vector, top_k)
            rows.append({
                "qid": case["qid"],
                "query": case["query"],
                "version": tag,
                "source_commit": sources[tag].commit,
                "source_path": sources[tag].path,
                "bm25": measure(sources[tag], lines, bm25),
                "dense": measure(sources[tag], lines, dense),
            })

    import torch

    source_files = sorted(Path(__file__).parent.glob("*.py"))
    return {
        "dataset": DATASET.name,
        "dataset_sha256": hashlib.sha256(raw_dataset).hexdigest(),
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
        "summary": {
            name: {
                "all_evidence_cases": sum(
                    all(rank is not None for rank in row[name]["evidence_ranks"].values()) for row in rows
                ),
                "total_cases": len(rows),
                "anchors_found": sum(
                    rank is not None for row in rows for rank in row[name]["evidence_ranks"].values()
                ),
                "total_anchors": sum(len(row[name]["evidence_ranks"]) for row in rows),
            }
            for name in ("bm25", "dense")
        },
        "cases": rows,
    }
