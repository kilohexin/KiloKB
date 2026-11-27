"""Replay identical candidate pools to compare ranking quality and timing."""

from math import ceil
from statistics import median
from time import perf_counter

from .dense import BATCH_SIZE, MAX_TOKENS, MODEL_ID, MODEL_REVISION
from .evaluate import evidence_ranks, runtime_metadata, source_hashes, summarize
from .frozen import FrozenCase
from .rerank import CROSS_MODEL, CROSS_REVISION, CrossEncoder, MaxSim, maxsim_score, reorder


ENGINES = ("rrf", "cross_encoder", "maxsim")


def timings(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "samples": len(values),
        "p50_ms": median(ordered),
        "p95_ms": ordered[ceil(len(ordered) * 0.95) - 1],
    }


def compare_rerankers(
    manifest: dict, cases: list[FrozenCase], device: str, local_files_only: bool,
    top_k: int, warmup: int, rounds: int,
) -> dict:
    import torch

    started = perf_counter()
    cross = CrossEncoder(device, local_files_only)
    cross_load_ms = (perf_counter() - started) * 1000
    started = perf_counter()
    maxsim = MaxSim(device, local_files_only)
    maxsim_load_ms = (perf_counter() - started) * 1000

    windows = {
        candidate.chunk_id: candidate.chunk.retrieval_text
        for case in cases for candidate in case.candidates
    }
    text_groups = [[candidate.chunk.retrieval_text for candidate in case.candidates] for case in cases]
    cross_lengths = [
        length for case, texts in zip(cases, text_groups, strict=True)
        for length in cross.lengths(case.query, texts)
    ]
    maxsim_doc_lengths = maxsim.lengths(list(windows.values()))
    maxsim_query_lengths = maxsim.lengths([case.query for case in cases])
    started = perf_counter()
    doc_vectors = dict(zip(windows, maxsim.encode(list(windows.values())), strict=True))
    if device == "cuda":
        torch.cuda.synchronize()
    precompute_ms = (perf_counter() - started) * 1000
    vector_bytes = sum(vector.numel() * vector.element_size() for vector in doc_vectors.values())

    def rrf_rank(index: int):
        return [(candidate, candidate.rrf_score) for candidate in cases[index].candidates[:top_k]]

    def cross_rank(index: int):
        case = cases[index]
        return reorder(case.candidates, cross.score(case.query, text_groups[index]), top_k)

    def maxsim_rank(index: int):
        case = cases[index]
        query_vector = maxsim.encode([case.query])[0]
        scores = [
            maxsim_score(query_vector, doc_vectors[candidate.chunk_id])
            for candidate in case.candidates
        ]
        return reorder(case.candidates, scores, top_k)

    rows = [
        {
            "qid": case.qid,
            "query": case.query,
            "version": case.source.release,
            "question_set": case.group,
            "candidate_count": len(case.candidates),
            "input_evidence_ranks": evidence_ranks(
                case.lines, [(candidate.chunk, candidate.rrf_score) for candidate in case.candidates]
            ),
        }
        for case in cases
    ]
    latency = {}
    for name, rank in zip(ENGINES, (rrf_rank, cross_rank, maxsim_rank), strict=True):
        for ordinal in range(warmup):
            rank(ordinal % len(cases))
        if device == "cuda":
            resident_bytes = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(rounds):
            for index, (row, case) in enumerate(zip(rows, cases, strict=True)):
                started = perf_counter()
                hits = rank(index)
                elapsed = (perf_counter() - started) * 1000
                samples.append(elapsed)
                if name not in row:
                    row[name] = {
                        "evidence_ranks": evidence_ranks(
                            case.lines, [(candidate.chunk, score) for candidate, score in hits]
                        ),
                        "hits": [
                            {
                                "chunk_id": candidate.chunk_id,
                                "input_rank": candidate.input_rank,
                                "start_line": candidate.chunk.start_line,
                                "end_line": candidate.chunk.end_line,
                                "score": score,
                                "url": case.source.hit(candidate.chunk, score)["url"],
                            }
                            for candidate, score in hits
                        ],
                        "latency_ms": [],
                    }
                row[name]["latency_ms"].append(elapsed)
        latency[name] = timings(samples)
        if device == "cuda":
            latency[name]["peak_additional_cuda_bytes"] = torch.cuda.max_memory_allocated() - resident_bytes

    return {
        "freeze": manifest,
        "source_sha256": source_hashes(),
        "runtime": runtime_metadata(device),
        "settings": {
            "device": device, "precision": "float32", "batch_size": BATCH_SIZE,
            "max_tokens": MAX_TOKENS, "top_k": top_k, "warmup_per_engine": warmup,
            "rounds": rounds, "concurrency": 1, "percentile": "median / nearest-rank P95",
            "latency_scope": "warm frozen-candidate replay. Includes query work, scoring and ranking.",
        },
        "models": {
            "cross_encoder": {"id": CROSS_MODEL, "revision": CROSS_REVISION, "load_ms": cross_load_ms},
            "maxsim": {
                "id": MODEL_ID, "revision": MODEL_REVISION, "load_ms": maxsim_load_ms,
                "document_precompute_ms": precompute_ms, "document_count": len(doc_vectors),
                "raw_vector_bytes": vector_bytes,
            },
        },
        "truncation": {
            name: {"over_limit": sum(length > MAX_TOKENS for length in lengths), "inputs": len(lengths)}
            for name, lengths in (
                ("cross_pairs", cross_lengths), ("maxsim_queries", maxsim_query_lengths),
                ("maxsim_unique_documents", maxsim_doc_lengths),
            )
        },
        "latency": latency,
        "summary": summarize(rows, ENGINES),
        "summary_by_question_set": {
            group: summarize([row for row in rows if row["question_set"] == group], ENGINES)
            for group in ("zh", "api")
        },
        "cases": rows,
    }
