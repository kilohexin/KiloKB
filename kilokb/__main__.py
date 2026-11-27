"""Run the reproducible Pydantic ORM retrieval demo."""

import argparse
import json
import sys
from pathlib import Path

from .bm25 import search_bm25
from .corpus import load_source, read_lock
from .markdown import chunk_markdown
from .rrf import RRF_CANDIDATES, fuse_rrf


DEFAULT_QUERY = "Pydantic ORM 迁移时，orm_mode / from_orm 与 from_attributes / model_validate 怎么用？"


def add_ranking_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--chunk-words", type=int, default=112)
    parser.add_argument("--top-k", type=int, default=5)


def add_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--local-files-only", action="store_true", help="Use cached model files only")


def main() -> None:
    lock = read_lock()
    releases = lock["releases"]
    parser = argparse.ArgumentParser(prog="python -m kilokb")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("orm-demo", help="Retrieve evidence from pinned Pydantic documents")
    demo.add_argument("--query", default=DEFAULT_QUERY)
    demo.add_argument("--version", action="append", choices=releases, help="Repeat to compare versions")
    demo.add_argument("--retriever", choices=("bm25", "dense", "hybrid"), default="bm25")
    add_ranking_options(demo)
    add_model_options(demo)
    from .search import PROFILES

    search = commands.add_parser("search", help="Search one or compare two Pydantic releases")
    search.add_argument("--query", required=True)
    search.add_argument("--mode", choices=("single", "compare"), default="single")
    search.add_argument("--versions", nargs="+", default=())
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--profile", choices=PROFILES, default="bm25")
    search.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    search.add_argument("--trace", action="store_true", help="Include the retrieval trace in JSON")
    search.add_argument("--abstain", action="store_true", help="Use the calibrated evidence policy")
    search.add_argument("--snapshot-db", type=Path, help="Read an active or pinned published snapshot")
    search.add_argument("--kb-id", default="pydantic")
    search.add_argument("--generation-id", help="Read a specific published generation")
    compare = commands.add_parser("orm-compare", help="Compare BM25 and BGE-M3 on fixed Chinese questions")
    compare.add_argument("--hybrid", action="store_true", help="Include RRF and questions with API names")
    add_ranking_options(compare)
    add_model_options(compare)
    freeze = commands.add_parser("orm-rerank-freeze", help="Freeze RRF Top 30 per question and version")
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--chunk-words", type=int, default=112)
    add_model_options(freeze)
    rerank = commands.add_parser("orm-rerank", help="Replay frozen candidates with both BGE rerankers")
    rerank.add_argument("--candidates", type=Path, required=True)
    rerank.add_argument("--top-k", type=int, default=5)
    rerank.add_argument("--rounds", type=int, default=5)
    rerank.add_argument("--warmup", type=int, default=10)
    add_model_options(rerank)
    confidence = commands.add_parser("confidence-eval", help="Calibrate on dev and evaluate on held-out families")
    confidence.add_argument("--policy-out", type=Path)
    add_model_options(confidence)
    index = commands.add_parser("index", help="Enqueue and publish durable snapshot jobs")
    index.add_argument("--db", type=Path, required=True)
    operations = index.add_subparsers(dest="operation", required=True)
    enqueue = operations.add_parser("enqueue", help="Capture a complete target manifest")
    enqueue.add_argument("--manifest", type=Path, required=True)
    run = operations.add_parser("run", help="Build, validate and publish a job, or retry a failure")
    run.add_argument("--job", type=int, required=True)
    add_model_options(run)
    inspect = operations.add_parser("status", help="Inspect a job and current active generation")
    inspect.add_argument("--job", type=int, required=True)
    options = parser.parse_args()
    command = {"orm-demo": demo, "orm-compare": compare, "search": search,
               "orm-rerank-freeze": freeze, "orm-rerank": rerank,
               "confidence-eval": confidence, "index": index}[options.command]
    for field in ("chunk_words", "top_k", "rounds"):
        if getattr(options, field, 1) < 1:
            command.error(f"--{field.replace('_', '-')} must be positive")
    if getattr(options, "warmup", 0) < 0:
        command.error("--warmup must be non-negative")
    if options.command == "orm-demo" and not options.query.strip():
        demo.error("--query must not be blank")

    try:
        if options.command == "search":
            from .search import SearchRequest, SearchService
            from .version_scope import SearchError

            if options.snapshot_db is None:
                service = SearchService(device=options.device)
                generation = None
            else:
                from .snapshot_search import PublishedSearch
                from .snapshots import SnapshotStore

                service = PublishedSearch(SnapshotStore(options.snapshot_db), device=options.device)
                generation = options.generation_id
            try:
                request = SearchRequest(
                    query=options.query, mode=options.mode, versions=tuple(options.versions),
                    top_k=options.top_k, profile=options.profile, abstain=options.abstain,
                    kb_id=options.kb_id,
                )
                if options.snapshot_db is not None:
                    result = service.search(request, generation)
                else:
                    if options.generation_id is not None and options.generation_id != service.generation_id:
                        raise SearchError("unknown_generation", "Published generation not found", 404)
                    result = service.search(request)
            except SearchError as exc:
                sys.exit(json.dumps(exc.detail))
            if options.trace:
                result["trace"] = service.trace(result["trace_id"])
        elif options.command == "orm-compare":
            from .dense import BgeM3
            from .evaluate import compare_orm

            model = BgeM3(options.device, options.local_files_only)
            result = compare_orm(lock, model, options.chunk_words, options.top_k,
                                 options.device, include_hybrid=options.hybrid)
        elif options.command == "orm-rerank-freeze":
            from .dense import BgeM3
            from .frozen import freeze_orm

            model = BgeM3(options.device, options.local_files_only)
            result = freeze_orm(lock, model, options.output, options.chunk_words)
        elif options.command == "orm-rerank":
            from .frozen import load_frozen
            from .rerank_experiment import compare_rerankers

            manifest, cases = load_frozen(lock, options.candidates)
            result = compare_rerankers(
                manifest, cases, options.device, options.local_files_only,
                options.top_k, options.warmup, options.rounds,
            )
        elif options.command == "confidence-eval":
            from .confidence_eval import DATASET, run_evaluation

            result = run_evaluation(DATASET, options.device, options.local_files_only)
            if options.policy_out:
                options.policy_out.write_bytes((json.dumps(result["policy"], indent=2) + "\n").encode())
        elif options.command == "index":
            from .snapshots import SnapshotStore

            store = SnapshotStore(options.db)
            if options.operation == "enqueue":
                job = store.enqueue(options.manifest)
            else:
                job = options.job
                if options.operation == "run":
                    from .dense import BgeM3

                    model = None

                    def encode(texts):
                        nonlocal model
                        if model is None:
                            model = BgeM3(options.device, options.local_files_only)
                        return model.encode(texts)

                    store.run(job, encode)
            result = {**store.status(job), "active_generation": store.active_id()}
        else:
            result = run_demo(lock, options)
    except (OSError, RuntimeError, ValueError) as exc:
        sys.exit(str(exc))
    print(json.dumps(result, indent=2))


def run_demo(lock: dict, options: argparse.Namespace) -> dict:
    if options.retriever != "bm25":
        from .dense import BgeM3, rank_dense

        model = BgeM3(options.device, options.local_files_only)
        query_vector = model.encode([options.query])[0]

    results = {}
    for release in dict.fromkeys(options.version or lock["releases"]):
        source = load_source(lock, release)
        chunks = chunk_markdown(source.text, options.chunk_words)
        if options.retriever == "bm25":
            hits = search_bm25(options.query, chunks, options.top_k)
            results[release] = [source.hit(chunk, score) for chunk, score in hits]
        else:
            vectors = model.encode([chunk.retrieval_text for chunk in chunks])
            if options.retriever == "dense":
                hits = rank_dense(chunks, vectors, query_vector, options.top_k)
                results[release] = [source.hit(chunk, score) for chunk, score in hits]
            else:
                bm25 = search_bm25(options.query, chunks, RRF_CANDIDATES)
                dense = rank_dense(chunks, vectors, query_vector, RRF_CANDIDATES)
                results[release] = [
                    source.hit(hit.chunk, hit.score, ranks=hit.ranks)
                    for hit in fuse_rrf(bm25, dense, options.top_k)
                ]
    return {
        "query": options.query,
        "chunk_words": options.chunk_words,
        "versions": results,
    }


if __name__ == "__main__":
    main()
