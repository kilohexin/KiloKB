"""Run the reproducible Pydantic ORM retrieval demo."""

import argparse
import json
import sys

from .bm25 import search_bm25
from .corpus import load_source, read_lock
from .markdown import chunk_markdown


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
    demo.add_argument("--retriever", choices=("bm25", "dense"), default="bm25")
    add_ranking_options(demo)
    add_model_options(demo)
    compare = commands.add_parser("orm-compare", help="Compare BM25 and BGE-M3 on fixed Chinese questions")
    add_ranking_options(compare)
    add_model_options(compare)
    options = parser.parse_args()
    command = demo if options.command == "orm-demo" else compare
    if options.chunk_words < 1 or options.top_k < 1:
        command.error("--chunk-words and --top-k must be positive")
    if options.command == "orm-demo" and not options.query.strip():
        demo.error("--query must not be blank")

    try:
        if options.command == "orm-compare":
            from .dense import BgeM3
            from .evaluate import compare_orm

            model = BgeM3(options.device, options.local_files_only)
            result = compare_orm(lock, model, options.chunk_words, options.top_k, options.device)
        else:
            result = run_demo(lock, options)
    except (OSError, RuntimeError, ValueError) as exc:
        sys.exit(str(exc))
    print(json.dumps(result, indent=2))


def run_demo(lock: dict, options: argparse.Namespace) -> dict:
    if options.retriever == "dense":
        from .dense import BgeM3, rank_dense

        model = BgeM3(options.device, options.local_files_only)
        query_vector = model.encode([options.query])[0]

    results = {}
    for release in dict.fromkeys(options.version or lock["releases"]):
        source = load_source(lock, release)
        chunks = chunk_markdown(source.text, options.chunk_words)
        if options.retriever == "dense":
            vectors = model.encode([chunk.retrieval_text for chunk in chunks])
            hits = rank_dense(chunks, vectors, query_vector, options.top_k)
        else:
            hits = search_bm25(options.query, chunks, options.top_k)
        results[release] = [source.hit(chunk, score) for chunk, score in hits]
    return {
        "query": options.query,
        "chunk_words": options.chunk_words,
        "versions": results,
    }


if __name__ == "__main__":
    main()
