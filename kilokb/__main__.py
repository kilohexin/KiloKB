"""Run the reproducible Pydantic ORM retrieval demo."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .bm25 import search_bm25
from .markdown import chunk_markdown


LOCK = Path(__file__).resolve().parents[1] / "corpora" / "pydantic.lock.json"
DEFAULT_QUERY = "Pydantic ORM 迁移时，orm_mode / from_orm 与 from_attributes / model_validate 怎么用？"


def main() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    releases = lock["releases"]
    parser = argparse.ArgumentParser(prog="python -m kilokb")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("orm-demo", help="Retrieve evidence from pinned Pydantic documents")
    demo.add_argument("--query", default=DEFAULT_QUERY)
    demo.add_argument("--version", action="append", choices=releases, help="Repeat to compare versions")
    demo.add_argument("--chunk-words", type=int, default=112)
    demo.add_argument("--top-k", type=int, default=5)
    options = parser.parse_args()
    if options.chunk_words < 1 or options.top_k < 1:
        demo.error("--chunk-words and --top-k must be positive")
    if not options.query.strip():
        demo.error("--query must not be blank")

    results = {}
    for version in dict.fromkeys(options.version or releases):
        source = releases[version]
        data = (LOCK.parent / source["snapshot"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != source["sha256"]:
            sys.exit(f"Corpus checksum mismatch: {source['snapshot']}")

        chunks = chunk_markdown(data.decode("utf-8"), options.chunk_words)
        root = f"https://github.com/{lock['repo']}/blob/{source['commit']}/{source['path']}"
        results[version] = [
            {
                "release": version,
                "commit": source["commit"],
                "path": source["path"],
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "heading_path": chunk.heading_path,
                "text": chunk.text,
                "score": score,
                "url": f"{root}#L{chunk.start_line}-L{chunk.end_line}",
            }
            for chunk, score in search_bm25(options.query, chunks, options.top_k)
        ]
    print(json.dumps(
        {"query": options.query, "chunk_words": options.chunk_words, "versions": results},
        indent=2,
    ))


if __name__ == "__main__":
    main()
