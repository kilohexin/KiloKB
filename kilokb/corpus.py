"""Load pinned source documents and serialize citation-bearing hits."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .markdown import Chunk


LOCK = Path(__file__).resolve().parents[1] / "corpora" / "pydantic.lock.json"


@dataclass(frozen=True)
class Source:
    repo: str
    release: str
    commit: str
    path: str
    text: str

    def hit(self, chunk: Chunk, score: float, *, ranks: dict[str, int | None] | None = None) -> dict:
        root = f"https://github.com/{self.repo}/blob/{self.commit}/{self.path}"
        result = {
            "release": self.release,
            "commit": self.commit,
            "path": self.path,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "heading_path": chunk.heading_path,
            "text": chunk.text,
            "score": score,
            "url": f"{root}#L{chunk.start_line}-L{chunk.end_line}",
        }
        if ranks is not None:
            result["ranks"] = ranks
        return result


def read_lock() -> dict:
    return json.loads(LOCK.read_text(encoding="utf-8"))


def load_source(lock: dict, release: str) -> Source:
    source = lock["releases"][release]
    data = (LOCK.parent / source["snapshot"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != source["sha256"]:
        raise ValueError(f"Corpus checksum mismatch: {source['snapshot']}")
    return Source(lock["repo"], release, source["commit"], source["path"], data.decode("utf-8"))
