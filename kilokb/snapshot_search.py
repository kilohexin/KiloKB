"""Search the active publication or pin a previously published generation."""

from collections import deque
from dataclasses import dataclass
from threading import Lock

from .corpus import Source
from .markdown import Chunk
from .search import SearchRequest, SearchService
from .snapshots import INDEX_CONTRACT, SnapshotStore
from .version_scope import SearchError


@dataclass(frozen=True)
class SearchSnapshot:
    generation_id: str
    scope: dict
    sources: dict
    chunks: dict
    vectors: dict


def materialize(store: SnapshotStore, generation_id: str) -> SearchSnapshot:
    import numpy as np

    data, cache = store.read(generation_id)
    if data["contract"] != INDEX_CONTRACT:
        raise ValueError("Snapshot encoder contract is incompatible")
    sources = {}
    chunks = {tag: [] for tag in data["scope"]["releases"]}
    vectors = {tag: [] for tag in chunks}
    for key, document in data["documents"].items():
        source = Source(**{field: value for field, value in document["source"].items() if field != "sha256"})
        sources[key] = source
        for hit in document["chunks"]:
            raw = hit["chunk"]
            chunks[source.release].append(Chunk(
                raw["start_line"], raw["end_line"], tuple(raw["heading_path"]), raw["text"], key,
            ))
            vectors[source.release].append(cache[hit["vector"]])
    return SearchSnapshot(generation_id, data["scope"], sources, chunks, {
        tag: np.asarray(items, dtype=np.float32).reshape(-1, INDEX_CONTRACT["dimension"])
        for tag, items in vectors.items()
    })


class PublishedSearch:
    def __init__(self, store: SnapshotStore, device: str = "cpu") -> None:
        self.store = store
        self.device = device
        self.cached = None
        self.guard = Lock()
        self.trace_guard = Lock()
        self.traces: deque[dict] = deque(maxlen=128)

    def engine(self, generation_id: str | None = None) -> SearchService:
        selected = generation_id or self.store.active_id()
        if selected is None:
            raise SearchError("no_active_snapshot", "Publish a snapshot before searching", 503)
        with self.guard:
            if self.cached is None or self.cached.generation_id != selected:
                try:
                    snapshot = materialize(self.store, selected)
                except KeyError as exc:
                    raise SearchError("unknown_generation", "Published generation not found", 404) from exc
                engine = SearchService(device=self.device, snapshot=snapshot)
                engine.traces = self.traces
                engine.trace_guard = self.trace_guard
                self.cached = engine
            return self.cached

    def search(self, request: SearchRequest, generation_id: str | None = None) -> dict:
        return self.engine(generation_id).search(request)

    def trace(self, trace_id: str) -> dict:
        with self.trace_guard:
            for entry in reversed(self.traces):
                if entry["trace_id"] == trace_id:
                    return entry
        raise SearchError("unknown_trace", "Trace not found", 404)
