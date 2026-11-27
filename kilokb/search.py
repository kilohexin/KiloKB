"""Search a verified Pydantic snapshot with release-scoped retrieval and traces."""

import hashlib
import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from threading import Lock
from time import perf_counter
from uuid import uuid4

from .bm25 import search_bm25
from .confidence import PROFILE, SCORE, SCORE_DECIMALS, TOP_K, accepts, confidence_score
from .corpus import LOCK, Source, load_source, read_lock
from .dense import MAX_TOKENS, MODEL_ID, MODEL_REVISION, BgeM3, rank_dense
from .markdown import Chunk, chunk_markdown
from .rrf import RRF_CANDIDATES, RRF_K, fuse_rrf
from .version_scope import SearchError, resolve_scope


PROFILES = ("bm25", "dense", "hybrid_rrf")
MAX_TOP_K = 30
CHUNK_WORDS = 112
CONFIDENCE_POLICY = LOCK.parent.parent / "evals" / "pydantic-confidence-policy.json"


def calibration_contract() -> dict:
    return {
        "chunk_words": CHUNK_WORDS, "dense_model": MODEL_ID, "dense_revision": MODEL_REVISION,
        "dense_max_tokens": MAX_TOKENS, "rrf_k": RRF_K, "branch_limit": RRF_CANDIDATES,
        "rrf_pool": MAX_TOP_K, "score_decimals": SCORE_DECIMALS,
    }


@dataclass(frozen=True)
class SearchRequest:
    query: str
    mode: str = "single"
    versions: tuple[str, ...] = ()
    top_k: int = 5
    profile: str = "bm25"
    kb_id: str = "pydantic"
    abstain: bool = False


def stage_items(source: Source, hits: list[tuple[Chunk, float]]) -> list[dict]:
    return [
        {"chunk_id": source.chunk_id(chunk), "rank": rank, "score": score}
        for rank, (chunk, score) in enumerate(hits, 1)
    ]


class SearchService:
    def __init__(self, device: str = "cpu", local_files_only: bool = True) -> None:
        self.lock = read_lock()
        self.sources = {tag: load_source(self.lock, tag) for tag in self.lock["releases"]}
        self.chunks = {tag: chunk_markdown(source.text, CHUNK_WORDS) for tag, source in self.sources.items()}
        signature = json.dumps({
            "corpus": self.lock, "chunk_words": CHUNK_WORDS, "dense_revision": MODEL_REVISION,
            "chunk_ids": {tag: [self.sources[tag].chunk_id(chunk) for chunk in chunks]
                          for tag, chunks in self.chunks.items()},
        }, sort_keys=True)
        self.generation_id = hashlib.sha256(signature.encode()).hexdigest()
        self.device = device
        self.local_files_only = local_files_only
        self.model = None
        self.vectors = {}
        self.model_guard = Lock()
        self.traces: deque[dict] = deque(maxlen=128)
        self.trace_guard = Lock()

    def search(self, request: SearchRequest) -> dict:
        if request.kb_id != "pydantic":
            raise SearchError("unknown_kb", "Available knowledge base: pydantic", 404)
        if request.mode not in ("single", "compare"):
            raise SearchError("invalid_mode", "Choose single or compare")
        if request.profile not in PROFILES:
            raise SearchError("unknown_profile", f"Available profiles: {', '.join(PROFILES)}")
        if not request.query.strip():
            raise SearchError("empty_query", "Provide a query with at least one non-whitespace character")
        if not 1 <= request.top_k <= MAX_TOP_K:
            raise SearchError("invalid_top_k", f"Choose top_k from 1 to {MAX_TOP_K}")

        started = perf_counter()
        selected, origin = resolve_scope(request.query, request.mode, request.versions, self.lock)
        resolve_ms = (perf_counter() - started) * 1000
        threshold = self._threshold(request) if request.abstain else None
        candidates = {}
        embedding_ms = None
        if request.profile != "bm25":
            candidates, embedding_ms = self._dense_candidates(request.query, selected)

        groups = []
        traces = []
        for tag in selected:
            group, trace = self._group(tag, request, candidates.get(tag), threshold)
            groups.append(group)
            traces.append(trace)

        available = sum(bool(group["hits"]) for group in groups)
        status = "complete" if available == len(groups) else ("partial" if available else "insufficient")
        common = {
            "trace_id": uuid4().hex,
            "generation_id": self.generation_id,
            "resolved_versions": list(selected),
            "resolution_source": origin,
            "profile": request.profile,
            "status": status,
        }
        trace = {
            **common, "request": asdict(request), "groups": traces,
            "duration_ms": (perf_counter() - started) * 1000,
            "resolution_ms": resolve_ms, "query_embedding_ms": embedding_ms,
        }
        with self.trace_guard:
            self.traces.append(trace)
        return {**common, "groups": groups}

    def trace(self, trace_id: str) -> dict:
        with self.trace_guard:
            for entry in reversed(self.traces):
                if entry["trace_id"] == trace_id:
                    return entry
        raise SearchError("unknown_trace", "Trace not found", 404)

    def _threshold(self, request: SearchRequest) -> float:
        if request.profile != PROFILE or request.top_k != TOP_K:
            raise SearchError("uncalibrated_profile", f"Abstention is calibrated for {PROFILE} with top_k={TOP_K}")
        try:
            policy = json.loads(CONFIDENCE_POLICY.read_bytes())
            threshold = float(policy["threshold"])
            expected = {
                "schema": 1, "profile": PROFILE, "top_k": TOP_K, "score": SCORE,
                "generation_id": self.generation_id, "fitted_on": "dev",
                "contract": calibration_contract(),
            }
            if any(policy.get(key) != value for key, value in expected.items()) or not math.isfinite(threshold):
                raise ValueError("Incompatible calibration")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise SearchError("confidence_policy_unavailable", "No compatible confidence policy for this snapshot", 503) from exc
        return threshold

    def _dense_candidates(self, query: str, selected: tuple[str, ...]) -> tuple[dict, float]:
        try:
            with self.model_guard:
                if self.model is None:
                    self.model = BgeM3(self.device, self.local_files_only)
                for tag in selected:
                    if tag not in self.vectors:
                        texts = [chunk.retrieval_text for chunk in self.chunks[tag]]
                        self.vectors[tag] = self.model.encode(texts)
                started = perf_counter()
                query_vector = self.model.encode([query])[0]
                embedding_ms = (perf_counter() - started) * 1000
        except (ImportError, OSError, RuntimeError) as exc:
            raise SearchError("model_unavailable", "BGE-M3 is unavailable. Check the model cache and device.", 503) from exc

        ranked = {}
        for tag in selected:
            started = perf_counter()
            hits = rank_dense(self.chunks[tag], self.vectors[tag], query_vector, RRF_CANDIDATES)
            ranked[tag] = (hits, (perf_counter() - started) * 1000)
        return ranked, embedding_ms

    def _group(
        self, tag: str, request: SearchRequest, dense_result: tuple | None, threshold: float | None,
    ) -> tuple[dict, dict]:
        source = self.sources[tag]
        stages = {}
        timings = {}
        if request.profile != "dense":
            started = perf_counter()
            bm25 = search_bm25(request.query, self.chunks[tag], RRF_CANDIDATES)
            timings["bm25"] = (perf_counter() - started) * 1000
            stages["bm25"] = stage_items(source, bm25)
        if dense_result is not None:
            dense, timings["dense"] = dense_result
            stages["dense"] = stage_items(source, dense)

        if request.profile == "hybrid_rrf":
            started = perf_counter()
            fused = fuse_rrf(bm25, dense, MAX_TOP_K)
            timings["rrf"] = (perf_counter() - started) * 1000
            stages["rrf"] = stage_items(source, [(hit.chunk, hit.score) for hit in fused])
            ranked = [(hit.chunk, hit.score, {**hit.ranks, "rrf": index})
                      for index, hit in enumerate(fused, 1)]
            score_stage = "rrf"
        else:
            selected = bm25 if request.profile == "bm25" else dense
            ranked = [(chunk, score, {request.profile: index})
                      for index, (chunk, score) in enumerate(selected, 1)]
            score_stage = request.profile

        hits = []
        for chunk, score, ranks in ranked[:request.top_k]:
            citation = source.hit(chunk, score)
            hits.append({
                "chunk_id": source.chunk_id(chunk),
                "raw_text": citation["text"],
                "heading_path": list(citation["heading_path"]),
                "source_commit": citation["commit"],
                "path": citation["path"],
                "start_line": citation["start_line"],
                "end_line": citation["end_line"],
                "source_url": citation["url"],
                "ranks": ranks,
                "final_score": score,
                "score_stage": score_stage,
            })
        decision = None
        if threshold is not None:
            score = confidence_score(hits, stages["dense"])
            decision = {"score_stage": SCORE, "score": score, "threshold": threshold,
                        "accepted": accepts(score, threshold)}
            if not decision["accepted"]:
                hits = []
        status = "available" if hits else "insufficient"
        group = {"release": tag, "evidence_status": status, "hits": hits}
        trace = {"release": tag, "evidence_status": status, "stages": stages,
                 "stage_ms": timings, "returned": len(hits)}
        if decision is not None:
            group["confidence"] = decision
            trace["confidence"] = decision
        return group, trace
