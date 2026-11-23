"""Reciprocal rank fusion of lexical and semantic candidates for one release."""

from dataclasses import dataclass

from .markdown import Chunk


RRF_K = 60
RRF_CANDIDATES = 40


@dataclass(frozen=True)
class FusedHit:
    chunk: Chunk
    score: float
    ranks: dict[str, int | None]


def fuse_rrf(
    bm25: list[tuple[Chunk, float]], dense: list[tuple[Chunk, float]], top_k: int
) -> list[FusedHit]:
    candidates: dict[Chunk, dict[str, int]] = {}
    for branch, hits in (("bm25", bm25), ("dense", dense)):
        for rank, (chunk, _) in enumerate(hits[:RRF_CANDIDATES], 1):
            candidates.setdefault(chunk, {}).setdefault(branch, rank)

    fused = [
        FusedHit(
            chunk,
            sum(1 / (RRF_K + rank) for rank in ranks.values()),
            {"bm25": ranks.get("bm25"), "dense": ranks.get("dense")},
        )
        for chunk, ranks in candidates.items()
    ]
    return sorted(fused, key=lambda hit: (-hit.score, hit.chunk.start_line, hit.chunk.end_line))[:top_k]
