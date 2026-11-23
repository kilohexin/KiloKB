"""In-memory Okapi BM25 for a single release."""

import re
from collections import Counter
from math import log1p

from .markdown import Chunk


TERM = re.compile(r"[a-z0-9_]+(?:[.-][a-z0-9_]+)*", re.ASCII | re.IGNORECASE)
SEPARATOR = re.compile(r"[._-]+")


def tokens(text: str) -> list[str]:
    terms = []
    for match in TERM.finditer(text.lower()):
        term = match[0]
        terms.append(term)
        terms.extend(part for part in SEPARATOR.split(term) if part and part != term)
    return terms


def search_bm25(query: str, chunks: list[Chunk], top_k: int) -> list[tuple[Chunk, float]]:
    if not chunks:
        return []

    counts = [Counter(tokens(chunk.retrieval_text)) for chunk in chunks]
    lengths = [sum(count.values()) for count in counts]
    average_length = sum(lengths) / len(chunks)
    if not average_length:
        return []

    query_terms = sorted(set(tokens(query)))
    idf = {}
    for term in query_terms:
        frequency = sum(term in count for count in counts)
        idf[term] = log1p((len(chunks) - frequency + 0.5) / (frequency + 0.5))

    k1, b = 1.2, 0.75
    hits = []
    for chunk, count, length in zip(chunks, counts, lengths, strict=True):
        norm = k1 * (1 - b + b * length / average_length)
        score = sum(
            weight * count[term] * (k1 + 1) / (count[term] + norm)
            for term, weight in idf.items() if term in count
        )
        if score > 0:
            hits.append((chunk, score))
    return sorted(hits, key=lambda hit: (-hit[1], hit[0].start_line))[:top_k]
