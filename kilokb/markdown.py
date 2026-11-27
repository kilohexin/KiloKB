"""Line-aligned Markdown chunks for source citations."""

import re
from dataclasses import dataclass


HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*$")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


@dataclass(frozen=True)
class Chunk:
    start_line: int
    end_line: int
    heading_path: tuple[str, ...]
    text: str
    document_key: str | None = None

    @property
    def retrieval_text(self) -> str:
        return " ".join((*self.heading_path, self.text))


def chunk_markdown(text: str, max_words: int) -> list[Chunk]:
    if max_words < 1:
        raise ValueError("max_words must be positive")

    lines = text.splitlines()
    headings: list[tuple[int, str]] = []
    chunks: list[Chunk] = []
    fence: str | None = None
    start = 0
    for index, line in enumerate(lines):
        marker = FENCE.match(line)
        if marker:
            run, tail = marker.groups()
            if fence is None:
                fence = run
            elif run.startswith(fence) and not tail.strip():
                fence = None
            continue
        if fence is not None:
            continue

        heading = HEADING.match(line)
        if heading:
            path = tuple(title for _, title in headings)
            chunks.extend(_chunk_section(lines, start, index, path, max_words))
            level = len(heading[1])
            headings = [(depth, title) for depth, title in headings if depth < level]
            headings.append((level, heading[2]))
            start = index

    path = tuple(title for _, title in headings)
    chunks.extend(_chunk_section(lines, start, len(lines), path, max_words))
    return chunks


def _chunk_section(
    lines: list[str], start: int, end: int, path: tuple[str, ...], max_words: int
) -> list[Chunk]:
    chunks: list[Chunk] = []
    counts = [len(line.split()) for line in lines[start:end]]
    left = right = words = 0
    while left < len(counts):
        # Keep an oversized line intact so citations always recover the original text.
        while right < len(counts) and (right == left or words + counts[right] <= max_words):
            words += counts[right]
            right += 1
        if words > 0:
            chunks.append(Chunk(start + left + 1, start + right, path,
                                "\n".join(lines[start + left:start + right])))
        if right == len(counts):
            break
        # Carry a short suffix of complete lines to reduce boundary losses.
        next_left = right
        overlap = 0
        while next_left > left + 1 and overlap + counts[next_left - 1] <= max_words // 4:
            next_left -= 1
            overlap += counts[next_left]
        left = next_left
        right = left
        words = 0
    return chunks
