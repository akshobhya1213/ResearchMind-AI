from __future__ import annotations

import re
from dataclasses import dataclass

from retrieval.retriever import RetrievedChunk


@dataclass
class Source:
    id: int
    chunk_id: str
    doc_id: str
    title: str
    filename: str
    page_start: int
    page_end: int
    text: str
    score: float = 0.0

    @property
    def page_label(self) -> str:
        return f"p. {self.page_start}" if self.page_start == self.page_end else f"pp. {self.page_start}-{self.page_end}"

    @property
    def citation(self) -> str:
        return f"[{self.id}] {self.title} ({self.filename}), {self.page_label}"


def build_context(results: list[RetrievedChunk], max_chars: int = 12000) -> tuple[str, list[Source]]:
    """Number the passages [1..n] with their document/page header; stop before exceeding max_chars."""
    sources: list[Source] = []
    parts: list[str] = []
    used = 0
    for r in results:
        c = r.chunk
        src = Source(len(sources) + 1, c.chunk_id, c.doc_id, c.doc_title, c.filename, c.page_start, c.page_end, c.text, r.score)
        block = f"[{src.id}] (Document: {c.doc_title}; {src.page_label})\n{c.text}"
        if sources and used + len(block) > max_chars:
            break
        parts.append(block)
        sources.append(src)
        used += len(block)
    return "\n\n".join(parts), sources


_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def parse_citations(answer: str, n_sources: int) -> tuple[list[int], list[int]]:
    """Return (valid cited ids in order of first use, invalid ids that point to no source)."""
    valid: list[int] = []
    invalid: list[int] = []
    for group in _CITE.findall(answer):
        for tok in group.split(","):
            i = int(tok)
            target = valid if 1 <= i <= n_sources else invalid
            if i not in target:
                target.append(i)
    return valid, invalid
