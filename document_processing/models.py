from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class PageText:
    page_number: int  # 1-based, as printed by a PDF viewer
    text: str
    ocr: bool = False


@dataclass
class ExtractedDocument:
    filename: str
    sha256: str
    title: str
    authors: str
    pages: list[PageText]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return self.sha256[:12]

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    @property
    def n_words(self) -> int:
        return sum(len(p.text.split()) for p in self.pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    filename: str
    chunk_index: int
    page_start: int
    page_end: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)

    @property
    def page_label(self) -> str:
        return f"p. {self.page_start}" if self.page_start == self.page_end else f"pp. {self.page_start}-{self.page_end}"
