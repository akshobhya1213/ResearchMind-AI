from __future__ import annotations

import hashlib
import json
from pathlib import Path

from document_processing.models import ExtractedDocument, PageText

EVAL_DIR = Path(__file__).parent / "eval_data"


def load_corpus_documents(path: Path | None = None) -> list[ExtractedDocument]:
    data = json.loads((path or EVAL_DIR / "corpus.json").read_text(encoding="utf-8"))
    return [ExtractedDocument(d["filename"], hashlib.sha256(d["filename"].encode()).hexdigest(), d["title"], "",
                              [PageText(i, t) for i, t in enumerate(d["pages"], start=1)]) for d in data["documents"]]


def load_qa(path: Path | None = None) -> dict:
    return json.loads((path or EVAL_DIR / "qa.json").read_text(encoding="utf-8"))
