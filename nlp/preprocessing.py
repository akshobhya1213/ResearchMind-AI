"""Text preprocessing: tokenisation and sentence splitting (NLTK punkt when available, regex otherwise)."""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = frozenset(ENGLISH_STOP_WORDS)
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
_SENT_FALLBACK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])")
_nltk_ok: bool | None = None


def tokenize(text: str, lowercase: bool = True, remove_stopwords: bool = False, min_len: int = 2) -> list[str]:
    tokens = _TOKEN.findall(text.lower() if lowercase else text)
    if remove_stopwords:
        tokens = [t for t in tokens if t.lower() not in STOPWORDS]
    return [t for t in tokens if len(t) >= min_len]


def sentence_split(text: str) -> list[str]:
    global _nltk_ok
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if _nltk_ok is not False:
        try:
            from nltk.tokenize import sent_tokenize

            sents = sent_tokenize(text)
            _nltk_ok = True
            return [s.strip() for s in sents if s.strip()]
        except Exception:  # ImportError, or LookupError when punkt data is not downloaded
            _nltk_ok = False
    return [s.strip() for s in _SENT_FALLBACK.split(text) if s.strip()]
