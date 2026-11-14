"""Okapi BM25 with an inverted index (own implementation; no extra dependency)."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from nlp.preprocessing import tokenize


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.n = k1, b, len(texts)
        tokens = [tokenize(t, remove_stopwords=True) for t in texts]
        self.dl = np.array([len(t) for t in tokens], dtype=np.float64)
        self.avgdl = float(self.dl.mean()) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, toks in enumerate(tokens):
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def get_scores(self, query: str) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float64)
        if not self.n:
            return scores
        for term in set(tokenize(query, remove_stopwords=True)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                denom = tf + self.k1 * (1 - self.b + self.b * self.dl[i] / (self.avgdl or 1.0))
                scores[i] += idf * tf * (self.k1 + 1) / denom
        return scores
