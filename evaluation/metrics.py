"""Rank-based retrieval metrics. `relevance` is a list of booleans in rank order (is the chunk relevant?)."""
from __future__ import annotations


def precision_at_k(relevance: list[bool], k: int) -> float:
    return sum(relevance[:k]) / k if k > 0 else 0.0


def recall_at_k(covered: int, total_relevant: int) -> float:
    return covered / total_relevant if total_relevant else 0.0


def mrr(relevance: list[bool]) -> float:
    for i, r in enumerate(relevance, start=1):
        if r:
            return 1.0 / i
    return 0.0


def average_precision(relevance: list[bool]) -> float:
    """RAGAS-style 'context precision': mean of precision@i over the ranks i of relevant chunks."""
    hits, total = 0, 0.0
    for i, r in enumerate(relevance, start=1):
        if r:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0
