"""Research-topic classifier: frozen transformer sentence embeddings (default: allenai-specter, a SciBERT-based
model) + multinomial logistic regression. Confidence = predicted class probability (not calibrated)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from config import CATEGORIES
from embeddings.base import Embedder

SEED_PATH = Path(__file__).parent / "data" / "seed_dataset.json"


def load_seed_dataset(path: str | Path = SEED_PATH) -> tuple[list[str], list[str]]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [r["text"] for r in rows], [r["label"] for r in rows]


@dataclass
class Prediction:
    label: str
    confidence: float
    scores: dict[str, float]


class PaperClassifier:
    def __init__(self, embedder: Embedder, C: float = 10.0):
        self.embedder, self.C, self.labels = embedder, C, list(CATEGORIES)
        self._clf: LogisticRegression | None = None

    @property
    def embedder_name(self) -> str:
        return self.embedder.name

    def fit(self, texts: list[str], labels: list[str]) -> "PaperClassifier":
        if len(set(labels)) < 2:
            raise ValueError("need at least two classes to train")
        self._clf = LogisticRegression(C=self.C, max_iter=3000)
        self._clf.fit(self.embedder.encode(texts), labels)
        return self

    def predict(self, texts: list[str]) -> list[Prediction]:
        if self._clf is None:
            raise RuntimeError("classifier is not trained")
        proba = self._clf.predict_proba(self.embedder.encode(texts))
        classes = list(self._clf.classes_)
        out = []
        for row in proba:
            scores = {c: float(p) for c, p in zip(classes, row)}
            best = max(scores, key=scores.get)
            out.append(Prediction(best, scores[best], dict(sorted(scores.items(), key=lambda kv: -kv[1]))))
        return out

    def cross_validate(self, texts: list[str], labels: list[str], folds: int = 4, seed: int = 0) -> dict:
        """Stratified k-fold cross-validated precision/recall/F1 per class (real measurements on the given data)."""
        X = self.embedder.encode(texts)
        clf = LogisticRegression(C=self.C, max_iter=3000)
        pred = cross_val_predict(clf, X, labels, cv=StratifiedKFold(folds, shuffle=True, random_state=seed))
        rep = classification_report(labels, pred, output_dict=True, zero_division=0)
        rep["_meta"] = {"embedder": self.embedder.name, "n_samples": len(texts), "folds": folds, "seed": seed}
        return rep
