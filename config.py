"""Central configuration. Every value can be overridden through environment variables
(or a local .env file). Secrets are only ever read from the environment."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:  # optional convenience for local development
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("DATA_DIR", "./data")))
    max_pdf_mb: int = field(default_factory=lambda: _int("MAX_PDF_MB", 50))
    embedding_model: str = field(default_factory=lambda: os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    classifier_model: str = field(default_factory=lambda: os.environ.get("CLASSIFIER_MODEL", "allenai-specter"))
    reranker_model: str = field(default_factory=lambda: os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    allow_fallback_models: bool = field(default_factory=lambda: _bool("ALLOW_FALLBACK_MODELS", True))
    device: str | None = field(default_factory=lambda: os.environ.get("DEVICE") or None)
    llm_provider: str = field(default_factory=lambda: os.environ.get("LLM_PROVIDER", "gemini"))
    llm_model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", "gemini-2.5-flash"))
    gemini_api_key: str | None = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    llm_temperature: float = field(default_factory=lambda: _float("LLM_TEMPERATURE", 0.2))
    chunk_size: int = field(default_factory=lambda: _int("CHUNK_SIZE", 220))
    chunk_overlap: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 40))
    top_k: int = field(default_factory=lambda: _int("TOP_K", 5))
    rerank_candidates: int = field(default_factory=lambda: _int("RERANK_CANDIDATES", 20))
    rrf_k: int = 60
    vector_weight: float = 1.0
    bm25_weight: float = 1.0
    min_vector_score: float = field(default_factory=lambda: _float("MIN_VECTOR_SCORE", 0.30))
    min_rerank_score: float = field(default_factory=lambda: _float("MIN_RERANK_SCORE", 0.05))
    max_context_chars: int = field(default_factory=lambda: _int("MAX_CONTEXT_CHARS", 12000))
    ocr_fallback: bool = field(default_factory=lambda: _bool("OCR_FALLBACK", True))
    classify_on_upload: bool = field(default_factory=lambda: _bool("CLASSIFY_ON_UPLOAD", True))
    preload_models: bool = field(default_factory=lambda: _bool("PRELOAD_MODELS", False))
    log_level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "evaluation"

    def public_dict(self) -> dict:
        d = asdict(self)
        d["data_dir"] = str(self.data_dir)
        d["gemini_api_key"] = "set" if self.gemini_api_key else "not set"
        return d


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


PROJECT_ROOT = Path(__file__).resolve().parent
CATEGORIES = ["Machine Learning", "Deep Learning", "NLP", "Computer Vision",
              "Generative AI", "LLMs", "Information Retrieval", "Cybersecurity"]
