---
title: ResearchMind AI
emoji: 🔬
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# ResearchMind AI — Intelligent Research Paper Analysis and Discovery Platform

Upload academic PDFs, then search, question, summarise, compare and classify them with a retrieval-augmented (RAG) pipeline.
Python only: PyTorch / Sentence-Transformers, scikit-learn, spaCy, PyMuPDF, FAISS, Gradio, Gemini (configurable provider).

> **Status (read this first).** The code base is complete and the parts that can run offline are tested (see
> [docs/TEST_REPORT.md](docs/TEST_REPORT.md)). It was built in a sandbox **without internet, PyTorch, PyMuPDF, FAISS, Gradio or Docker**, so the
> transformer models, real-PDF extraction, FAISS backend, Gradio UI, Gemini calls and the Docker image have **never been executed**.
> Nothing here is deployed. Expect to fix small integration issues on first run; run the checklist in `docs/TEST_REPORT.md`.

## Features
| Area | Implementation |
|---|---|
| PDF ingestion | multi-upload, validation (extension, size, `%PDF` header, encryption, empty), per-page text, metadata, title heuristic, OCR fallback (PyMuPDF + Tesseract), duplicate detection by file hash **and** normalised text |
| Chunking | page-aware word windows; every chunk keeps `page_start/page_end`, file and title; configurable size/overlap |
| Retrieval | dense (Sentence-Transformers, default `BAAI/bge-small-en-v1.5`), BM25 (own implementation), hybrid via Reciprocal Rank Fusion, cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`), document filter, cosine/rerank thresholds |
| RAG | numbered context, grounded prompt, `[n]` citations validated against retrieved sources, document + page citations taken from chunk metadata (not from the LLM), insufficient-context detection (thresholds + model abstention), retrieval-only mode without an LLM |
| LLM | provider abstraction (`rag/llm.py`), Gemini first, key only from environment, retry/backoff |
| Analysis | structured summaries (9 fields), multi-paper comparison with rule-based comparability warnings, gap analysis separating *author-stated evidence* / *potential gaps (hypotheses)* / *verified literature gaps (always empty: not verifiable here)* |
| NLP | TF-IDF keywords, lexicon + spaCy entities, reported-metric regex, tokenisation, NLTK/regex sentence splitting |
| Classification | 8 categories; frozen `allenai-specter` (SciBERT-based) embeddings + logistic regression; probabilities shown |
| Evaluation | Precision@K, Recall@K, MRR, context precision/recall (label based), LLM-judge faithfulness, embedding-proxy answer relevancy, optional RAGAS wrapper; compares vector vs hybrid vs hybrid+rerank |
| UI | Gradio, 10 tabs: Home, Library, Upload, Summarization, Ask, Comparison, Research Gaps, NLP & Classification, RAG Evaluation, Settings |

## Architecture
```mermaid
flowchart LR
  PDF[PDF upload] --> V[validate + extract (PyMuPDF/OCR)] --> C[page-aware chunking]
  C --> E[embeddings] --> S[(vector store: FAISS/NumPy + JSON on disk)]
  C --> B[BM25 index]
  Q[question] --> QE[query embedding] --> S
  Q --> B
  S --> F[RRF fusion]
  B --> F --> R[cross-encoder rerank] --> T[threshold / sufficiency check]
  T --> P[numbered context + prompt] --> L[LLM provider] --> A[answer + validated citations]
```
Modules: `document_processing/ nlp/ embeddings/ storage/ retrieval/ rag/ deep_learning/ analysis/ evaluation/ ui/ tests/ scripts/`.
`rag/services.py::AppContext` wires everything; `ui/handlers.py` holds UI logic (unit-tested without Gradio); `ui/app_ui.py` is layout only.

### Design decisions and deviations (be aware)
* **No LlamaIndex/LangChain.** The brief asked for one primary framework. I implemented the pipeline directly (FAISS + sentence-transformers + own BM25/RRF) because the framework integration could not be installed or verified offline, and it keeps the dependency set small. RAGAS (which needs LangChain) is an optional extra.
* **Fallback models.** If a transformer cannot be loaded and `ALLOW_FALLBACK_MODELS=true`, a *lexical hashing embedder* / lexical reranker is used so the app still runs. This is **not semantic**; the dashboard shows a warning. Set `ALLOW_FALLBACK_MODELS=false` in production to fail loudly instead.
* **Thresholds are heuristics** (`MIN_VECTOR_SCORE=0.30`, `MIN_RERANK_SCORE=0.05`), not calibrated. Tune them on your papers.
* **Classifier confidence** is a logistic-regression probability on frozen embeddings; the categories overlap (a paper can be both NLP and LLMs).

## Install and run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm            # optional but recommended (NER)
cp .env.example .env                                # then put GEMINI_API_KEY in .env (only external credential needed)
python app.py                                       # http://localhost:7860
```
Without `GEMINI_API_KEY` the app runs in retrieval-only mode: search, retrieval, NLP, classification, retrieval evaluation and extractive summaries work; generated answers/comparisons/gap hypotheses do not. OCR needs the `tesseract-ocr` system package.

Docker: `docker compose up --build` (see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for Hugging Face Spaces).

## Configuration
All in environment variables, see `.env.example` and `config.py`. Notable: `EMBEDDING_MODEL`, `CLASSIFIER_MODEL`, `RERANKER_MODEL`, `LLM_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `DATA_DIR`, `ALLOW_FALLBACK_MODELS`, `PRELOAD_MODELS`. Changing the embedding model triggers automatic re-embedding of the stored index on next use.

## Datasets
* **Classification seed set** (`deep_learning/data/seed_dataset.json`): 64 short abstract-style sentences (8 per category) **written by the project author**. It is tiny and not a benchmark. For a real dataset run `python scripts/build_arxiv_dataset.py out.json 200` (public arXiv API; category→label mapping is an approximation; not run here), then `python scripts/evaluate_classifier.py out.json`.
* **Evaluation corpus** (`evaluation/eval_data/`): 3 **synthetic, fictional** 3-page papers (JSON + rendered PDFs) and 12 answerable + 2 unanswerable questions with key-phrase relevance labels. Contents and numbers are invented; they exist to test the pipeline.

## Measured results (real runs, but see the caveats)
Produced in the development sandbox with the **lexical fallback models** (no transformers available). 12 answerable questions, 60-word chunks:

| Configuration | P@1 | P@3 | P@5 | R@1 | R@3 | R@5 | MRR | Context precision |
|---|---|---|---|---|---|---|---|---|
| Vector only | 0.667 | 0.333 | 0.217 | 0.611 | 0.861 | 0.944 | 0.785 | 0.771 |
| Hybrid (RRF) | 0.667 | 0.333 | 0.217 | 0.611 | 0.861 | 0.944 | 0.781 | 0.767 |
| Hybrid + lexical-fallback rerank | 0.667 | 0.333 | 0.217 | 0.611 | 0.861 | 0.944 | 0.781 | 0.767 |

The three rows being almost identical is expected here: the fallback "vector" model is itself lexical, the corpus has ~25 chunks and the reranker is a fallback. **These numbers say nothing about transformer or cross-encoder quality.** Re-run on your machine with `python scripts/run_evaluation.py` or the *RAG Evaluation* tab; results are written to `evaluation/results/` / `DATA_DIR/evaluation/`.

Classifier (4-fold stratified CV, 64 seed samples, lexical-hash features): accuracy 0.406, macro-F1 0.403 (chance = 0.125). The transformer-based classifier has **not** been measured. Faithfulness / answer relevancy (LLM-based) have **not** been measured (no API access).

## Tests
`pytest` (needs `requirements-dev.txt`). See [docs/TEST_REPORT.md](docs/TEST_REPORT.md) for what ran and what did not.

## Limitations
* Retrieval-based summaries/comparisons only see the top passages per field, so long papers can be summarised incompletely; the LLM is told to say "Not stated" rather than guess.
* Tables, figures and equations in PDFs are extracted as plain text (often garbled); two-column layouts may interleave.
* Entity extraction uses a finite lexicon; unknown datasets/models are not detected. Metric regexes are heuristic.
* LLM-judge faithfulness shares biases with the generator; embedding-based answer relevancy is a proxy, not RAGAS' definition.
* "Verified literature gaps" are intentionally never claimed.
* Free Hugging Face Spaces storage is ephemeral (see deployment guide).

## Future improvements
Framework adapters (LlamaIndex), table/figure-aware parsing, calibrated confidence, multi-label classification on arXiv data, per-field full-document map-reduce summarisation, background ingestion queue, auth for multi-user deployments.

## Screenshots
*Placeholders — no screenshots exist yet because the UI has not been run:* `docs/screenshots/home.png`, `ask.png`, `comparison.png`.

## License
MIT
