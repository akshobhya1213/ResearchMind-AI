"""Measured evaluation. Retrieval metrics are label-based (no LLM): a retrieved chunk is relevant when it comes
from the annotated document and contains an annotated key phrase. Generation metrics need an LLM."""
from __future__ import annotations

import json
import re
import tempfile
import time
from pathlib import Path

import numpy as np

from document_processing import chunk_document
from retrieval import Retriever, RetrievalParams
from storage.vector_store import VectorStore

from .dataset import load_corpus_documents, load_qa
from .metrics import average_precision, mrr, precision_at_k, recall_at_k

CONFIGS = [("Vector only", "vector", False), ("Hybrid (vector + BM25, RRF)", "hybrid", False), ("Hybrid + cross-encoder rerank", "hybrid", True)]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _build_eval_store(embedder, chunk_size: int, overlap: int, tmp: str):
    store = VectorStore(tmp, use_faiss=False)
    chunks = [c for d in load_corpus_documents() for c in chunk_document(d, chunk_size, overlap)]
    store.add(chunks, embedder.encode([c.text for c in chunks]), embedder.name)
    return store, chunks


def run_retrieval_comparison(embedder, reranker, ks=(1, 3, 5), chunk_size: int = 60, overlap: int = 15, save_to: Path | None = None) -> dict:
    items = [i for i in load_qa()["items"] if i["answerable"]]
    kmax = max(ks)
    with tempfile.TemporaryDirectory() as tmp:
        store, chunks = _build_eval_store(embedder, chunk_size, overlap, tmp)
        retriever = Retriever(store, lambda: embedder, lambda: reranker, RetrievalParams(top_k=kmax, rerank_candidates=max(20, kmax * 3)))
        rows = []
        for label, mode, use_rerank in CONFIGS:
            agg = {f"P@{k}": [] for k in ks} | {f"R@{k}": [] for k in ks} | {"MRR": [], "Context precision": []}
            for it in items:
                res = retriever.search(it["question"], top_k=kmax, mode=mode, rerank=use_rerank and reranker is not None, apply_thresholds=False)
                phrases = [(r["doc"], _norm(r["phrase"])) for r in it["relevant"]]
                rel = [any(c.chunk.filename == d and p in _norm(c.chunk.text) for d, p in phrases) for c in res]
                rel += [False] * (kmax - len(rel))
                for k in ks:
                    covered = sum(any(c.chunk.filename == d and p in _norm(c.chunk.text) for c in res[:k]) for d, p in phrases)
                    agg[f"P@{k}"].append(precision_at_k(rel, k))
                    agg[f"R@{k}"].append(recall_at_k(covered, len(phrases)))
                agg["MRR"].append(mrr(rel))
                agg["Context precision"].append(average_precision(rel))
            rows.append({"Configuration": label if not use_rerank else (f"Hybrid + rerank [{reranker.name}]" if reranker is not None else "Hybrid + rerank (no reranker available)"),
                         **{m: round(float(np.mean(v)), 4) for m, v in agg.items()}})
    result = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset": load_qa()["name"], "n_questions": len(items), "n_chunks": len(chunks),
              "chunk_size_words": chunk_size, "chunk_overlap_words": overlap, "embedder": embedder.name,
              "embedder_is_transformer": embedder.is_transformer, "reranker": getattr(reranker, "name", None),
              "reranker_is_transformer": getattr(reranker, "is_transformer", False), "rows": rows,
              "caveat": "Tiny synthetic corpus: use for relative comparison / regression checks, not as a benchmark."}
    if save_to:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        save_to.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


FAITHFULNESS_PROMPT = ("Context:\n{context}\n\nAnswer:\n{answer}\n\nSplit the answer into atomic factual claims and judge whether each is "
                       "supported by the context. Return JSON: {{\"claims\": [{{\"claim\": str, \"supported\": bool}}]}}")


def run_generation_eval(ctx_embedder, reranker, llm, chunk_size: int = 60, overlap: int = 15, top_k: int = 4, max_questions: int | None = None,
                        save_to: Path | None = None) -> dict:
    """LLM-based evaluation. faithfulness = supported claims / claims (LLM judge, same model family as the generator, so biased);
    answer_relevancy = cosine(question, answer) embedding PROXY; answer_similarity = cosine(answer, reference)."""
    from analysis.common import json_from_llm
    from rag.pipeline import RAGPipeline

    items = load_qa()["items"][:max_questions]
    with tempfile.TemporaryDirectory() as tmp:
        store, _ = _build_eval_store(ctx_embedder, chunk_size, overlap, tmp)
        retriever = Retriever(store, lambda: ctx_embedder, lambda: reranker, RetrievalParams(top_k=top_k))
        pipe = RAGPipeline(retriever, lambda: llm)
        per_q, abstain = [], []
        for it in items:
            ans = pipe.answer(it["question"], top_k=top_k)
            if not it["answerable"]:
                abstain.append(ans.insufficient)
                continue
            ctx_text = "\n".join(s.text for s in ans.sources)
            faith = None
            if not ans.insufficient and ans.sources:
                try:
                    claims = json_from_llm(llm.generate(FAITHFULNESS_PROMPT.format(context=ctx_text, answer=ans.answer), json_mode=True))["claims"]
                    faith = sum(bool(c.get("supported")) for c in claims) / len(claims) if claims else None
                except Exception:
                    faith = None
            q, a, r = ctx_embedder.encode([it["question"], ans.answer, it["reference_answer"]])
            per_q.append({"question": it["question"], "faithfulness": faith, "answer_relevancy_proxy": float(q @ a),
                          "answer_similarity": float(a @ r), "abstained": ans.insufficient})
    mean = lambda key: (round(float(np.mean([x[key] for x in per_q if x[key] is not None])), 4) if any(x[key] is not None for x in per_q) else None)
    result = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "llm": f"{llm.name}:{llm.model}", "n_answerable": len(per_q),
              "faithfulness_llm_judge": mean("faithfulness"), "answer_relevancy_proxy": mean("answer_relevancy_proxy"),
              "answer_similarity": mean("answer_similarity"),
              "abstention_rate_on_unanswerable": (round(sum(abstain) / len(abstain), 4) if abstain else None), "per_question": per_q}
    if save_to:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        save_to.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result
