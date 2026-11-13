"""OPTIONAL RAGAS integration (pip install -r requirements-ragas.txt). RAGAS' API changes between releases and this
wrapper has NOT been executed in the development sandbox; failures are returned as an error message, never hidden."""
from __future__ import annotations


def run_ragas(samples: list[dict], gemini_api_key: str, model: str = "gemini-2.5-flash") -> dict:
    """samples: [{question, answer, contexts: [str], ground_truth}]"""
    try:
        from datasets import Dataset
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=model, google_api_key=gemini_api_key))
        emb = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=gemini_api_key))
        data = Dataset.from_list([{"question": s["question"], "answer": s["answer"], "contexts": s["contexts"], "ground_truth": s["ground_truth"],
                                   "reference": s["ground_truth"]} for s in samples])
        res = evaluate(data, metrics=[faithfulness, answer_relevancy, context_precision, context_recall], llm=llm, embeddings=emb)
        return {"ok": True, "scores": {k: float(v) for k, v in res._repr_dict.items()} if hasattr(res, "_repr_dict") else dict(res)}
    except Exception as exc:
        return {"ok": False, "error": f"RAGAS run failed (unverified integration): {type(exc).__name__}: {exc}"}
