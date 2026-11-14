INSUFFICIENT_MARKER = "INSUFFICIENT_CONTEXT"

QA_SYSTEM = (
    "You are a careful research assistant. Answer ONLY from the numbered context passages. "
    "Cite every factual statement with the passage number in square brackets, e.g. [1] or [2][3]. "
    "Never cite a number that is not in the context. Do not use outside knowledge. "
    f"If the passages do not contain enough information, reply with exactly {INSUFFICIENT_MARKER} "
    "followed by one sentence explaining what is missing."
)


def qa_prompt(question: str, context: str) -> str:
    return f"CONTEXT PASSAGES:\n{context}\n\nQUESTION: {question}\n\nANSWER (with [n] citations):"


JSON_SYSTEM = (
    "You extract structured information from numbered passages of research papers. Use ONLY the passages. "
    "If something is not stated, write 'Not stated in the retrieved passages.' Never invent numbers, datasets or results. "
    "Cite passages inline as [n]. Respond with a single JSON object and nothing else."
)
