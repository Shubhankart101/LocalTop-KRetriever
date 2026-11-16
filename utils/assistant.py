from __future__ import annotations

import os

try:
    from groq import Groq
except ImportError:
    Groq = None  # type: ignore[assignment]


class GroqAnswerClient:
    """Real answer client used outside local deterministic evaluation."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY is required to call Groq.")
        if Groq is None:
            raise ImportError("Install groq before running the application.")
        self.client = Groq(api_key=key)

    def complete(self, question: str, context: str) -> str:
        response = self.client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            messages=[
                {"role": "system", "content": "Answer only from the supplied manual context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
            ],
        )
        return response.choices[0].message.content or ""


def answer_with_retrieval(retriever, question: str, client=None) -> str:
    documents = retriever.invoke(question)
    context = "\n\n".join(document.page_content for document in documents)
    return (client or GroqAnswerClient()).complete(question, context)
