from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Document:
    page_content: str
    metadata: dict[str, Any]


class DeterministicEmbedder:
    """Small local token-set embedder used instead of a downloaded model."""

    @staticmethod
    def _tokens(text: str) -> frozenset[str]:
        return frozenset(re.findall(r"[a-z0-9]+", text.lower()))

    def embed_documents(self, texts: list[str]) -> list[frozenset[str]]:
        return [self._tokens(text) for text in texts]

    def embed_query(self, text: str) -> frozenset[str]:
        return self._tokens(text)


class LocalRetriever:
    def __init__(self, documents, vectors, embedder, k: int) -> None:
        self.documents = documents
        self.vectors = vectors
        self.embedder = embedder
        self.k = k

    def invoke(self, query: str) -> list[Document]:
        query_vector = self.embedder.embed_query(query)
        ranked = []
        for position, (document, vector) in enumerate(zip(self.documents, self.vectors)):
            score = len(query_vector & vector)
            if score:
                ranked.append((-score, position, document))
        ranked.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in ranked[: self.k]]


class LocalVectorStore:
    def __init__(self, documents: list[Document], embedder) -> None:
        self.documents = list(documents)
        self.embedder = embedder
        self.vectors = embedder.embed_documents(
            [document.page_content for document in self.documents]
        )

    def as_retriever(self, search_kwargs: dict[str, int]) -> LocalRetriever:
        return LocalRetriever(
            self.documents,
            self.vectors,
            self.embedder,
            search_kwargs["k"],
        )


def local_vector_store_factory(documents: list[Document], embedder) -> LocalVectorStore:
    return LocalVectorStore(documents, embedder)
