"""Build a Local Top-K Retriever.

A maintenance assistant searches a supplied equipment manual. Complete only:
1. validate_chunk_config
2. make_chunks
3. build_retriever

The Document type, deterministic embedder, local vector store, factory, manual,
and Groq-backed answer layer are supplied. Evaluation is local and makes no
provider or network call.

Required behavior:
- chunk_size and k are positive non-boolean integers.
- overlap is a non-boolean integer from 0 through chunk_size - 1.
- Invalid values raise InvalidChunkConfigError.
- Empty text returns an empty list.
- Split text with a sliding character window of chunk_size and advance by
  chunk_size - overlap. Preserve exact characters.
- Every Document includes metadata {"source": source, "chunk_index": index}.
- Chunk indexes start at zero and follow source order.
- build_retriever passes the chunks and exact injected embedder to the supplied
  factory, then returns store.as_retriever(search_kwargs={"k": k}).

Not required: downloading models, vector mathematics, hosted databases, model
calls during evaluation, a user interface, or deployment.
"""

from __future__ import annotations

from collections.abc import Sequence

from utils.retrieval import Document


class InvalidChunkConfigError(ValueError):
    pass


def validate_chunk_config(chunk_size: int, overlap: int, k: int) -> None:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise InvalidChunkConfigError("chunk_size must be a positive non-boolean integer.")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise InvalidChunkConfigError("k must be a positive non-boolean integer.")
    if isinstance(overlap, bool) or not isinstance(overlap, int):
        raise InvalidChunkConfigError("overlap must be a non-boolean integer.")
    if overlap < 0 or overlap >= chunk_size:
        raise InvalidChunkConfigError("overlap must be at least 0 and strictly smaller than chunk_size.")


def make_chunks(
    text: str,
    source: str,
    chunk_size: int,
    overlap: int,
) -> list[Document]:
    validate_chunk_config(chunk_size, overlap, 1)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [Document(page_content=text, metadata={"source": source, "chunk_index": 0})]

    step = chunk_size - overlap
    chunks = []
    for index, start in enumerate(range(0, len(text), step)):
        chunks.append(
            Document(
                page_content=text[start : start + chunk_size],
                metadata={"source": source, "chunk_index": index},
            )
        )
    return chunks


def build_retriever(
    chunks: Sequence[Document],
    embedder,
    vector_store_factory,
    k: int,
):
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise InvalidChunkConfigError("k must be a positive non-boolean integer.")
    store = vector_store_factory(chunks, embedder)
    return store.as_retriever(search_kwargs={"k": k})
