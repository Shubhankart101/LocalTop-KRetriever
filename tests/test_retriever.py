"""Unit and integration tests for LocalTop-KRetriever."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from main import (
    InvalidChunkConfigError,
    build_retriever,
    make_chunks,
    validate_chunk_config,
)
from utils.assistant import answer_with_retrieval
from utils.retrieval import (
    DeterministicEmbedder,
    Document,
    local_vector_store_factory,
)


class TestValidateChunkConfig:
    """Test suite for validate_chunk_config validation rules and edge cases."""

    def test_valid_configurations(self):
        # Should not raise for valid positive integer combinations
        validate_chunk_config(4, 1, 1)
        validate_chunk_config(1, 0, 1)
        validate_chunk_config(100, 99, 10)
        validate_chunk_config(10, 0, 5)

    def test_invalid_chunk_size_types_and_ranges(self):
        # Booleans must be rejected
        with pytest.raises(InvalidChunkConfigError):
            validate_chunk_config(True, 0, 1)
        with pytest.raises(InvalidChunkConfigError):
            validate_chunk_config(False, 0, 1)

        # Zero or negative values
        for bad_size in [0, -1, -50]:
            with pytest.raises(InvalidChunkConfigError):
                validate_chunk_config(bad_size, 0, 1)

        # Non-integer types
        for bad_size in [1.5, "10", None, [], {}, (10,)]:
            with pytest.raises(InvalidChunkConfigError):
                validate_chunk_config(bad_size, 0, 1)

    def test_invalid_k_types_and_ranges(self):
        # Booleans must be rejected
        with pytest.raises(InvalidChunkConfigError):
            validate_chunk_config(10, 2, True)
        with pytest.raises(InvalidChunkConfigError):
            validate_chunk_config(10, 2, False)

        # Zero or negative values
        for bad_k in [0, -1, -20]:
            with pytest.raises(InvalidChunkConfigError):
                validate_chunk_config(10, 2, bad_k)

        # Non-integer types
        for bad_k in [2.5, "3", None, [], {}]:
            with pytest.raises(InvalidChunkConfigError):
                validate_chunk_config(10, 2, bad_k)

    def test_invalid_overlap_types_and_ranges(self):
        # Booleans must be rejected
        with pytest.raises(InvalidChunkConfigError):
            validate_chunk_config(10, True, 1)
        with pytest.raises(InvalidChunkConfigError):
            validate_chunk_config(10, False, 1)

        # Non-integer types
        for bad_overlap in [0.5, "0", None, [], {}]:
            with pytest.raises(InvalidChunkConfigError):
                validate_chunk_config(10, bad_overlap, 1)

        # Negative overlap
        for bad_overlap in [-1, -10]:
            with pytest.raises(InvalidChunkConfigError):
                validate_chunk_config(10, bad_overlap, 1)

        # Overlap equal to or exceeding chunk_size
        with pytest.raises(InvalidChunkConfigError):
            validate_chunk_config(4, 4, 1)
        with pytest.raises(InvalidChunkConfigError):
            validate_chunk_config(4, 5, 1)
        with pytest.raises(InvalidChunkConfigError):
            validate_chunk_config(1, 1, 1)

    def test_error_inheritance(self):
        assert issubclass(InvalidChunkConfigError, ValueError)


class TestMakeChunks:
    """Test suite for make_chunks deterministic overlapping chunk generator."""

    def test_invalid_config_raises(self):
        with pytest.raises(InvalidChunkConfigError):
            make_chunks("Sample text", "manual.txt", -1, 0)
        with pytest.raises(InvalidChunkConfigError):
            make_chunks("Sample text", "manual.txt", 4, 4)
        with pytest.raises(InvalidChunkConfigError):
            make_chunks("Sample text", "manual.txt", True, 0)
        with pytest.raises(InvalidChunkConfigError):
            make_chunks("Sample text", "manual.txt", 4, True)
        with pytest.raises(InvalidChunkConfigError):
            make_chunks("", "manual.txt", 0, 0)

    def test_empty_text(self):
        assert make_chunks("", "manual.txt", 4, 1) == []
        assert make_chunks("", "manual.txt", 10, 0) == []

    def test_text_shorter_than_chunk_size(self):
        chunks = make_chunks("ABC", "manual.txt", 4, 1)
        assert len(chunks) == 1
        assert chunks[0].page_content == "ABC"
        assert chunks[0].metadata == {"source": "manual.txt", "chunk_index": 0}

        chunks = make_chunks("A", "manual.txt", 4, 1)
        assert len(chunks) == 1
        assert chunks[0].page_content == "A"
        assert chunks[0].metadata == {"source": "manual.txt", "chunk_index": 0}

        # Even with high overlap, shorter text produces exactly one chunk
        chunks = make_chunks("ABC", "manual.txt", 4, 3)
        assert len(chunks) == 1
        assert chunks[0].page_content == "ABC"
        assert chunks[0].metadata == {"source": "manual.txt", "chunk_index": 0}

    def test_text_equal_to_chunk_size(self):
        chunks = make_chunks("ABCD", "manual.txt", 4, 1)
        assert len(chunks) == 1
        assert chunks[0].page_content == "ABCD"
        assert chunks[0].metadata == {"source": "manual.txt", "chunk_index": 0}

        chunks = make_chunks("ABCD", "manual.txt", 4, 0)
        assert len(chunks) == 1
        assert chunks[0].page_content == "ABCD"
        assert chunks[0].metadata == {"source": "manual.txt", "chunk_index": 0}

    def test_prompt_chunking_example(self):
        # text = "ABCDEFGHIJ", chunk_size = 4, overlap = 1 -> step = 3
        # Chunks: "ABCD", "DEFG", "GHIJ" (terminates once text is fully covered)
        text = "ABCDEFGHIJ"
        chunks = make_chunks(text, "equipment_manual.txt", 4, 1)
        expected = ["ABCD", "DEFG", "GHIJ"]
        assert [c.page_content for c in chunks] == expected
        for i, chunk in enumerate(chunks):
            assert chunk.metadata == {
                "source": "equipment_manual.txt",
                "chunk_index": i,
            }

    def test_zero_overlap(self):
        text = "ABCDEFGHIJ"
        chunks = make_chunks(text, "source.txt", 4, 0)
        expected = ["ABCD", "EFGH", "IJ"]
        assert [c.page_content for c in chunks] == expected
        for i, chunk in enumerate(chunks):
            assert chunk.metadata == {"source": "source.txt", "chunk_index": i}

    def test_exact_character_preservation(self):
        text = "  Raw formatted \n\ttext with spaces.  "
        chunks = make_chunks(text, "raw.txt", 10, 3)
        # Check first chunk matches verbatim
        assert chunks[0].page_content == text[:10]
        # Check no characters stripped
        assert chunks[0].page_content == "  Raw form"
        assert all(c.metadata["source"] == "raw.txt" for c in chunks)


class TestBuildRetriever:
    """Test suite for build_retriever and integration with VectorStore & AnswerClient."""

    def test_build_retriever_forwards_dependencies(self):
        mock_factory = MagicMock()
        mock_store = MagicMock()
        mock_factory.return_value = mock_store
        mock_retriever = MagicMock()
        mock_store.as_retriever.return_value = mock_retriever

        chunks = [
            Document("Chunk 1 content", {"source": "src", "chunk_index": 0}),
            Document("Chunk 2 content", {"source": "src", "chunk_index": 1}),
        ]
        embedder = object()

        retriever = build_retriever(chunks, embedder, mock_factory, k=3)

        mock_factory.assert_called_once_with(chunks, embedder)
        mock_store.as_retriever.assert_called_once_with(search_kwargs={"k": 3})
        assert retriever is mock_retriever

    def test_build_retriever_invalid_k(self):
        chunks = [Document("Content", {"source": "src", "chunk_index": 0})]
        embedder = object()
        factory = MagicMock()

        for invalid_k in [0, -1, True, False, 1.5, "2", None]:
            with pytest.raises(InvalidChunkConfigError):
                build_retriever(chunks, embedder, factory, k=invalid_k)
        assert factory.call_count == 0

    def test_end_to_end_retrieval_and_answering(self):
        manual_text = (
            "EQUIPMENT MANUAL - INDUSTRIAL COMPRESSOR C-500\n"
            "Section 1: Daily maintenance requires checking oil viscosity and pressure.\n"
            "Section 2: High temperature alarm triggers when manifold exceeds 95C.\n"
            "Section 3: Emergency shutoff valve is located on the lower left intake.\n"
        )
        chunks = make_chunks(manual_text, "compressor_manual.txt", chunk_size=60, overlap=15)
        embedder = DeterministicEmbedder()
        retriever = build_retriever(chunks, embedder, local_vector_store_factory, k=2)

        # Retrieve documents for query
        results = retriever.invoke("temperature alarm")
        assert len(results) <= 2
        assert any("temperature" in doc.page_content.lower() for doc in results)

        # Test answering with injected mock Groq client
        mock_client = MagicMock()
        mock_client.complete.return_value = "The high temperature alarm triggers above 95C."

        response = answer_with_retrieval(
            retriever,
            "When does high temperature alarm trigger?",
            client=mock_client,
        )
        assert response == "The high temperature alarm triggers above 95C."
        mock_client.complete.assert_called_once()
        question, context = mock_client.complete.call_args[0]
        assert question == "When does high temperature alarm trigger?"
        assert "temperature" in context.lower()
