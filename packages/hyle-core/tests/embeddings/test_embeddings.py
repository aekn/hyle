from array import array
from typing import Any

import pytest

from hyle.embeddings import Embedding, Embeddings
from hyle.usage import Usage


def test_embedding_is_compact_read_only_float32_sequence() -> None:
    embedding = Embedding([1.0, 2.0, 3.0])
    assert tuple(embedding) == (1.0, 2.0, 3.0)
    assert embedding.dimensions == 3
    assert embedding.nbytes == 12
    with memoryview(embedding) as view:
        assert view.readonly
        assert view.format == "f"


def test_embedding_from_buffer_copies() -> None:
    source = array("f", [1.0, 2.0])
    embedding = Embedding.from_buffer(source)
    source[0] = 9.0
    assert embedding[0] == 1.0


def test_embeddings_is_the_batch_result() -> None:
    values = Embeddings(
        (Embedding([1.0, 2.0]), Embedding([3.0, 4.0])),
        usage=Usage(10, None),
        provider="test",
        model="embed",
    )
    assert len(values) == 2
    assert values.dimensions == 2
    assert values[1][0] == 3.0
    assert values.usage.input == 10


def test_embeddings_reject_non_embedding_values() -> None:
    invalid: Any = object()
    with pytest.raises(TypeError, match="Embedding values"):
        Embeddings((invalid,), usage=Usage())


def test_embeddings_reject_invalid_usage_metadata() -> None:
    invalid: Any = object()
    with pytest.raises(TypeError, match="usage"):
        Embeddings((Embedding([1.0]),), usage=invalid)
