import json

import httpx
import pytest

from hyle.content import Binary
from hyle.errors import UnsupportedFeatureError
from hyle_ollama import (
    OllamaClient,
    OllamaEmbedder,
    OllamaEmbeddingConfig,
    OllamaError,
    OllamaOptions,
)


@pytest.mark.asyncio
async def test_embedding_request_preserves_portable_truncate_and_dimensions() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body == {
            "model": "embeddinggemma",
            "input": ["first", "second"],
            "truncate": False,
            "keep_alive": "5m",
            "dimensions": 2,
            "options": {"num_ctx": 4096},
        }
        return httpx.Response(
            200,
            json={
                "model": "embeddinggemma",
                "embeddings": [[1.0, 2.0], [3.0, 4.0]],
                "prompt_eval_count": 5,
                "total_duration": 100,
            },
        )

    config = OllamaEmbeddingConfig(
        keep_alive="5m",
        options=OllamaOptions(num_ctx=4096),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        embedder = OllamaEmbedder("embeddinggemma", client=OllamaClient(http=http), config=config)
        result = await embedder.embed("first", "second", dimensions=2, truncate=False)

    assert len(result) == 2
    assert result.dimensions == 2
    assert result.usage.input == 5
    assert result.timings.total_ns == 100


@pytest.mark.asyncio
async def test_embedding_logical_parts_are_joined_once() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["input"] == ["hello world"]
        return httpx.Response(200, json={"model": "embed", "embeddings": [[1.0, 2.0]]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        embedder = OllamaEmbedder("embed", client=OllamaClient(http=http))
        result = await embedder.embed(("hello ", "world"))
    assert result.dimensions == 2


@pytest.mark.asyncio
async def test_embedding_rejects_non_text_content_before_transport() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        embedder = OllamaEmbedder("embed", client=OllamaClient(http=http))
        with pytest.raises(UnsupportedFeatureError, match="text only"):
            await embedder.embed(Binary(b"image", media_type="image/png"))
    assert calls == 0


@pytest.mark.asyncio
async def test_embedding_cardinality_is_validated() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "embed", "embeddings": [[1.0, 2.0]]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        embedder = OllamaEmbedder("embed", client=OllamaClient(http=http))
        with pytest.raises(OllamaError, match="embedding count mismatch"):
            await embedder.embed("first", "second")


@pytest.mark.asyncio
async def test_malformed_embedding_dimensions_are_provider_errors() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "embeddinggemma",
                "embeddings": [[1.0, 2.0], [3.0]],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        embedder = OllamaEmbedder("embeddinggemma", client=OllamaClient(http=http))
        with pytest.raises(OllamaError, match="invalid embedding response"):
            await embedder.embed("first", "second")


@pytest.mark.asyncio
async def test_empty_multipart_embedding_input_is_rejected_before_transport() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        embedder = OllamaEmbedder("embed", client=OllamaClient(http=http))
        with pytest.raises(ValueError, match="embedding input must not be empty"):
            await embedder.embed(())

    assert calls == 0
