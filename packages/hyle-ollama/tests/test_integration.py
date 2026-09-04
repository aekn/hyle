import os

import pytest

from hyle import Request
from hyle_ollama import OllamaClient, OllamaEmbedder, OllamaGenerationConfig, OllamaModel

pytestmark = pytest.mark.integration
_LIVE = os.getenv("HYLE_OLLAMA_INTEGRATION") == "1"


@pytest.mark.asyncio
@pytest.mark.enable_socket
@pytest.mark.skipif(not _LIVE, reason="set HYLE_OLLAMA_INTEGRATION=1")
async def test_live_generation_streaming_and_embeddings() -> None:
    host = os.getenv("HYLE_OLLAMA_HOST", "http://localhost:11434")
    model_name = os.getenv("HYLE_OLLAMA_MODEL", "qwen3:4b")
    embedding_name = os.getenv("HYLE_OLLAMA_EMBEDDING_MODEL", "embeddinggemma")

    async with OllamaClient(host) as client:
        model = OllamaModel(
            model_name,
            client=client,
            config=OllamaGenerationConfig(think=False),
        )

        response = await model.generate(Request("Reply with exactly: ok", max_output_tokens=16))
        assert response.text

        stream = model.stream(Request("Reply with exactly: ok", max_output_tokens=16))
        async with stream:
            _ = [part async for part in stream]
            assert stream.response().text

        embeddings = await OllamaEmbedder(
            embedding_name,
            client=client,
        ).embed("hello", "world")
        assert len(embeddings) == 2
        assert embeddings.dimensions > 0
