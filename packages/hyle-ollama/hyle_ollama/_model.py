__all__ = ("OllamaEmbedder", "OllamaModel")

from dataclasses import dataclass

from hyle.content import Content
from hyle.embeddings import Embeddable
from hyle.errors import UnsupportedFeatureError
from hyle.generation import Request
from hyle_ollama._client import OllamaClient, is_direct_cloud, send_chat, send_embed
from hyle_ollama._config import OllamaEmbeddingConfig, OllamaGenerationConfig
from hyle_ollama._errors import OllamaError
from hyle_ollama._response import OllamaEmbeddings, OllamaResponse
from hyle_ollama._stream import OllamaStream
from hyle_ollama._translate import (
    decode_chat_response,
    decode_embed_response,
    encode_chat_request,
    encode_embed_request,
)
from hyle_ollama._validate import boolean, positive_int, text
from hyle_ollama._wire import ChatRequestWire

_DEFAULT_GENERATION_CONFIG = OllamaGenerationConfig()
_DEFAULT_EMBEDDING_CONFIG = OllamaEmbeddingConfig()


@dataclass(frozen=True, slots=True, init=False)
class OllamaModel:
    model: str
    client: OllamaClient
    config: OllamaGenerationConfig

    def __init__(
        self,
        model: str,
        /,
        *,
        client: OllamaClient,
        config: OllamaGenerationConfig = _DEFAULT_GENERATION_CONFIG,
    ) -> None:
        config = _generation_config(config)
        object.__setattr__(self, "model", text(model, "model", blank=False).strip())
        object.__setattr__(self, "client", client)
        object.__setattr__(self, "config", config)

    async def generate(self, request: Request, /) -> OllamaResponse:
        response = await send_chat(self.client, self._encode(request, stream=False))
        return decode_chat_response(response)

    def stream(self, request: Request, /) -> OllamaStream:
        return OllamaStream(self.client, self._encode(request, stream=True))

    def _encode(self, request: Request, /, *, stream: bool) -> ChatRequestWire:
        if is_direct_cloud(self.client) and (request.schema is not None or self.config.json_mode):
            raise UnsupportedFeatureError("Ollama Cloud does not support structured output")
        return encode_chat_request(self.model, request, self.config, stream=stream)


@dataclass(frozen=True, slots=True, init=False)
class OllamaEmbedder:
    model: str
    client: OllamaClient
    config: OllamaEmbeddingConfig

    def __init__(
        self,
        model: str,
        /,
        *,
        client: OllamaClient,
        config: OllamaEmbeddingConfig = _DEFAULT_EMBEDDING_CONFIG,
    ) -> None:
        config = _embedding_config(config)
        object.__setattr__(self, "model", text(model, "model", blank=False).strip())
        object.__setattr__(self, "client", client)
        object.__setattr__(self, "config", config)

    async def embed(
        self,
        first: Embeddable,
        /,
        *inputs: Embeddable,
        dimensions: int | None = None,
        truncate: bool = False,
    ) -> OllamaEmbeddings:
        if dimensions is not None:
            dimensions = positive_int(dimensions, "dimensions")
        boolean(truncate, "truncate")

        normalized = tuple(_embedding_text(value) for value in (first, *inputs))
        response = await send_embed(
            self.client,
            encode_embed_request(
                self.model,
                normalized,
                dimensions,
                truncate,
                self.config,
            ),
        )
        result = decode_embed_response(response)
        if len(result) != len(normalized):
            raise OllamaError("embedding count mismatch", transient=False)
        return result


def _embedding_text(value: Embeddable, /) -> str:
    if isinstance(value, str):
        return text(value, "embedding input", blank=True)
    if isinstance(value, Content):
        raise UnsupportedFeatureError("Ollama embeddings support text only")

    parts = tuple(value)
    if not parts:
        raise ValueError("embedding input must not be empty")

    chunks: list[str] = []
    for part in parts:
        if not isinstance(part, str):
            raise UnsupportedFeatureError("Ollama embeddings support text only")
        chunks.append(text(part, "embedding input", blank=True))
    return "".join(chunks)


def _generation_config(value: object, /) -> OllamaGenerationConfig:
    if not isinstance(value, OllamaGenerationConfig):
        raise TypeError("config must be OllamaGenerationConfig")
    return value


def _embedding_config(value: object, /) -> OllamaEmbeddingConfig:
    if not isinstance(value, OllamaEmbeddingConfig):
        raise TypeError("config must be OllamaEmbeddingConfig")
    return value
