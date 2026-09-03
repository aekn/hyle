__all__ = ("OllamaStream",)

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from types import TracebackType
from typing import Literal, Self

import httpx
import msgspec
from msgspec import UNSET

from hyle.generation import Part, PartBuffer
from hyle_ollama._client import OllamaClient, iter_ndjson, open_chat_stream
from hyle_ollama._errors import OllamaError
from hyle_ollama._response import OllamaLogprob, OllamaResponse
from hyle_ollama._translate import build_response, decode_logprobs, decode_stream_parts
from hyle_ollama._wire import ChatRecordWire, ChatRequestWire

type _State = Literal["new", "open", "iterating", "done", "failed", "closed"]

_DECODER = msgspec.json.Decoder(ChatRecordWire)


class OllamaStream:
    __slots__ = (
        "_client",
        "_final",
        "_iterator",
        "_request",
        "_response",
        "_state",
    )

    def __init__(self, client: OllamaClient, request: ChatRequestWire, /) -> None:
        self._client = client
        self._request = request
        self._response: httpx.Response | None = None
        self._iterator: AsyncGenerator[Part] | None = None
        self._final: OllamaResponse | None = None
        self._state: _State = "new"

    async def __aenter__(self) -> Self:
        if self._state != "new":
            raise RuntimeError("stream already entered")
        try:
            self._response = await open_chat_stream(self._client, self._request)
        except BaseException:
            self._state = "failed"
            raise
        self._state = "open"
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        iterator = self._iterator
        response = self._response
        try:
            if iterator is not None:
                await iterator.aclose()
        finally:
            self._iterator = None
            self._response = None
            if self._state not in {"done", "failed"}:
                self._state = "closed"
            if response is not None:
                await response.aclose()

    def __aiter__(self) -> AsyncIterator[Part]:
        if self._state == "new" or self._state == "closed":
            raise RuntimeError("stream is not open")
        if self._state != "open":
            raise RuntimeError("stream already iterated")
        self._state = "iterating"
        iterator = self._iterate()
        self._iterator = iterator
        return iterator

    def response(self) -> OllamaResponse:
        if self._state != "done" or self._final is None:
            raise RuntimeError("stream is not complete")
        return self._final

    async def _iterate(self) -> AsyncGenerator[Part]:
        response = self._response
        if response is None:
            self._state = "failed"
            raise RuntimeError("stream is not open")

        parts = PartBuffer()
        logprobs: list[OllamaLogprob] = []
        terminal: ChatRecordWire | None = None
        remote_model: str | None = None
        remote_host: str | None = None

        try:
            async with aclosing(iter_ndjson(self._client, response)) as records:
                async for data in records:
                    if terminal is not None:
                        raise OllamaError("data after terminal record", transient=False)

                    try:
                        record = _DECODER.decode(data)
                    except msgspec.DecodeError as error:
                        raise OllamaError("invalid chat record", transient=False) from error

                    if record.error is not UNSET and record.error:
                        raise OllamaError(record.error, transient=False)

                    if record.message is not UNSET:
                        for part, key in decode_stream_parts(record.message):
                            parts.append(part, key=key)
                            yield part

                    logprobs.extend(decode_logprobs(record.logprobs))
                    if record.remote_model is not UNSET and record.remote_model:
                        remote_model = record.remote_model
                    if record.remote_host is not UNSET and record.remote_host:
                        remote_host = record.remote_host
                    if record.done is True:
                        terminal = record

            if terminal is None:
                raise OllamaError("stream ended before terminal record", transient=False)

            final = build_response(
                terminal,
                parts.finish(),
                tuple(logprobs),
                remote_model=remote_model,
                remote_host=remote_host,
            )
        except GeneratorExit:
            self._state = "closed"
            raise
        except BaseException:
            self._state = "failed"
            raise
        else:
            self._final = final
            self._state = "done"
