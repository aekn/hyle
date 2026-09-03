from asyncio import CancelledError, Event, create_task
from collections.abc import AsyncIterator

import httpx
import pytest

from hyle.generation import Reasoning, Request
from hyle.tools import ToolCall
from hyle_ollama import OllamaClient, OllamaError, OllamaModel


class Chunks(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes) -> None:
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class BlockingChunks(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.started = Event()
        self.cancelled = Event()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.started.set()
        try:
            await Event().wait()
        except CancelledError:
            self.cancelled.set()
            raise
        yield b""

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_stream_is_incremental_and_reconstructs_terminal_response() -> None:
    body = Chunks(
        b'{"model":"qwen3","created_at":"2026-09-02T12:00:00Z",'
        b'"message":{"role":"assistant","thinking":"think ","content":"hel"},'
        b'"done":false}\n',
        b'{"model":"qwen3","created_at":"2026-09-02T12:00:00Z",'
        b'"message":{"role":"assistant","thinking":"more","content":"lo",'
        b'"tool_calls":[{"id":"1","function":{"index":0,"name":"lookup",'
        b'"arguments":{"q":"x"}}}]},"done":false}\n',
        b'{"model":"qwen3","created_at":"2026-09-02T12:00:00Z",'
        b'"message":{"role":"assistant","content":""},"done":true,'
        b'"done_reason":"stop","prompt_eval_count":2,"eval_count":3}\n',
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        stream = OllamaModel("qwen3", client=OllamaClient(http=http)).stream(Request("hi"))
        async with stream:
            parts = [part async for part in stream]
            response = stream.response()

    assert parts == [
        Reasoning("think "),
        "hel",
        Reasoning("more"),
        "lo",
        ToolCall("lookup", '{"q":"x"}', call_id="1"),
    ]
    assert response.parts[0] == Reasoning("think more")
    assert response.require_text() == "hello"
    assert response.tool_calls[0].call_id == "1"
    assert body.closed


@pytest.mark.asyncio
async def test_stream_state_errors_are_short_and_deterministic() -> None:
    client = OllamaClient()
    stream = OllamaModel("qwen3", client=client).stream(Request("hi"))
    with pytest.raises(RuntimeError, match="stream is not open"):
        stream.__aiter__()
    with pytest.raises(RuntimeError, match="stream is not complete"):
        stream.response()
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_requires_terminal_record() -> None:
    body = Chunks(
        b'{"model":"qwen3","created_at":"2026-09-02T12:00:00Z",'
        b'"message":{"role":"assistant","content":"x"},"done":false}\n'
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        stream = OllamaModel("qwen3", client=OllamaClient(http=http)).stream(Request("hi"))
        async with stream:
            with pytest.raises(OllamaError, match="stream ended before terminal record"):
                _ = [part async for part in stream]


@pytest.mark.asyncio
async def test_stream_rejects_data_after_terminal_record() -> None:
    body = Chunks(
        b'{"model":"qwen3","created_at":"2026-09-02T12:00:00Z",'
        b'"message":{"role":"assistant","content":"x"},"done":true}\n',
        b'{"model":"qwen3","created_at":"2026-09-02T12:00:00Z",'
        b'"message":{"role":"assistant","content":"y"},"done":false}\n',
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        stream = OllamaModel("qwen3", client=OllamaClient(http=http)).stream(Request("hi"))
        async with stream:
            with pytest.raises(OllamaError, match="data after terminal record"):
                _ = [part async for part in stream]


@pytest.mark.asyncio
async def test_stream_record_limit_closes_response() -> None:
    body = Chunks(b"{" + b"x" * 32)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = OllamaClient(http=http, max_record_bytes=16)
        stream = OllamaModel("qwen3", client=client).stream(Request("hi"))
        async with stream:
            with pytest.raises(OllamaError, match="stream record too large"):
                _ = [part async for part in stream]
    assert body.closed


@pytest.mark.asyncio
async def test_stream_accepts_crlf_and_final_record_without_newline() -> None:
    body = Chunks(
        b'{"model":"qwen3","created_at":"2026-09-02T12:00:00Z",'
        b'"message":{"role":"assistant","content":"a"},"done":false}\r\n',
        b'{"model":"qwen3","created_at":"2026-09-02T12:00:00Z",'
        b'"message":{"role":"assistant","content":"b"},"done":true,"done_reason":"stop"}',
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        stream = OllamaModel("qwen3", client=OllamaClient(http=http)).stream(Request("hi"))
        async with stream:
            assert [part async for part in stream] == ["a", "b"]
            assert stream.response().require_text() == "ab"

    assert body.closed


@pytest.mark.asyncio
async def test_stream_total_limit_closes_response() -> None:
    body = Chunks(b'{"message":{"role":"assistant","content":"' + b"x" * 64)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = OllamaClient(http=http, max_stream_bytes=32, max_record_bytes=128)
        stream = OllamaModel("qwen3", client=client).stream(Request("hi"))
        async with stream:
            with pytest.raises(OllamaError, match="stream too large"):
                _ = [part async for part in stream]

    assert body.closed


@pytest.mark.asyncio
async def test_early_stream_exit_closes_response() -> None:
    body = Chunks(
        b'{"model":"qwen3","created_at":"2026-09-02T12:00:00Z",'
        b'"message":{"role":"assistant","content":"first"},"done":false}\n',
        b'{"model":"qwen3","created_at":"2026-09-02T12:00:00Z",'
        b'"message":{"role":"assistant","content":"second"},"done":true}\n',
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        stream = OllamaModel("qwen3", client=OllamaClient(http=http)).stream(Request("hi"))
        async with stream:
            async for part in stream:
                assert part == "first"
                break

    assert body.closed


@pytest.mark.asyncio
async def test_stream_rejects_invalid_json_record() -> None:
    body = Chunks(b"not-json\n")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        stream = OllamaModel("qwen3", client=OllamaClient(http=http)).stream(Request("hi"))
        async with stream:
            with pytest.raises(OllamaError, match="invalid chat record"):
                _ = [part async for part in stream]


@pytest.mark.asyncio
async def test_stream_cancellation_closes_response_and_propagates() -> None:
    body = BlockingChunks()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        stream = OllamaModel("qwen3", client=OllamaClient(http=http)).stream(Request("hi"))

        async def consume() -> None:
            async with stream:
                _ = [part async for part in stream]

        task = create_task(consume())
        await body.started.wait()
        task.cancel()
        with pytest.raises(CancelledError):
            await task

    assert body.cancelled.is_set()
    assert body.closed


@pytest.mark.asyncio
async def test_stream_preserves_tool_calls_in_arrival_order_independent_of_index() -> None:
    body = Chunks(
        b'{"model":"qwen3","created_at":"2026-09-02T12:00:00Z",'
        b'"message":{"role":"assistant","tool_calls":[{"id":"a","function":'
        b'{"index":0,"name":"first","arguments":{"value":1}}}]},"done":false}\n',
        b'{"model":"qwen3","created_at":"2026-09-02T12:00:00Z",'
        b'"message":{"role":"assistant","tool_calls":[{"id":"b","function":'
        b'{"index":0,"name":"second","arguments":{"value":2}}}]},"done":false}\n',
        b'{"model":"qwen3","created_at":"2026-09-02T12:00:00Z",'
        b'"message":{"role":"assistant"},"done":true,"done_reason":"stop"}\n',
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        stream = OllamaModel("qwen3", client=OllamaClient(http=http)).stream(Request("hi"))
        async with stream:
            parts = [part async for part in stream]
            response = stream.response()

    expected = [
        ToolCall("first", '{"value":1}', call_id="a"),
        ToolCall("second", '{"value":2}', call_id="b"),
    ]
    assert parts == expected
    assert list(response.tool_calls) == expected
