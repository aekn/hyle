import asyncio
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any, Self

import pytest

from hyle import Request, Response, run, tool
from hyle.generation import Part
from hyle.runtime import RunEvent, RunLimitError, RunLimits, RunTimeoutError, run_stream
from hyle.tools import ToolCall, ToolFinished, ToolStarted
from hyle.usage import Usage

_DEFAULT_USAGE = Usage(1, 1)


class Model:
    def __init__(self, responses: list[Response]) -> None:
        self.responses = iter(responses)
        self.requests: list[Request] = []

    async def generate(self, request: Request, /) -> Response:
        self.requests.append(request)
        return next(self.responses)


class Stream:
    def __init__(self, response: Response) -> None:
        self._response = response
        self._done = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    async def _iterate(self) -> AsyncIterator[Part]:
        for part in self._response.parts:
            yield part
        self._done = True

    def __aiter__(self) -> AsyncIterator[Part]:
        return self._iterate()

    def response(self) -> Response:
        if not self._done:
            raise RuntimeError("not done")
        return self._response


class Streaming:
    def __init__(self, responses: list[Response]) -> None:
        self.responses = iter(responses)
        self.requests: list[Request] = []

    async def generate(self, request: Request, /) -> Response:
        self.requests.append(request)
        return next(self.responses)

    def stream(self, request: Request, /) -> Stream:
        self.requests.append(request)
        return Stream(next(self.responses))


def _response(*parts: Part, usage: Usage = _DEFAULT_USAGE) -> Response:
    stop = "tool_call" if any(isinstance(part, ToolCall) for part in parts) else "stop"
    return Response(parts=parts, usage=usage, stop_reason=stop)


@pytest.mark.asyncio
async def test_run_uses_response_directly_as_history() -> None:
    first = _response("done", usage=Usage(3, 2))
    result = await run(Model([first]), "question")
    assert result.response is first
    assert result.history[-1] is first
    assert result.require_text() == "done"
    assert result.usage == Usage(3, 2)
    assert result.turns == 1


@pytest.mark.asyncio
async def test_run_executes_tools_and_replays_results_in_call_order() -> None:
    @tool
    async def identity(value: int) -> int:
        return value

    calls = (
        ToolCall("identity", '{"value":0}', call_id="0"),
        ToolCall("identity", '{"value":1}', call_id="1"),
    )
    model = Model([_response(*calls), _response("done")])
    result = await run(
        model,
        "question",
        tools=(identity,),
        limits=RunLimits(parallel_tools=2),
    )
    assert [value.call.call_id for value in result.tool_results] == ["0", "1"]
    assert result.require_text() == "done"


@pytest.mark.asyncio
async def test_stream_exposes_live_tool_lifecycle() -> None:
    gate = asyncio.Event()

    @tool
    async def wait_tool() -> str:
        await gate.wait()
        return "ok"

    call = ToolCall("wait_tool", call_id="1")
    first = _response(call)
    final = _response("done")
    stream = run_stream(Streaming([first, final]), "question", tools=(wait_tool,))

    async with stream:
        iterator = stream.__aiter__()
        assert await anext(iterator) == RunEvent(0, call)
        assert await anext(iterator) == RunEvent(0, first)

        started = await anext(iterator)
        assert isinstance(started.item, ToolStarted)
        assert started.item.call is call

        gate.set()
        finished = await anext(iterator)
        assert isinstance(finished.item, ToolFinished)
        assert finished.item.result.call is call

        assert await anext(iterator) == RunEvent(1, "done")
        assert await anext(iterator) == RunEvent(1, final)

        with pytest.raises(StopAsyncIteration):
            await anext(iterator)

        result = stream.result()

    assert result.response is final
    assert result.tool_results[0].call is call


class BlockingStream:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.closed = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.closed = True

    async def _iterate(self) -> AsyncIterator[Part]:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        yield "unreachable"

    def __aiter__(self) -> AsyncIterator[Part]:
        return self._iterate()

    def response(self) -> Response:
        raise RuntimeError("not complete")


class BlockingModel:
    def __init__(self) -> None:
        self.active = BlockingStream()

    async def generate(self, request: Request, /) -> Response:
        raise AssertionError("generate should not be used by run_stream")

    def stream(self, request: Request, /) -> BlockingStream:
        return self.active


@pytest.mark.asyncio
async def test_stream_turn_timeout_interrupts_blocked_next_part() -> None:
    model = BlockingModel()
    stream = run_stream(model, "question", limits=RunLimits(turn_timeout=0.01))

    async with stream:
        with pytest.raises(RunTimeoutError):
            _ = [event async for event in stream]

    assert model.active.cancelled.is_set()
    assert model.active.closed


@pytest.mark.asyncio
async def test_zero_tool_call_limit_is_valid_and_enforced() -> None:
    @tool
    async def identity(value: int) -> int:
        return value

    call = ToolCall("identity", '{"value":1}')
    model = Model([_response(call)])

    with pytest.raises(RunLimitError) as captured:
        await run(model, "question", tools=(identity,), limits=RunLimits(tool_calls=0))

    assert captured.value.kind == "tool calls"
    assert captured.value.actual == 1


@pytest.mark.asyncio
async def test_turn_limit_reports_required_next_turn() -> None:
    @tool
    async def identity(value: int) -> int:
        return value

    call = ToolCall("identity", '{"value":1}')
    model = Model([_response(call)])

    with pytest.raises(RunLimitError) as captured:
        await run(model, "question", tools=(identity,), limits=RunLimits(turns=1))

    assert captured.value.kind == "turns"
    assert captured.value.actual == 2


@pytest.mark.asyncio
async def test_runtime_rejects_invalid_prepare_result() -> None:
    invalid: Any = object()

    def prepare(request: Request) -> Any:
        return invalid

    with pytest.raises(TypeError, match="prepare must return Request"):
        await run(Model([_response("unused")]), "question", prepare=prepare)


@pytest.mark.asyncio
async def test_runtime_rejects_invalid_model_result() -> None:
    invalid: Any = object()

    class InvalidModel:
        async def generate(self, request: Request, /) -> Any:
            return invalid

    model: Any = InvalidModel()
    with pytest.raises(TypeError, match="model must return Response"):
        await run(model, "question")
