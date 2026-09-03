__all__ = ("RunStream", "run", "run_stream")

from asyncio import CancelledError, Queue, Task, create_task
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import aclosing, suppress
from dataclasses import dataclass, replace
from inspect import isawaitable
from types import TracebackType
from typing import Literal, Self

from hyle.generation import Model, Msg, Request, Response, StreamingModel
from hyle.runtime._deadline import Deadline, check, deadline, earliest, enter, wait
from hyle.runtime._errors import RunLimitError, ToolChoiceError
from hyle.runtime._events import RunEvent, RunResult
from hyle.runtime._limits import RunLimits
from hyle.tools import (
    Tool,
    ToolCall,
    ToolFailed,
    ToolFinished,
    ToolResult,
    ToolSpec,
    ToolStarted,
    execute_tools,
)
from hyle.tools._batch import ToolEvent, batch_events
from hyle.tools._catalog import Tools, catalog
from hyle.usage import Usage

type _HistoryItem = Msg | Response | ToolResult
type _Input = str | _HistoryItem | Sequence[_HistoryItem]
type _Prepare = Callable[[Request], Request | Awaitable[Request]]
type _Choice = Literal["auto", "required"] | ToolSpec

_DEFAULT_LIMITS = RunLimits()


@dataclass(slots=True)
class _State:
    history: list[_HistoryItem]
    start: int
    usage: Usage
    tool_calls: int
    choice: _Choice
    result: RunResult | None = None


async def run(
    model: Model,
    request: Request | _Input,
    /,
    *,
    tools: Tools = (),
    limits: RunLimits = _DEFAULT_LIMITS,
    prepare: _Prepare | None = None,
) -> RunResult:
    base, executable = _base_request(request, tools)
    state = _state(base)

    async with aclosing(
        _drive(
            model,
            base,
            executable,
            limits,
            prepare,
            state,
            stream=None,
        )
    ) as driver:
        async for _ in driver:
            raise RuntimeError("non-streaming runtime emitted an event")

    if state.result is None:
        raise RuntimeError("run finished without a result")
    return state.result


class RunStream:
    __slots__ = (
        "_executable",
        "_iterator",
        "_limits",
        "_model",
        "_prepare",
        "_request",
        "_result",
        "_state",
    )

    def __init__(
        self,
        model: StreamingModel,
        request: Request | _Input,
        /,
        *,
        tools: Tools,
        limits: RunLimits,
        prepare: _Prepare | None,
    ) -> None:
        base, executable = _base_request(request, tools)
        self._model = model
        self._request = base
        self._executable = executable
        self._limits = limits
        self._prepare = prepare
        self._state: Literal["new", "open", "iterating", "done", "failed", "closed"] = "new"
        self._iterator: AsyncGenerator[RunEvent] | None = None
        self._result: RunResult | None = None

    async def __aenter__(self) -> Self:
        if self._state != "new":
            raise RuntimeError("run stream can only be entered once")
        self._state = "open"
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        iterator = self._iterator
        if iterator is not None:
            await iterator.aclose()
        if self._state not in {"done", "failed"}:
            self._state = "closed"

    def __aiter__(self) -> AsyncGenerator[RunEvent]:
        if self._state != "open":
            raise RuntimeError("run stream must be entered and can only be iterated once")
        self._state = "iterating"
        iterator = self._iterate()
        self._iterator = iterator
        return iterator

    async def _iterate(self) -> AsyncGenerator[RunEvent]:
        state = _state(self._request)
        try:
            async with aclosing(
                _drive(
                    self._model,
                    self._request,
                    self._executable,
                    self._limits,
                    self._prepare,
                    state,
                    stream=self._model,
                )
            ) as driver:
                async for event in driver:
                    yield event
        except BaseException:
            self._state = "failed"
            raise
        else:
            if state.result is None:
                raise RuntimeError("run stream finished without a result")
            self._result = state.result
            self._state = "done"

    def result(self) -> RunResult:
        if self._state != "done" or self._result is None:
            raise RuntimeError("run result is available only after successful stream completion")
        return self._result


def run_stream(
    model: StreamingModel,
    request: Request | _Input,
    /,
    *,
    tools: Tools = (),
    limits: RunLimits = _DEFAULT_LIMITS,
    prepare: _Prepare | None = None,
) -> RunStream:
    return RunStream(model, request, tools=tools, limits=limits, prepare=prepare)


async def _drive(
    model: Model,
    request: Request,
    tools: dict[str, Tool],
    limits: RunLimits,
    prepare: _Prepare | None,
    state: _State,
    /,
    *,
    stream: StreamingModel | None,
) -> AsyncGenerator[RunEvent]:
    run_deadline = deadline(limits.timeout)

    for turn in range(limits.turns):
        check(run_deadline)
        turn_deadline = earliest(run_deadline, deadline(limits.turn_timeout, turn=turn))

        current = replace(request, input=tuple(state.history), tool_choice=state.choice)
        if prepare is not None:
            prepared = prepare(current)
            prepared = await wait(prepared, turn_deadline) if isawaitable(prepared) else prepared
            current = _prepared(prepared)
        _validate_tools(current, tools)
        check(turn_deadline)

        if stream is None:
            response = await wait(model.generate(current), turn_deadline)
        else:
            generation = stream.stream(current)
            async with enter(generation, turn_deadline) as active:
                iterator = active.__aiter__()
                while True:
                    try:
                        part = await wait(anext(iterator), turn_deadline)
                    except StopAsyncIteration:
                        break
                    yield RunEvent(turn, part)
                response = active.response()

        response = _response(response)
        check(turn_deadline)
        state.history.append(response)
        state.usage += response.usage
        _check_usage(state, response, limits, turn)

        if stream is not None:
            yield RunEvent(turn, response)

        calls = response.tool_calls
        state.choice = _next_choice(current.tool_choice, calls, turn)

        if not calls:
            state.result = RunResult(tuple(state.history), state.start)
            return

        _reserve_tools(state, len(calls), limits, turn)

        if stream is not None:
            results: list[ToolResult | None] = [None] * len(calls)
            async with aclosing(
                _observed_tools(
                    calls,
                    tools,
                    limits,
                    turn_deadline,
                    results,
                )
            ) as events:
                async for event in events:
                    yield RunEvent(turn, event)

            if any(result is None for result in results):
                raise RuntimeError("successful tool batch is missing a result")
            ordered = tuple(result for result in results if result is not None)
        else:
            ordered = await wait(
                execute_tools(
                    calls,
                    tools,
                    max_parallel=limits.parallel_tools,
                    timeout=limits.tool_timeout,
                ),
                turn_deadline,
            )

        state.history.extend(ordered)
        check(turn_deadline)

    check(run_deadline)
    raise RunLimitError("turns", limits.turns, limits.turns)


async def _observed_tools(
    calls: tuple[ToolCall, ...],
    tools: dict[str, Tool],
    limits: RunLimits,
    boundary: Deadline | None,
    results: list[ToolResult | None],
    /,
) -> AsyncGenerator[ToolEvent]:
    queue: Queue[ToolEvent | object] = Queue()
    done = object()

    async def produce() -> None:
        try:

            async def execute() -> None:
                async with aclosing(
                    batch_events(
                        calls,
                        tools,
                        max_parallel=min(limits.parallel_tools, len(calls)),
                        continue_on_failure=False,
                        timeout=limits.tool_timeout,
                    )
                ) as events:
                    async for event in events:
                        if isinstance(event, ToolFinished):
                            results[event.index] = event.result
                        queue.put_nowait(event)

            await wait(execute(), boundary)
        finally:
            queue.put_nowait(done)

    producer = create_task(produce(), name="hyle-tool-batch")
    try:
        while True:
            item = await queue.get()
            if item is done:
                break
            if not isinstance(item, ToolStarted | ToolFinished | ToolFailed):
                raise RuntimeError("tool event queue contained an invalid value")
            yield item
        await producer
    finally:
        await _close(producer)


async def _close(task: Task[None], /) -> None:
    if not task.done():
        task.cancel()
    with suppress(CancelledError, Exception):
        await task


def _prepared(value: object, /) -> Request:
    if isinstance(value, Request):
        return value
    raise TypeError("prepare must return Request")


def _response(value: object, /) -> Response:
    if isinstance(value, Response):
        return value
    raise TypeError("model must return Response")


def _base_request(
    request: Request | _Input,
    tools: Tools,
    /,
) -> tuple[Request, dict[str, Tool]]:
    base = request if isinstance(request, Request) else Request(request)
    executable = catalog(tools)

    if base.tools:
        _validate_tools(base, executable)
    elif executable:
        base = replace(base, tools=tuple(tool.spec for tool in executable.values()))

    return base, executable


def _state(request: Request, /) -> _State:
    history = list(request.input)
    return _State(
        history=history,
        start=len(history),
        usage=Usage(0, 0),
        tool_calls=0,
        choice=request.tool_choice,
    )


def _validate_tools(request: Request, executable: dict[str, Tool], /) -> None:
    for spec in request.tools:
        tool = executable.get(spec.name)
        if tool is None:
            raise ValueError(
                f"request exposes tool {spec.name!r} without an executable implementation"
            )
        if tool.spec != spec:
            raise ValueError(
                f"request tool spec for {spec.name!r} does not match its implementation"
            )


def _next_choice(choice: _Choice, calls: tuple[ToolCall, ...], turn: int, /) -> _Choice:
    if choice == "auto":
        return "auto"

    names = tuple(call.name for call in calls)
    if choice == "required":
        if not calls:
            raise ToolChoiceError(choice, names, turn=turn)
        return "auto"

    if not calls or any(call.name != choice.name for call in calls):
        raise ToolChoiceError(choice, names, turn=turn)
    return "auto"


def _reserve_tools(state: _State, count: int, limits: RunLimits, turn: int, /) -> None:
    if turn + 1 >= limits.turns:
        raise RunLimitError("turns", limits.turns, turn + 2)

    requested = state.tool_calls + count
    if requested > limits.tool_calls:
        raise RunLimitError("tool calls", limits.tool_calls, requested, turn=turn)
    state.tool_calls = requested


def _check_usage(state: _State, response: Response, limits: RunLimits, turn: int, /) -> None:
    _check_limit("input tokens per turn", response.usage.input, limits.input_tokens_per_turn, turn)
    _check_limit("input tokens", state.usage.input, limits.input_tokens, None)
    _check_limit("output tokens", state.usage.output, limits.output_tokens, None)
    _check_limit("total tokens", state.usage.total, limits.total_tokens, None)


def _check_limit(kind: str, actual: int | None, limit: int | None, turn: int | None, /) -> None:
    if limit is None:
        return
    if actual is None or actual > limit:
        raise RunLimitError(kind, limit, actual, turn=turn)
