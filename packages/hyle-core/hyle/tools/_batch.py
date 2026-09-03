__all__ = (
    "ToolBatchError",
    "ToolFailed",
    "ToolFinished",
    "ToolStarted",
    "execute_tools",
)

from asyncio import Queue, QueueEmpty, Task, TaskGroup
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from time import perf_counter_ns
from typing import final

from hyle._validate import (
    require_bool,
    require_non_negative_int,
    require_positive_float,
    require_positive_int,
)
from hyle.tools._catalog import Tools, catalog
from hyle.tools._errors import ToolError
from hyle.tools._invoke import invoke
from hyle.tools._tool import Tool, ToolCall, ToolResult

_DEFAULT_TIMEOUT = 30.0


@final
@dataclass
class ToolStarted:
    index: int
    call: ToolCall

    def __post_init__(self) -> None:
        require_non_negative_int(self.index, "tool call index")


@final
@dataclass(frozen=True, slots=True)
class ToolFinished:
    index: int
    result: ToolResult
    duration_ns: int

    def __post_init__(self) -> None:
        require_non_negative_int(self.index, "tool call index")
        require_non_negative_int(self.duration_ns, "tool duration")


@final
@dataclass(frozen=True, slots=True)
class ToolFailed:
    index: int
    call: ToolCall
    error: Exception
    duration_ns: int

    def __post_init__(self) -> None:
        require_non_negative_int(self.index, "tool call index")
        require_non_negative_int(self.duration_ns, "tool duration")


type ToolEvent = ToolStarted | ToolFinished | ToolFailed
type ToolOutcome = ToolFinished | ToolFailed


class ToolBatchError(ToolError):
    __slots__ = ("outcomes", "unstarted")

    def __init__(
        self,
        outcomes: Sequence[ToolOutcome],
        unstarted: Sequence[ToolCall],
        /,
    ) -> None:
        settled = tuple(outcomes)
        failures = tuple(item for item in settled if isinstance(item, ToolFailed))
        if not failures:
            raise ValueError("a tool batch error must contain at least one hard failure")

        self.outcomes = settled
        self.unstarted = tuple(unstarted)
        failed = len(failures)
        started = len(settled)
        unstarted_count = len(self.unstarted)
        failed_noun = "call" if failed == 1 else "calls"
        super().__init__(
            f"{failed} tool {failed_noun} failed; {started} started, {unstarted_count} unstarted"
        )

    @property
    def results(self) -> tuple[ToolResult, ...]:
        return tuple(item.result for item in self.outcomes if isinstance(item, ToolFinished))

    @property
    def failures(self) -> tuple[ToolFailed, ...]:
        return tuple(item for item in self.outcomes if isinstance(item, ToolFailed))


async def execute_tools(
    calls: Sequence[ToolCall],
    tools: Tools,
    /,
    *,
    max_parallel: int = 1,
    continue_on_failure: bool = False,
    timeout: float | None = _DEFAULT_TIMEOUT,
) -> tuple[ToolResult, ...]:
    calls = tuple(calls)
    if not calls:
        return ()

    max_parallel = require_positive_int(max_parallel, "max parallel tools")
    continue_on_failure = require_bool(continue_on_failure, "continue on failure")
    timeout = require_positive_float.optional(timeout, "tool timeout")
    tools_by_name = catalog(tools)
    results: list[ToolResult | None] = [None] * len(calls)

    async with aclosing(
        batch_events(
            calls,
            tools_by_name,
            max_parallel=min(max_parallel, len(calls)),
            continue_on_failure=continue_on_failure,
            timeout=timeout,
        )
    ) as events:
        async for event in events:
            if isinstance(event, ToolFinished):
                results[event.index] = event.result

    if any(result is None for result in results):
        raise RuntimeError("successful tool batch is missing a result")
    return tuple(result for result in results if result is not None)


async def batch_events(
    calls: tuple[ToolCall, ...],
    tools: dict[str, Tool],
    /,
    *,
    max_parallel: int,
    continue_on_failure: bool,
    timeout: float | None,
) -> AsyncGenerator[ToolEvent]:
    outcomes: list[ToolOutcome | None] = [None] * len(calls)
    settled: Queue[Task[ToolOutcome]] = Queue()
    active: dict[Task[ToolOutcome], int] = {}
    next_index = 0
    stopped = False

    def done(task: Task[ToolOutcome], /) -> None:
        settled.put_nowait(task)

    def admit(group: TaskGroup) -> tuple[ToolStarted, ...]:
        nonlocal next_index
        events: list[ToolStarted] = []
        while (
            len(active) < max_parallel
            and next_index < len(calls)
            and (continue_on_failure or not stopped)
        ):
            index = next_index
            call = calls[index]
            next_index += 1
            started_ns = perf_counter_ns()
            task = group.create_task(
                _settle(index, call, tools, started_ns, timeout),
                name=f"hyle-tool:{index}",
                eager_start=False,
            )
            active[task] = index
            task.add_done_callback(done)
            events.append(ToolStarted(index, call))
        return tuple(events)

    async with TaskGroup() as group:
        for event in admit(group):
            yield event

        while active:
            ready = [await settled.get()]
            while True:
                try:
                    ready.append(settled.get_nowait())
                except QueueEmpty:
                    break

            lifecycle: list[ToolOutcome] = []
            for task in ready:
                index = active.pop(task)
                outcome = task.result()
                outcomes[index] = outcome
                lifecycle.append(outcome)
                if isinstance(outcome, ToolFailed) and not continue_on_failure:
                    stopped = True

            for event in lifecycle:
                yield event
            for event in admit(group):
                yield event

    started = next_index
    ordered = outcomes[:started]
    if any(outcome is None for outcome in ordered):
        raise RuntimeError("started tool call did not settle")

    normalized = tuple(outcome for outcome in ordered if outcome is not None)
    failures = tuple(item for item in normalized if isinstance(item, ToolFailed))
    if failures:
        error = ToolBatchError(normalized, calls[started:])
        causes = [failure.error for failure in failures]
        cause: BaseException = (
            causes[0]
            if len(causes) == 1
            else ExceptionGroup("tool execution failures", causes)
        )  # fmt: skip
        raise error from cause


async def _settle(
    index: int,
    call: ToolCall,
    tools: dict[str, Tool],
    started_ns: int,
    timeout: float | None,
    /,
) -> ToolOutcome:
    try:
        result = await invoke(call, tools, timeout=timeout)
    except Exception as error:
        error.add_note(f"while executing tool {call.name!r} at batch index {index}")
        return ToolFailed(index, call, error, perf_counter_ns() - started_ns)
    return ToolFinished(index, result, perf_counter_ns() - started_ns)
