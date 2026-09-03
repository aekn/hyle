import asyncio
from typing import Any

import pytest

from hyle import tool
from hyle.schema import JsonSchema
from hyle.tools import (
    ToolBatchError,
    ToolCall,
    ToolFailed,
    ToolFinished,
    ToolResult,
    ToolSpec,
    ToolStarted,
    execute_tool,
    execute_tools,
)
from hyle.tools._batch import batch_events
from hyle.tools._catalog import catalog


@tool
def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right


@pytest.mark.asyncio
async def test_execute_tool_returns_model_visible_result_directly() -> None:
    call = ToolCall("add", '{"left":2,"right":3}', call_id="1")
    result = await execute_tool(call, (add,))
    assert result.call is call
    assert result.structured_json == b"5"
    assert not result.is_error


@pytest.mark.asyncio
async def test_unknown_tool_and_bad_arguments_are_recoverable_results() -> None:
    unknown = ToolCall("missing", call_id="1")
    result = await execute_tool(unknown, (add,))
    assert result.is_error

    invalid = ToolCall("add", '{"left":"x","right":3}', call_id="2")
    result = await execute_tool(invalid, (add,))
    assert result.is_error


@pytest.mark.asyncio
async def test_batch_returns_call_order_not_completion_order() -> None:
    gates = [asyncio.Event(), asyncio.Event(), asyncio.Event()]

    @tool
    async def work(index: int) -> int:
        await gates[index].wait()
        return index

    calls = tuple(
        ToolCall("work", f'{{"index":{index}}}', call_id=str(index)) for index in range(3)
    )
    task = asyncio.create_task(execute_tools(calls, (work,), max_parallel=3))
    await asyncio.sleep(0)
    for index in (1, 2, 0):
        gates[index].set()
        await asyncio.sleep(0)
    results = await task
    assert [result.call.call_id for result in results] == ["0", "1", "2"]


@pytest.mark.asyncio
async def test_lifecycle_is_chronological_while_admission_is_bounded() -> None:
    gates = [asyncio.Event(), asyncio.Event(), asyncio.Event()]

    @tool
    async def work(index: int) -> int:
        await gates[index].wait()
        return index

    calls = tuple(
        ToolCall("work", f'{{"index":{index}}}', call_id=str(index)) for index in range(3)
    )
    events = batch_events(
        calls,
        catalog((work,)),
        max_parallel=2,
        continue_on_failure=False,
        timeout=1.0,
    )
    iterator = events.__aiter__()
    assert await anext(iterator) == ToolStarted(0, calls[0])
    assert await anext(iterator) == ToolStarted(1, calls[1])

    gates[1].set()
    settled = await anext(iterator)
    assert isinstance(settled, ToolFinished)
    assert settled.index == 1
    assert await anext(iterator) == ToolStarted(2, calls[2])

    gates[0].set()
    gates[2].set()
    remaining = [event async for event in iterator]
    assert all(isinstance(event, ToolFinished) for event in remaining)


@pytest.mark.asyncio
async def test_hard_failure_stops_admission_and_drains_started_sibling() -> None:
    fail = asyncio.Event()
    release = asyncio.Event()
    sibling_done = asyncio.Event()

    @tool
    async def work(index: int) -> int:
        if index == 0:
            await fail.wait()
            raise RuntimeError("boom")
        if index == 1:
            await release.wait()
            sibling_done.set()
        return index

    calls = tuple(ToolCall("work", f'{{"index":{index}}}') for index in range(3))
    task = asyncio.create_task(execute_tools(calls, (work,), max_parallel=2))
    await asyncio.sleep(0)
    fail.set()
    await asyncio.sleep(0)
    release.set()

    with pytest.raises(ToolBatchError) as captured:
        await task

    assert sibling_done.is_set()
    assert captured.value.unstarted == (calls[2],)
    assert [type(item) for item in captured.value.outcomes] == [ToolFailed, ToolFinished]


def test_tool_values_reject_invalid_semantic_dependencies() -> None:
    invalid: Any = object()

    with pytest.raises(TypeError, match="input_schema"):
        ToolSpec(name="lookup", input_schema=invalid)

    with pytest.raises(TypeError, match="call"):
        ToolResult(invalid, "x")

    schema = JsonSchema({"type": "object"})
    assert ToolSpec(name="lookup", input_schema=schema).input_schema is schema
