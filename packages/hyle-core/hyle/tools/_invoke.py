__all__ = ("execute_tool",)

from asyncio import get_running_loop, timeout_at

from hyle._validate import require_positive_float
from hyle.tools._catalog import Tools, catalog
from hyle.tools._errors import ToolArgumentError, ToolOutputError, ToolTimeoutError
from hyle.tools._tool import Tool, ToolCall, ToolResult

_DEFAULT_TIMEOUT = 30.0
_MAX_UNKNOWN_TOOL_SUGGESTIONS = 32


async def execute_tool(
    call: ToolCall,
    tools: Tools,
    /,
    *,
    timeout: float | None = _DEFAULT_TIMEOUT,
) -> ToolResult:
    timeout = require_positive_float.optional(timeout, "tool timeout")
    return await invoke(call, catalog(tools), timeout=timeout)


async def invoke(
    call: ToolCall,
    tools: dict[str, Tool],
    /,
    *,
    timeout: float | None,
) -> ToolResult:
    try:
        tool = tools.get(call.name)
        if tool is None:
            return _unknown(call, tools)

        try:
            result = await _invoke(call, tool, timeout)
        except ToolArgumentError as error:
            if error.name != call.name:
                raise
            return _invalid_arguments(call, error)

        if result.call != call:
            raise ToolOutputError(call.name, detail="returned a result for a different tool call")
        return result
    except Exception as error:
        note = f"while executing tool {call.name!r}"
        if call.call_id is not None:
            note += f" (call_id={call.call_id!r})"
        error.add_note(note)
        raise


async def _invoke(call: ToolCall, tool: Tool, timeout: float | None, /) -> ToolResult:
    if timeout is None:
        return await tool.execute(call)

    loop = get_running_loop()
    when = loop.time() + timeout
    scope = timeout_at(when)

    try:
        async with scope:
            result = await tool.execute(call)
    except TimeoutError:
        if not scope.expired():
            raise
        raise ToolTimeoutError(call, timeout) from None

    if scope.expired() or loop.time() >= when:
        raise ToolTimeoutError(call, timeout)
    return result


def _unknown(call: ToolCall, tools: dict[str, Tool], /) -> ToolResult:
    names = tuple(tools)[:_MAX_UNKNOWN_TOOL_SUGGESTIONS]
    total = len(tools)
    if names:
        available = ", ".join(repr(name) for name in names)
        suffix = "" if len(names) == total else f"; {total - len(names)} more"
        message = (
            f"The requested tool {call.name!r} is unavailable. Available tools: {available}{suffix}"
        )
    else:
        message = f"The requested tool {call.name!r} is unavailable. No tools are available."
    return ToolResult.json(
        call,
        {
            "error": "unknown_tool",
            "tool": call.name,
            "message": message,
            "available_tools": names,
            "available_tool_count": total,
        },
        is_error=True,
    )


def _invalid_arguments(call: ToolCall, error: ToolArgumentError, /) -> ToolResult:
    return ToolResult.json(
        call,
        {
            "error": "invalid_tool_arguments",
            "tool": call.name,
            "message": "The tool arguments were invalid.",
            "issues": [
                {"path": issue.pointer, "message": issue.message, "code": issue.code}
                for issue in error.issues
            ],
        },
        is_error=True,
    )
