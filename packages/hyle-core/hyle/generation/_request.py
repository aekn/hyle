from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeIs, final

from hyle._validate import require_positive_int
from hyle.generation._msg import Msg
from hyle.generation._response import Response
from hyle.schema import JsonSchema
from hyle.tools import Tool, ToolResult, ToolSpec

type _Item = Msg | Response | ToolResult
type _Input = str | _Item | Sequence[_Item]
type _ToolChoice = Literal["auto", "required"] | ToolSpec


@final
@dataclass(frozen=True, slots=True, init=False)
class Request:
    input: tuple[_Item, ...]
    tools: tuple[ToolSpec, ...]
    tool_choice: _ToolChoice
    schema: JsonSchema | None
    max_output_tokens: int | None

    def __init__(
        self,
        input: _Input,
        *,
        instructions: str | Msg | None = None,
        tools: Iterable[Tool | ToolSpec] = (),
        tool_choice: _ToolChoice | Tool = "auto",
        schema: JsonSchema | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        items = _input(input)
        if instructions is not None:
            system = _instructions(instructions)
            if system.role != "system":
                raise ValueError("instructions must be a system message")
            items = (system, *items)

        schema = _schema(schema)
        specs = tuple(_spec(tool) for tool in tools)
        _unique_tools(specs)

        object.__setattr__(self, "input", items)
        object.__setattr__(self, "tools", specs)
        object.__setattr__(self, "tool_choice", _choice(tool_choice, specs))
        object.__setattr__(self, "schema", schema)
        object.__setattr__(
            self,
            "max_output_tokens",
            require_positive_int.optional(max_output_tokens, "max output tokens"),
        )


def _instructions(value: object, /) -> Msg:
    if isinstance(value, str):
        return Msg.system(value)
    if isinstance(value, Msg):
        return value
    raise TypeError("instructions must be text or Msg")


def _schema(value: object, /) -> JsonSchema | None:
    if value is None or isinstance(value, JsonSchema):
        return value
    raise TypeError("schema must be JsonSchema")


def _input(value: object, /) -> tuple[_Item, ...]:
    if isinstance(value, str):
        return (Msg.user(value),)
    if isinstance(value, Msg | Response | ToolResult):
        return (value,)

    if not _is_sequence(value):
        raise TypeError(
            "request input must be text, a history value, or a sequence of history values"
        )
    items: list[_Item] = []
    for item in value:
        if not isinstance(item, Msg | Response | ToolResult):
            raise TypeError("request input must contain Msg, Response, or ToolResult values")
        items.append(item)
    return tuple(items)


def _spec(value: object, /) -> ToolSpec:
    if isinstance(value, ToolSpec):
        return value
    if isinstance(value, Tool):
        return value.spec
    raise TypeError("tools must contain Tool or ToolSpec values")


def _unique_tools(tools: tuple[ToolSpec, ...], /) -> None:
    names: set[str] = set()
    for tool in tools:
        if tool.name in names:
            raise ValueError(f"request contains duplicate tool name {tool.name!r}")
        names.add(tool.name)


def _choice(value: _ToolChoice | Tool, tools: tuple[ToolSpec, ...], /) -> _ToolChoice:
    if isinstance(value, str):
        if value not in {"auto", "required"}:
            raise ValueError("tool choice must be 'auto', 'required', a Tool, or a ToolSpec")
        if value == "required" and not tools:
            raise ValueError("required tool choice needs at least one declared tool")
        return value

    selected = _spec(value)
    if not any(tool.name == selected.name for tool in tools):
        raise ValueError(f"selected tool {selected.name!r} is not declared by this request")
    return selected


def _is_sequence(value: object, /) -> TypeIs[Sequence[object]]:
    return isinstance(value, Sequence)
