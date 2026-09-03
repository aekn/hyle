__all__ = ("Tool", "ToolCall", "ToolResult", "ToolSpec")

from dataclasses import dataclass
from typing import Protocol, Self, final, override, runtime_checkable

from hyle._json import JsonInput, encode_json, normalize_json, normalize_json_object
from hyle._validate import require_bool, require_non_blank_utf8_str, require_utf8_str
from hyle.content import Content
from hyle.schema import JsonSchema


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ToolSpec:
    name: str
    input_schema: JsonSchema
    description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_schema", _input_schema(self.input_schema))
        description = self.description
        if description is not None:
            description = require_non_blank_utf8_str(description, "tool description").strip()
        object.__setattr__(self, "name", require_tool_name(self.name))
        object.__setattr__(self, "description", description)


@final
@dataclass(frozen=True, slots=True, init=False, repr=False)
class ToolCall:
    name: str
    arguments_json: bytes
    call_id: str | None

    def __init__(
        self,
        name: str,
        arguments_json: JsonInput = b"{}",
        /,
        *,
        call_id: str | None = None,
    ) -> None:
        try:
            arguments = normalize_json_object(arguments_json)
        except (TypeError, ValueError) as error:
            raise ValueError("tool arguments must be a valid strict JSON object") from error

        object.__setattr__(self, "name", require_tool_name(name))
        object.__setattr__(self, "arguments_json", arguments)
        object.__setattr__(self, "call_id", require_utf8_str.optional(call_id, "tool call ID"))

    @property
    def arguments_size(self) -> int:
        return len(self.arguments_json)

    @override
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, arguments_size={self.arguments_size}, "
            f"call_id={self.call_id!r})"
        )


@final
@dataclass(frozen=True, slots=True, init=False, repr=False)
class ToolResult:
    call: ToolCall
    parts: tuple[str | Content, ...]
    structured_json: bytes | None
    is_error: bool

    def __init__(
        self,
        call: ToolCall,
        /,
        *parts: str | Content,
        structured_json: JsonInput | None = None,
        is_error: bool = False,
    ) -> None:
        object.__setattr__(self, "call", _call(call))
        object.__setattr__(self, "parts", tuple(_part(part) for part in parts))
        object.__setattr__(self, "structured_json", _structured(structured_json))
        object.__setattr__(self, "is_error", require_bool(is_error, "tool result error marker"))

    @classmethod
    def json(cls, call: ToolCall, value: object, /, *, is_error: bool = False) -> Self:
        call = _call(call)
        result = cls.__new__(cls)
        object.__setattr__(result, "call", call)
        object.__setattr__(result, "parts", ())
        try:
            encoded = encode_json(value)
        except (TypeError, ValueError) as error:
            raise ValueError("tool structured result must be valid JSON") from error
        object.__setattr__(result, "structured_json", encoded)
        object.__setattr__(result, "is_error", require_bool(is_error, "tool result error marker"))
        return result

    @classmethod
    def error(cls, call: ToolCall, message: str, /) -> Self:
        return cls(call, message, is_error=True)

    @property
    def text(self) -> str | None:
        chunks = [part for part in self.parts if isinstance(part, str)]
        return "".join(chunks) if chunks else None

    @override
    def __repr__(self) -> str:
        size = None if self.structured_json is None else len(self.structured_json)
        return (
            f"{type(self).__name__}(call={self.call!r}, parts={len(self.parts)}, "
            f"structured_size={size!r}, is_error={self.is_error})"
        )


@runtime_checkable
class Tool(Protocol):
    """Executable tool capability."""

    @property
    def spec(self) -> ToolSpec: ...

    async def execute(self, call: ToolCall, /) -> ToolResult: ...


def _input_schema(value: object, /) -> JsonSchema:
    if isinstance(value, JsonSchema):
        return value
    raise TypeError("input_schema must be JsonSchema")


def _call(value: object, /) -> ToolCall:
    if isinstance(value, ToolCall):
        return value
    raise TypeError("call must be ToolCall")


def require_tool_name(value: str, /) -> str:
    return require_non_blank_utf8_str(value, "tool name").strip()


def _part(value: object, /) -> str | Content:
    if isinstance(value, str):
        return require_utf8_str(value, "tool result text")
    if isinstance(value, Content) and type(value) is not Content:
        return value
    raise TypeError("tool result parts must be strings or Content values")


def _structured(value: JsonInput | None, /) -> bytes | None:
    if value is None:
        return None
    try:
        return normalize_json(value)
    except (TypeError, ValueError) as error:
        raise ValueError("tool structured result must be valid strict JSON") from error
