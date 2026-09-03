__all__ = ("Part", "PartBuffer", "Reasoning", "Response", "StopReason")

from collections.abc import Buffer, Hashable
from dataclasses import dataclass
from typing import Literal, TypeIs, final, override

from hyle._validate import require_label, require_utf8_str
from hyle.content import Content
from hyle.tools import ToolCall
from hyle.usage import Usage

type StopReason = Literal["stop", "length", "tool_call", "content_filter", "error", "other"]
type Part = str | Content | Reasoning | ToolCall

_STOP_REASONS: tuple[StopReason, ...] = (
    "stop",
    "length",
    "tool_call",
    "content_filter",
    "error",
    "other",
)


@final
@dataclass(frozen=True, slots=True, init=False, repr=False)
class Reasoning:
    text: str | None
    state: bytes | None
    scope: str | None

    def __init__(
        self,
        text: str | None = None,
        /,
        *,
        state: Buffer | None = None,
        scope: str | None = None,
    ) -> None:
        if text is not None:
            text = require_utf8_str(text, "reasoning text")
        if (state is None) != (scope is None):
            raise ValueError("reasoning state and scope must be provided together")

        owned: bytes | None = None
        if state is not None:
            if isinstance(state, bytes):
                owned = state
            else:
                with memoryview(state) as view:
                    owned = view.tobytes()
            if not owned:
                raise ValueError("reasoning state must not be empty")
            scope = require_label(scope, "reasoning scope")

        if text is None and owned is None:
            raise ValueError("reasoning must contain text or provider state")

        object.__setattr__(self, "text", text)
        object.__setattr__(self, "state", owned)
        object.__setattr__(self, "scope", scope)

    @override
    def __repr__(self) -> str:
        state_size = None if self.state is None else len(self.state)
        return f"Reasoning(text={self.text!r}, scope={self.scope!r}, state_size={state_size!r})"


@dataclass(frozen=True, slots=True, kw_only=True)
class Response:
    parts: tuple[Part, ...]
    usage: Usage
    stop_reason: StopReason
    provider: str | None = None
    model: str | None = None
    id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "usage", _usage(self.usage))
        object.__setattr__(self, "stop_reason", _stop_reason(self.stop_reason))
        object.__setattr__(self, "parts", tuple(_part(part) for part in self.parts))
        object.__setattr__(
            self,
            "provider",
            require_utf8_str.optional(self.provider, "response provider"),
        )
        object.__setattr__(
            self,
            "model",
            require_utf8_str.optional(self.model, "response model"),
        )
        object.__setattr__(
            self,
            "id",
            require_utf8_str.optional(self.id, "response ID"),
        )

    @property
    def text(self) -> str | None:
        chunks = [part for part in self.parts if isinstance(part, str)]
        return "".join(chunks) if chunks else None

    def require_text(self) -> str:
        text = self.text
        if text is None or not text.strip():
            raise ValueError("response contains no non-blank text")
        return text

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return tuple(part for part in self.parts if isinstance(part, ToolCall))


@final
class _TextRun:
    __slots__ = ("chunks",)

    def __init__(self, text: str, /) -> None:
        self.chunks = [text]

    def extend(self, text: str, /) -> None:
        self.chunks.append(text)

    def finish(self) -> str:
        return "".join(self.chunks)


@final
class _ReasoningRun:
    __slots__ = ("chunks", "scope", "state")

    def __init__(self, reasoning: Reasoning, /) -> None:
        self.chunks = [] if reasoning.text is None else [reasoning.text]
        self.state = reasoning.state
        self.scope = reasoning.scope

    def extend(self, reasoning: Reasoning, /) -> None:
        if reasoning.state is not None:
            incoming = (reasoning.scope, reasoning.state)
            current = (self.scope, self.state)
            if self.state is not None and current != incoming:
                raise ValueError("a streamed reasoning run changed provider state")
            self.scope, self.state = incoming
        if reasoning.text is not None:
            self.chunks.append(reasoning.text)

    def finish(self) -> Reasoning:
        text = "".join(self.chunks) if self.chunks else None
        return Reasoning(text, state=self.state, scope=self.scope)


type _Buffered = _TextRun | _ReasoningRun | Content | ToolCall


@final
class PartBuffer:
    __slots__ = ("_finished", "_keys", "_parts", "_tail_keyed")

    def __init__(self) -> None:
        self._parts: list[_Buffered] = []
        self._keys: dict[Hashable, _Buffered] = {}
        self._tail_keyed = False
        self._finished: tuple[Part, ...] | None = None

    def append(self, part: Part, /, *, key: Hashable | None = None) -> None:
        if self._finished is not None:
            raise RuntimeError("a finished part buffer cannot be modified")
        part = _part(part)
        if key is None:
            self._append_adjacent(part)
        else:
            self._append_keyed(key, part)

    def finish(self) -> tuple[Part, ...]:
        if self._finished is None:
            self._finished = tuple(_finish(part) for part in self._parts)
            self._parts.clear()
            self._keys.clear()
        return self._finished

    def _append_keyed(self, key: Hashable, part: Part, /) -> None:
        current = self._keys.get(key)
        if current is not None:
            if isinstance(current, _TextRun) and isinstance(part, str):
                current.extend(part)
                return
            if isinstance(current, _ReasoningRun) and isinstance(part, Reasoning):
                current.extend(part)
                return
            raise ValueError("a provider stream reused a part key for an incompatible part")

        buffered = _buffer(part)
        self._keys[key] = buffered
        self._parts.append(buffered)
        self._tail_keyed = True

    def _append_adjacent(self, part: Part, /) -> None:
        if not self._tail_keyed and self._parts:
            previous = self._parts[-1]
            if isinstance(previous, _TextRun) and isinstance(part, str):
                previous.extend(part)
                return
            if isinstance(previous, _ReasoningRun) and isinstance(part, Reasoning):
                previous.extend(part)
                return

        self._parts.append(_buffer(part))
        self._tail_keyed = False


def _usage(value: object, /) -> Usage:
    if isinstance(value, Usage):
        return value
    raise TypeError("usage must be Usage")


def _stop_reason(value: object, /) -> StopReason:
    if _is_stop_reason(value):
        return value
    raise ValueError(f"invalid stop reason {value!r}")


def _is_stop_reason(value: object, /) -> TypeIs[StopReason]:
    return value in _STOP_REASONS


def _part(value: object, /) -> Part:
    if isinstance(value, str):
        return require_utf8_str(value, "response text")
    if isinstance(value, Content):
        if type(value) is Content:
            raise TypeError("Content is an extension base and cannot be used directly")
        return value
    if isinstance(value, Reasoning | ToolCall):
        return value
    raise TypeError("response parts must be strings, Content, Reasoning, or ToolCall values")


def _buffer(part: Part, /) -> _Buffered:
    if isinstance(part, str):
        return _TextRun(part)
    if isinstance(part, Reasoning):
        return _ReasoningRun(part)
    return part


def _finish(part: _Buffered, /) -> Part:
    if isinstance(part, _TextRun | _ReasoningRun):
        return part.finish()
    return part
