__all__ = (
    "ToolArgumentError",
    "ToolError",
    "ToolIssue",
    "ToolOutputError",
    "ToolTimeoutError",
)

from collections.abc import Sequence
from dataclasses import dataclass
from typing import final

from hyle._json import json_pointer
from hyle._validate import require_label, require_non_blank_utf8_str
from hyle.errors import HyleError
from hyle.tools._tool import ToolCall

type _IssuePath = tuple[str | int, ...]


@final
@dataclass(frozen=True, slots=True)
class ToolIssue:
    path: _IssuePath
    message: str
    code: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", tuple(_path_part(part) for part in self.path))
        object.__setattr__(
            self,
            "message",
            require_non_blank_utf8_str(self.message, "tool issue message").strip(),
        )
        object.__setattr__(self, "code", require_label(self.code, "tool issue code"))

    @property
    def pointer(self) -> str:
        return json_pointer(self.path)

    @property
    def location(self) -> str:
        return self.pointer or "<root>"


class ToolError(HyleError):
    __slots__ = ()


class ToolArgumentError(ToolError):
    __slots__ = ("issues", "name")

    def __init__(self, name: str, issues: Sequence[ToolIssue], /) -> None:
        normalized = tuple(issues)
        if not normalized:
            raise ValueError("a tool argument error must contain at least one issue")
        self.name = name
        self.issues = normalized
        super().__init__(
            f"arguments for tool {name!r} did not match its contract ({_summary(normalized)})"
        )

    @property
    def issue_count(self) -> int:
        return len(self.issues)


class ToolOutputError(ToolError):
    __slots__ = ("detail", "issues", "name")

    def __init__(
        self,
        name: str,
        issues: Sequence[ToolIssue] = (),
        /,
        *,
        detail: str | None = None,
    ) -> None:
        normalized = tuple(issues)
        if normalized and detail is not None:
            raise ValueError("tool output errors cannot contain both issues and detail")
        if detail is not None:
            detail = require_non_blank_utf8_str(detail, "tool output error detail").strip()
        self.name = name
        self.issues = normalized
        self.detail = detail
        if normalized:
            message = f"failed validation ({_summary(normalized)})"
        elif detail is not None:
            message = detail
        else:
            message = "could not be serialized"
        super().__init__(f"output from tool {name!r} {message}")

    @property
    def issue_count(self) -> int:
        return len(self.issues)


class ToolTimeoutError(ToolError):
    __slots__ = ("call", "seconds")

    def __init__(self, call: ToolCall, seconds: float, /) -> None:
        self.call = call
        self.seconds = seconds
        super().__init__(f"tool {call.name!r} exceeded its {seconds:g}-second timeout")


def _path_part(value: object, /) -> str | int:
    if isinstance(value, bool) or not isinstance(value, str | int):
        raise TypeError("tool issue path components must be strings or integers")
    return value


def _summary(issues: tuple[ToolIssue, ...], /) -> str:
    count = len(issues)
    noun = "issue" if count == 1 else "issues"
    first = issues[0]
    return f"{count} {noun}; {first.location}: {first.message}"
