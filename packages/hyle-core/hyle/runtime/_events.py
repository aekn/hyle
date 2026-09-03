__all__ = ("RunEvent", "RunResult")

from dataclasses import dataclass
from typing import final

from hyle._validate import require_non_negative_int
from hyle.generation import Msg, Part, Response
from hyle.tools import ToolFailed, ToolFinished, ToolResult, ToolStarted
from hyle.usage import Usage

type _HistoryItem = Msg | Response | ToolResult
type _EventItem = Part | Response | ToolStarted | ToolFinished | ToolFailed


@final
@dataclass(frozen=True, slots=True)
class RunEvent:
    turn: int
    item: _EventItem

    def __post_init__(self) -> None:
        require_non_negative_int(self.turn, "turn")


@final
@dataclass(frozen=True, slots=True, init=False)
class RunResult:
    history: tuple[_HistoryItem, ...]
    _start: int

    def __init__(self, history: tuple[_HistoryItem, ...], start: int, /) -> None:
        history = tuple(history)
        start = require_non_negative_int(start, "run history start")
        if start > len(history):
            raise ValueError("run history start exceeds history length")
        if not any(isinstance(history[index], Response) for index in range(start, len(history))):
            raise ValueError("a run result must contain a response produced by this run")
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "_start", start)

    @property
    def response(self) -> Response:
        for index in range(len(self.history) - 1, self._start - 1, -1):
            item = self.history[index]
            if isinstance(item, Response):
                return item
        raise RuntimeError("run result invariant violated: missing response")

    @property
    def responses(self) -> tuple[Response, ...]:
        return tuple(
            item
            for index, item in enumerate(self.history)
            if index >= self._start and isinstance(item, Response)
        )

    @property
    def tool_results(self) -> tuple[ToolResult, ...]:
        return tuple(
            item
            for index, item in enumerate(self.history)
            if index >= self._start and isinstance(item, ToolResult)
        )

    @property
    def usage(self) -> Usage:
        usage = Usage(0, 0)
        for index in range(self._start, len(self.history)):
            item = self.history[index]
            if isinstance(item, Response):
                usage += item.usage
        return usage

    @property
    def turns(self) -> int:
        return sum(
            isinstance(self.history[index], Response)
            for index in range(self._start, len(self.history))
        )

    @property
    def tool_calls(self) -> int:
        return sum(
            isinstance(self.history[index], ToolResult)
            for index in range(self._start, len(self.history))
        )

    @property
    def text(self) -> str | None:
        return self.response.text

    def require_text(self) -> str:
        return self.response.require_text()
