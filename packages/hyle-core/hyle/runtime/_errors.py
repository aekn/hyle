__all__ = ("RunError", "RunLimitError", "RunTimeoutError", "ToolChoiceError")

from hyle.errors import HyleError
from hyle.tools import ToolSpec


class RunError(HyleError):
    """Base class for standard runtime failures."""

    __slots__ = ()


class RunLimitError(RunError):
    """A configured run bound was exceeded or could not be enforced."""

    __slots__ = ("actual", "kind", "limit", "turn")

    def __init__(
        self,
        kind: str,
        limit: int,
        actual: int | None,
        /,
        *,
        turn: int | None = None,
    ) -> None:
        self.kind = kind
        self.limit = limit
        self.actual = actual
        self.turn = turn

        scope = "run" if turn is None else f"turn {turn + 1}"
        if actual is None:
            message = (
                f"{scope} {kind} limit {limit} cannot be enforced because usage is unavailable"
            )
        else:
            message = f"{scope} {kind} limit {limit} exceeded by {actual}"
        super().__init__(message)


class RunTimeoutError(RunError):
    """A run or one of its turns exceeded its deadline."""

    def __init__(self, seconds: float, /, *, turn: int | None = None) -> None:
        self.seconds = seconds
        self.turn = turn
        scope = "run" if turn is None else f"turn {turn + 1}"
        super().__init__(f"{scope} exceeded its {seconds:g}s timeout")


class ToolChoiceError(RunError):
    """A model response violated the request's tool-choice contract."""

    __slots__ = ("choice", "names", "turn")

    def __init__(
        self,
        choice: str | ToolSpec,
        names: tuple[str, ...],
        /,
        *,
        turn: int,
    ) -> None:
        self.choice = choice
        self.names = names
        self.turn = turn
        if choice == "required":
            message = f"turn {turn + 1} required a tool call but the model returned none"
        elif isinstance(choice, ToolSpec):
            message = (
                f"turn {turn + 1} required tool {choice.name!r}; model called {names or 'no tools'}"
            )
        else:
            message = f"turn {turn + 1} violated tool choice {choice!r}"
        super().__init__(message)
