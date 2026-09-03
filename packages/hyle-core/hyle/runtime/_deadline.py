from asyncio import get_running_loop, timeout_at
from collections.abc import Awaitable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from types import TracebackType

from hyle._validate import require_positive_float
from hyle.runtime._errors import RunTimeoutError


@dataclass(frozen=True, slots=True)
class Deadline:
    when: float
    seconds: float
    turn: int | None = None


class _EnterBefore[T](AbstractAsyncContextManager[T]):
    __slots__ = ("_context", "_deadline", "_entered")

    def __init__(
        self,
        context: AbstractAsyncContextManager[T],
        deadline: Deadline | None,
        /,
    ) -> None:
        self._context = context
        self._deadline = deadline
        self._entered: T | None = None

    async def __aenter__(self) -> T:
        if self._deadline is None:
            entered = await self._context.__aenter__()
        else:
            try:
                async with timeout_at(self._deadline.when):
                    entered = await self._context.__aenter__()
            except TimeoutError:
                raise _timeout(self._deadline) from None
        self._entered = entered
        return entered

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return await self._context.__aexit__(exc_type, exc, traceback)


def deadline(seconds: float | None, /, *, turn: int | None = None) -> Deadline | None:
    seconds = require_positive_float.optional(seconds, "timeout")
    if seconds is None:
        return None
    return Deadline(get_running_loop().time() + seconds, seconds, turn)


def earliest(left: Deadline | None, right: Deadline | None, /) -> Deadline | None:
    if left is None:
        return right
    if right is None:
        return left
    return left if left.when <= right.when else right


async def wait[T](awaitable: Awaitable[T], boundary: Deadline | None, /) -> T:
    if boundary is None:
        return await awaitable
    try:
        async with timeout_at(boundary.when):
            return await awaitable
    except TimeoutError:
        raise _timeout(boundary) from None


def enter[T](
    context: AbstractAsyncContextManager[T],
    boundary: Deadline | None,
    /,
) -> AbstractAsyncContextManager[T]:
    return _EnterBefore(context, boundary)


def check(boundary: Deadline | None, /) -> None:
    if boundary is not None and get_running_loop().time() >= boundary.when:
        raise _timeout(boundary)


def _timeout(boundary: Deadline, /) -> RunTimeoutError:
    return RunTimeoutError(boundary.seconds, turn=boundary.turn)
