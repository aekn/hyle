__all__ = ("Model", "Stream", "StreamingModel")

from collections.abc import AsyncIterator
from types import TracebackType
from typing import Protocol, Self

from hyle.generation._request import Request
from hyle.generation._response import Part, Response


class Model(Protocol):
    async def generate(self, request: Request, /) -> Response: ...


class Stream(Protocol):
    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def __aiter__(self) -> AsyncIterator[Part]: ...

    def response(self) -> Response: ...


class StreamingModel(Model, Protocol):
    def stream(self, request: Request, /) -> Stream: ...
