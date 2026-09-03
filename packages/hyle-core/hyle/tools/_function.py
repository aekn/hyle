__all__ = ("FunctionTool", "tool")

from asyncio import to_thread
from collections.abc import Callable
from inspect import isawaitable, iscoroutinefunction
from typing import final, overload, override

from hyle._validate import require_bool
from hyle.tools._function_schema import compile_function
from hyle.tools._tool import ToolCall, ToolResult, ToolSpec


@final
class FunctionTool[**P, ResultT]:
    __slots__ = ("_compiled", "_function", "_is_async")

    def __init__(
        self,
        function: Callable[P, ResultT],
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool = True,
    ) -> None:
        self._function = function
        self._is_async = _is_async_callable(function)
        self._compiled = compile_function(
            function,
            name=name,
            description=description,
            strict=require_bool(strict, "function tool strict mode"),
        )

    @property
    def spec(self) -> ToolSpec:
        return self._compiled.spec

    @property
    def __wrapped__(self) -> Callable[P, ResultT]:
        return self._function

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> ResultT:
        return self._function(*args, **kwargs)

    async def execute(self, call: ToolCall, /) -> ToolResult:
        args, kwargs = self._compiled.read_arguments(call)
        if self._is_async:
            result = self._function(*args, **kwargs)
        else:
            result = await to_thread(self._function, *args, **kwargs)
        value = await result if isawaitable(result) else result
        return self._compiled.write_result(value, call)

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.spec.name!r})"


@overload
def tool[**P, ResultT](
    function: Callable[P, ResultT],
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    strict: bool = True,
) -> FunctionTool[P, ResultT]: ...


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    strict: bool = True,
) -> Callable[[Callable[..., object]], FunctionTool[..., object]]: ...


def tool(
    function: Callable[..., object] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    strict: bool = True,
) -> FunctionTool[..., object] | Callable[[Callable[..., object]], FunctionTool[..., object]]:
    def decorate(function: Callable[..., object], /) -> FunctionTool[..., object]:
        return FunctionTool(function, name=name, description=description, strict=strict)

    return decorate if function is None else decorate(function)


def _is_async_callable(function: object, /) -> bool:
    return iscoroutinefunction(function) or (
        callable(function) and iscoroutinefunction(function.__call__)
    )
